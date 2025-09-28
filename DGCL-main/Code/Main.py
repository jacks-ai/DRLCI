import torch
import torch as t
from matplotlib import pyplot as plt
import matplotlib
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model_sparse import Model
from DataHandler import DataHandler
from Utils.Utils import *
import os
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

import numpy as np
import random
from copy import deepcopy
import time

# Function to set random seed for reproducibility
#通过设置一个固定的种子值，我们可以确保每次实验的初始化和随机过程相同，从而使得实验的结果可重复
def set_seed(seed):
    print(seed)
    # 设置Python随机种子
    random.seed(seed)
    # 设置NumPy随机种子
    np.random.seed(seed)
    # 设置PyTorch随机种子
    t.manual_seed(seed)

    if t.cuda.is_available():
        t.cuda.manual_seed(seed)
        #避免 cuDNN 根据硬件自动选择高效但不确定的算法，保证每次运行的算法一致
        #降低训练速度
        t.backends.cudnn.benchmark = False  #是否使用 cuDNN 的自动算法优化功能
        #消除算法本身的不确定性（如某些浮点运算的舍入误差），确保 GPU 计算结果严格一致
        #可能牺牲部分性能
        t.backends.cudnn.deterministic = True  #是否强制 cuDNN 使用确定性算法


# Define the Coach class for model training and evaluation
class Coach:
    def __init__(self, handler):
        self.handler = handler

        print('DRUG    ', args.drug, 'GENE', args.gene)
        #数据集长度等价于DGI的长度
        print('NUM  OF   INTERACTIONSSS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()  #哈希表
        mets = ['Loss', 'preLoss', 'Acc']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
        
        # 新增：二跳邻居筛选情况统计
        self.two_hop_stats = {
            'enough_filtered': 0,      # 情况1：有足够的符合条件的二跳邻居
            'some_filtered': 0,        # 情况2：有部分符合条件的二跳邻居，需要重复采样
            'enough_random': 0,        # 情况3：没有二跳邻居，有足够的无交互基因
            'few_random': 0           # 情况4：极端情况，无交互基因不够，需要重复采样
        }
        
        # epoch级别的统计（每个epoch重置）
        self.epoch_two_hop_stats = {
            'enough_filtered': 0,
            'some_filtered': 0,
            'enough_random': 0,
            'few_random': 0
        }
        
        # iteration级别的统计（存储每个iteration的平均值）
        self.iteration_two_hop_stats = []
        
        # 新增：构建基因邻接关系用于二跳邻居查找,只需要执行一次
        print("Building gene-gene adjacency matrix for two-hop neighbors...")
        self.build_gene_adjacency_matrix()
        print("Gene adjacency matrix built successfully!")
        
        # 设置二跳邻居缓存文件路径
        cache_dir = r"D:\桌面\研\论文\实验代码\DGCL-main\DGCL-main\Data\cache"
        cache_filename = f"two_hop_{args.data}_{args.num_two_hop}.npz"
        self.two_hop_cache_path = os.path.normpath(os.path.join(cache_dir, cache_filename))
        print(f"💾 Expected cache file: {self.two_hop_cache_path}")

        # 尝试加载缓存，如果找不到就报错
        self.two_hop_cache = self.load_two_hop_cache()
        print("⚡ Using cached two-hop neighbors for fast training!")

    # Function to create a formatted print statement
    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    # Function to perform external testing
    def external_test_run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        return reses['Acc']

    # Function to train and evaluate the model
    def run(self,i):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:  # 有模型需要加载
            self.loadModel()
            # 已经训练批次*每轮批次的epoch
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        epoch_list = []
        acc_list = []

        aucMax = 0
        bestEpoch = 0

        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            train_loss = reses
            
            # NAN检测：检查训练结果是否包含NAN
            if any(np.isnan(val) if isinstance(val, (int, float)) else False for val in reses.values()):
                print("❌ NAN detected in training results! Stopping current iteration early...")
                print(f"Training stopped at epoch {ep}")
                print(f"NAN detected in training losses: {reses}")
                
                # 输出到目前为止的最佳结果
                if bestEpoch > 0:
                    output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. Best epoch so far: {bestEpoch}, ACC: {round(aucMax, 4)}'
                    print(output_str)
                    # 返回当前最佳结果，继续下一个iteration
                    return aucMax, output_str, aucMax
                else:
                    output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. No valid test results yet.'
                    print(output_str)
                    print("➡️ Continuing to next iteration...")
                    # 没有有效结果，返回NAN，但继续下一个iteration
                    return float('nan'), output_str, float('nan')
            
            # 记录训练结果
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                test_r = reses
                
                # NAN检测：检查测试结果是否包含NAN
                if any(np.isnan(val) if isinstance(val, (int, float)) else False for val in reses.values()):
                    print("❌ NAN detected in test results! Stopping current iteration early...")
                    print(f"Training stopped at epoch {ep}")
                    print(f"NAN detected in test results: {reses}")
                    
                    # 输出到目前为止的最佳结果
                    if bestEpoch > 0:
                        output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. Best epoch so far: {bestEpoch}, ACC: {round(aucMax, 4)}'
                        print(output_str)
                        # 返回当前最佳结果，继续下一个iteration
                        return aucMax, output_str, aucMax
                    else:
                        output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. No valid test results yet.'
                        print(output_str)
                        print("➡️ Continuing to next iteration...")
                        # 没有有效结果，返回NAN，但继续下一个iteration
                        return float('nan'), output_str, float('nan')

                if reses['Acc'] > aucMax:
                    aucMax = reses['Acc']
                    bestEpoch = ep

                # 记录测试结果
                log(self.makePrint('Test', ep, reses, tstFlag))
            logs = {'loss_all': train_loss['Loss'], 'loss_pre': train_loss['preLoss'],
                    'test_acc': test_r['Acc']}
            epoch_list.append(ep)
            acc_list.append(test_r['Acc'])
        # wandb.log(logs)
        plt.plot(epoch_list, acc_list)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        #这里没有权限来保存图片
#        plt.savefig('/home/huangpeng/DGCL-main/dgcl{}.png'.format(i))

        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        self.save_model('{}'.format(config['iteration']))
        
        # 计算当前iteration的二跳邻居统计
        total_iteration_samples = sum(self.two_hop_stats.values())
        if total_iteration_samples > 0:
            iteration_stats = {}
            print(f"\n🔍 Iteration {config['iteration']} Two-hop Neighbor Statistics (Total samples: {total_iteration_samples}):")
            for key, count in self.two_hop_stats.items():
                percentage = count / total_iteration_samples * 100
                iteration_stats[key] = percentage
                situation_names = {
                    'enough_filtered': '情况1 - 足够的二跳邻居',
                    'some_filtered': '情况2 - 部分二跳邻居',
                    'enough_random': '情况3 - 足够的无交互基因',
                    'few_random': '情况4 - 极端情况重复采样'
                }
                print(f"  {situation_names[key]}: {count} ({percentage:.1f}%)")
            
            # 保存当前iteration的统计
            self.iteration_two_hop_stats.append(iteration_stats)
        
        # 重置总统计，为下一个iteration准备
        self.two_hop_stats = {
            'enough_filtered': 0,
            'some_filtered': 0,
            'enough_random': 0,
            'few_random': 0
        }
        
        # 每轮itration结束补充输出一个最好的结果
        output_str = f'Best epoch : {bestEpoch} , ACC : {round(aucMax, 4)}'
        print(output_str)
        return reses['Acc'],output_str,aucMax

    # Function to prepare the model and optimizer
    def prepareModel(self):
        self.model = Model().cuda()
        self.is_data_parallel = False
        
        # 如果启用多GPU并且有多个GPU可用，使用DataParallel
        if args.multi_gpu and t.cuda.device_count() > 1:
            gpu_list = [int(x) for x in args.gpu_list.split(',')]
            available_gpus = [gpu for gpu in gpu_list if gpu < t.cuda.device_count()]
            if len(available_gpus) > 1:
                print(f"Wrapping model with DataParallel on GPUs: {available_gpus}")
                self.model = t.nn.DataParallel(self.model, device_ids=available_gpus)
                self.is_data_parallel = True
        
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
    
    def get_model(self):
        """获取原始模型，处理DataParallel包装问题"""
        if self.is_data_parallel:
            return self.model.module
        else:
            return self.model

    def load_two_hop_cache(self):
        """
        加载二跳邻居缓存文件，如果找不到就报错
        返回: 缓存字典
        """
        print(f"🔍 Looking for cache file: {self.two_hop_cache_path}")
        
        if not os.path.exists(self.two_hop_cache_path):
            raise FileNotFoundError(
                f"❌ Two-hop neighbor cache file not found!\n"
                f"Expected: {self.two_hop_cache_path}\n"
                f"Please create the cache file first."
            )
        
        try:
            print(f"🔄 Loading cache...")
            
            # 检查文件大小
            actual_size = os.path.getsize(self.two_hop_cache_path)
            if actual_size == 0:
                raise ValueError(f"❌ Cache file is empty (0 bytes)")
            
            cache_data = np.load(self.two_hop_cache_path, allow_pickle=True)
            
            # 验证缓存文件结构
            if 'two_hop_neighbors' not in cache_data:
                raise KeyError(f"❌ Cache file missing 'two_hop_neighbors' key")
            
            if 'params' not in cache_data:
                raise KeyError(f"❌ Cache file missing 'params' key")
            
            # 验证缓存参数是否匹配
            cached_params = cache_data['params'].item()
            current_params = {
                'data': args.data,
                'drug': args.drug,
                'gene': args.gene,
                'num_two_hop': args.num_two_hop
            }
            
            if cached_params != current_params:
                raise ValueError(
                    f"❌ Cache parameters mismatch!\n"
                    f"Cached: {cached_params}\n"
                    f"Current: {current_params}"
                )
            
            cache_dict = cache_data['two_hop_neighbors'].item()
            print(f"✅ Loaded {len(cache_dict)} cached pairs ({actual_size/(1024*1024):.2f} MB)")
            
            return cache_dict
            
        except Exception as e:
            raise RuntimeError(f"❌ Failed to load cache file: {e}")

    def save_two_hop_cache(self, cache_dict):
        """
        保存二跳邻居缓存到文件（仅更新现有文件）
        参数: cache_dict - 缓存字典
        """
        try:
            print(f"💾 Updating two-hop neighbors cache...")
            print(f"📁 Target path: {self.two_hop_cache_path}")
            
            # 准备保存的数据
            save_params = {
                'data': args.data,
                'drug': args.drug,
                'gene': args.gene,
                'num_two_hop': args.num_two_hop
            }
            
            # 添加元数据
            metadata = {
                'total_pairs': len(cache_dict),
                'updated_time': time.time(),
                'version': '1.0',
                'description': f'Two-hop neighbors for {args.data} dataset'
            }
            
            # 保存缓存文件（覆盖现有文件）
            print(f"🔄 Writing cache file...")
            np.savez_compressed(
                self.two_hop_cache_path,
                two_hop_neighbors=cache_dict,
                params=save_params,
                metadata=metadata
            )
            
            # 验证文件更新
            if os.path.exists(self.two_hop_cache_path):
                file_size = os.path.getsize(self.two_hop_cache_path) / (1024 * 1024)
                print(f"✅ Cache updated successfully!")
                print(f"📊 Cache file size: {file_size:.2f} MB")
                print(f"📈 Cached {len(cache_dict)} unique (drug, gene) pairs")
            
        except Exception as e:
            print(f"❌ Failed to update cache: {e}")
            import traceback
            traceback.print_exc()

    def precompute_all_two_hop_neighbors(self):
        """
        预计算所有可能的(drug, gene)对的二跳邻居
        """
        print("🚀 Precomputing all two-hop neighbors...")
        cache_dict = {}
        total_pairs = 0
        start_time = time.time()
        
        # 获取所有训练数据中的(drug, gene)对
        train_dataset = self.handler.trnLoader.dataset
        unique_pairs = set()
        
        # 收集所有唯一的(drug, gene)对
        for idx in range(len(train_dataset)):
            drug_idx, gene_idx, _, _ = train_dataset[idx]
            unique_pairs.add((int(drug_idx), int(gene_idx)))
        
        print(f"📊 Found {len(unique_pairs)} unique (drug, gene) pairs")
        
        # 设置随机种子确保缓存结果一致
        np.random.seed(42)
        
        # 预计算每个唯一对的二跳邻居
        for i, (drug_idx, gene_idx) in enumerate(unique_pairs):
            if i % 1000 == 0:
                progress = i / len(unique_pairs) * 100
                elapsed = time.time() - start_time
                print(f"  Progress: {i}/{len(unique_pairs)} ({progress:.1f}%) - {elapsed:.1f}s elapsed")
            
            # 计算这个(drug, gene)对的二跳邻居
            two_hop_neighbors = self._compute_single_two_hop_neighbor(
                drug_idx, gene_idx, args.num_two_hop
            )
            cache_dict[(drug_idx, gene_idx)] = two_hop_neighbors
            total_pairs += 1
        
        elapsed_time = time.time() - start_time
        print(f"✅ Precomputed {total_pairs} unique (drug, gene) pairs in {elapsed_time:.2f}s")
        return cache_dict

    def _compute_single_two_hop_neighbor(self, drug_idx, gene_idx, num_neighbors):
        """
        计算单个(drug, gene)对的二跳邻居
        """
        drug_gene_matrix = self.handler.trnLoader.dataset.dokmat
        
        # 获取一跳邻居
        one_hop = self.gene_neighbors.get(gene_idx, set())
        
        # 获取二跳邻居
        two_hop = set()
        for one_hop_gene in one_hop:
            second_hop = self.gene_neighbors.get(one_hop_gene, set())
            for two_hop_gene in second_hop:
                if two_hop_gene != gene_idx and two_hop_gene not in one_hop:
                    two_hop.add(two_hop_gene)
        
        # 过滤掉与当前药物有交互关系的基因
        filtered_two_hop = []
        for candidate_gene in two_hop:
            if (drug_idx, candidate_gene) not in drug_gene_matrix:
                filtered_two_hop.append(candidate_gene)
        
        # 选择二跳邻居
        if len(filtered_two_hop) >= num_neighbors:
            selected = np.random.choice(filtered_two_hop, size=num_neighbors, replace=False)
        elif len(filtered_two_hop) > 0:
            selected = np.random.choice(filtered_two_hop, size=num_neighbors, replace=True)
        else:
            # 随机选择与药物无交互的基因
            all_genes = set(range(args.gene))
            drug_connected_genes = set()
            
            for (drug, gene) in drug_gene_matrix.keys():
                if drug == drug_idx:
                    drug_connected_genes.add(gene)
            
            no_interaction_genes = list(all_genes - drug_connected_genes)
            
            if len(no_interaction_genes) >= num_neighbors:
                selected = np.random.choice(no_interaction_genes, size=num_neighbors, replace=False)
            else:
                selected = np.random.choice(no_interaction_genes, size=num_neighbors, replace=True)
        
        return selected.tolist()

    def build_gene_adjacency_matrix(self):
        """
        构建基因-基因邻接矩阵
        基于药物-基因交互关系：如果两个基因都与同一个药物有交互，则认为它们间接相关
        """
        # 获取药物-基因交互矩阵 (稀疏矩阵格式)
        drug_gene_matrix = self.handler.trnLoader.dataset.dokmat
        
        # 构建基因-基因邻接字典
        self.gene_neighbors = {}
        
        # 初始化每个基因的邻居集合
        for gene in range(args.gene):
            self.gene_neighbors[gene] = set()
        
        # 遍历所有药物，找到与每个药物交互的基因
        drug_gene_dict = {}
        for (drug, gene) in drug_gene_matrix.keys():
            if drug not in drug_gene_dict:
                drug_gene_dict[drug] = set()
            drug_gene_dict[drug].add(gene)
        
        # 基于药物构建基因间的邻接关系
        for drug, genes in drug_gene_dict.items():
            genes_list = list(genes)
            # 对于每对基因，如果它们与同一药物交互，则它们是邻居
            for i in range(len(genes_list)):
                for j in range(i + 1, len(genes_list)):
                    gene1, gene2 = genes_list[i], genes_list[j]
                    self.gene_neighbors[gene1].add(gene2)
                    self.gene_neighbors[gene2].add(gene1)
        
        print(f"Gene adjacency built: {len(self.gene_neighbors)} genes")
        
    def get_two_hop_neighbors(self, drugs_batch, genes_batch, num_neighbors=5):
        """
        获取基因的二跳邻居，从缓存读取（必须存在缓存文件）
        
        参数:
        drugs_batch: [batch_size] - 药物索引批次
        genes_batch: [batch_size] - 基因索引批次
        num_neighbors: int - 每个基因返回的二跳邻居数量
        
        返回:
        two_hop_neighbors: [batch_size, num_neighbors] - 二跳邻居基因索引（与药物无交互）
        """
        batch_size = len(genes_batch)
        two_hop_neighbors = []
        
        cache_hits = 0
        cache_misses = 0
        
        for i, gene_idx in enumerate(genes_batch):
            gene_idx = gene_idx.item() if hasattr(gene_idx, 'item') else int(gene_idx)
            drug_idx = drugs_batch[i].item() if hasattr(drugs_batch[i], 'item') else int(drugs_batch[i])
            
            key = (drug_idx, gene_idx)
            
            if key in self.two_hop_cache:
                # 从缓存读取
                cached_neighbors = self.two_hop_cache[key]
                two_hop_neighbors.append(cached_neighbors)
                cache_hits += 1
                
                # 根据缓存结果推断情况类型（简化统计）
                self.epoch_two_hop_stats['enough_filtered'] += 1
            else:
                # 缓存未命中，报错
                cache_misses += 1
                raise KeyError(
                    f"❌ Cache miss for (drug={drug_idx}, gene={gene_idx})!\n"
                    f"This key is not in the cache file.\n"
                    f"Please regenerate the cache file with complete data."
                )
        
        return t.tensor(two_hop_neighbors, dtype=t.long)

    # 排序筛选出困难负样本 4096 100
    def sort_hard_embedding(self, itmEmbeds, negEmbeds):
        """
        计算负样本与所有正样本(基因)的相似度，筛选困难负样本
        使用双GPU并行计算，避免显存爆炸
        itmEmbeds: [num_genes, 128] - 所有基因嵌入
        negEmbeds: [batch_size, num_neg, 128] - 负样本嵌入
        """
        batch_size, num_neg, embed_dim = negEmbeds.shape
        num_genes = itmEmbeds.shape[0]
        
        # 重塑negEmbeds为 [batch_size * num_neg, 128] 以便批量计算
        negEmbeds_flat = negEmbeds.view(-1, embed_dim)  # [4096*100, 128]
        total_neg_samples = negEmbeds_flat.shape[0]
        
        print(f"Computing similarity for {total_neg_samples} neg samples vs {num_genes} genes")
        print(f"Estimated memory needed: {(total_neg_samples * num_genes * 4) / 1024**3:.2f} GB")
        
        if args.multi_gpu and torch.cuda.device_count() > 1:
            # 双GPU并行方案
            gpu_list = [int(x) for x in args.gpu_list.split(',')]
            available_gpus = [gpu for gpu in gpu_list if gpu < torch.cuda.device_count()][:2]  # 只使用前两个GPU
            print(f"Using dual GPUs : {available_gpus}")
            
            # 检查GPU显存状态
            for gpu_id in available_gpus:
                gpu_memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
                gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
                gpu_memory_free = gpu_memory_total - gpu_memory_allocated
                print(f"GPU {gpu_id}: {gpu_memory_free:.1f}GB free / {gpu_memory_total:.1f}GB total")
            
            # 将数据分成两半
            mid_point = total_neg_samples // 2
            neg_part1 = negEmbeds_flat[:mid_point]  # 前半部分
            neg_part2 = negEmbeds_flat[mid_point:]  # 后半部分
            
            # 将itmEmbeds复制到两个GPU
            itmEmbeds_gpu0 = itmEmbeds.to(f'cuda:{available_gpus[0]}')
            itmEmbeds_gpu1 = itmEmbeds.to(f'cuda:{available_gpus[1]}')
            
            # GPU 0 计算前半部分
            neg_part1_gpu0 = neg_part1.to(f'cuda:{available_gpus[0]}')
            print(f"GPU {available_gpus[0]} processing {neg_part1.shape[0]} samples...")
            with torch.cuda.device(available_gpus[0]):
                similarity_part1 = t.mm(neg_part1_gpu0, itmEmbeds_gpu0.T)
                # 清理GPU 0上的中间变量
                del neg_part1_gpu0, itmEmbeds_gpu0
                torch.cuda.empty_cache()
                # 将第一部分结果移到GPU 1，为合并做准备
                similarity_part1 = similarity_part1.to(f'cuda:{available_gpus[1]}')
            
            # GPU 1 计算后半部分
            neg_part2_gpu1 = neg_part2.to(f'cuda:{available_gpus[1]}')
            print(f"GPU {available_gpus[1]} processing {neg_part2.shape[0]} samples...")
            with torch.cuda.device(available_gpus[1]):
                similarity_part2 = t.mm(neg_part2_gpu1, itmEmbeds_gpu1.T)
                # 清理GPU 1上的中间变量，为合并腾出空间
                del neg_part2_gpu1, itmEmbeds_gpu1
                torch.cuda.empty_cache()
                
                # 在GPU 1上进行合并操作，避免主GPU显存爆炸
                print(f"GPU {available_gpus[1]} merging results....")
                similarity_scores = t.cat([similarity_part1, similarity_part2], dim=0)
                
                # 清理合并用的临时变量
                del similarity_part1, similarity_part2
                torch.cuda.empty_cache()
                
                # 合并完成后移回主GPU
                similarity_scores = similarity_scores.to(negEmbeds.device)
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            print("✅ Dual GPU computation completed!")
            
        else:
            # 单GPU方案（原始计算）  DGIdb使用多GPU empty_cache()会出现异常
            print("⚠️ Multi-GPU not enabled, using single GPU computation")
            similarity_scores = t.mm(negEmbeds_flat, itmEmbeds.T)

        # 重塑回原来的形状: [batch_size, num_neg, num_genes]
        similarity_scores = similarity_scores.view(batch_size, num_neg, num_genes)
        
        # 对每个负样本，计算其与所有基因的平均相似度（整体困难程度）
        avg_similarities = t.mean(similarity_scores, dim=2)  # [batch_size, num_neg]

        # 使用数值稳定的exp计算
        avg_similarities_clamped = t.clamp(avg_similarities, min=args.score_clamp_min, max=args.score_clamp_max)
        avg_exp = t.exp(avg_similarities_clamped)
        
        # 对每个样本的负样本按困难程度排序（相似度越高越困难）
        sorted_scores, sorted_indices = t.sort(avg_exp, dim=1, descending=True)

        return sorted_indices, sorted_scores



    # Function to train a single epoch
    # 返回损失值 更新参数
    def trainEpoch(self):
        self.model.train()
        trnLoader = self.handler.trnLoader
        # print("开始生成负样本")
        # 这一步如果在DataHandler中就执行会牺牲随机性
        # trnLoader.dataset.negSampling() #这一步非常消耗时间

        # 重置epoch级别的二跳邻居统计
        self.epoch_two_hop_stats = {
            'enough_filtered': 0,
            'some_filtered': 0,
            'enough_random': 0,
            'few_random': 0
        }

        hard_loss_sum=0
        epLoss, epPreLoss = 0, 0
        bprLoss, bpr_loss,reg_loss ,regLoss,im_loss= 0, 0 , 0,0,0
        # 数据集长度
        len__ = trnLoader.dataset.__len__()
        steps = len__ // args.batch  # 步数=长度/批次数
        for i, tem in enumerate(trnLoader):
            data = deepcopy(self.handler.torchBiAdj).cuda()
            drugs, genes, labels , negs = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            # negs = negs.long().cuda()
            usrEmbeds, itmEmbeds = self.get_model().forward_gcn(data)
            # 获取正样本和负样本的嵌入
            drugEmbeds = usrEmbeds[drugs]  # 药物嵌入 4096 128
            posEmbeds = itmEmbeds[genes]  # 正样本嵌入 4096 128
            # 获取基因的二跳邻居
            two_hop_neighbors = self.get_two_hop_neighbors(drugs.cpu(), genes.cpu(), num_neighbors=args.num_two_hop)
            two_hop_neighbors = two_hop_neighbors.cuda()
            two_hop_neighbor_embeds = itmEmbeds[two_hop_neighbors]  # [batch_size, num_two_hop, 128]

            print(f"Two-hop neighbors shape: {two_hop_neighbor_embeds.shape}")
            # print(f"Original genes: {genes[:5]}")
            # print(f"Two-hop neighbors for first 5 genes: {two_hop_neighbors[:5]}")

            # negEmbeds = itmEmbeds[negs]  # 负样本嵌入 4096 100 128
            # HaSa损失 - 筛选困难负样本 暂时不用
            # if (args.num_hard_neg > 0):
            #     hard_neg_indices, hard_neg_scores = self.sort_hard_embedding(usrEmbeds, two_hop_neighbor_embeds)
            #
            #     # 筛选出前num_hard_neg个最困难的负样本 使用预定义的超参数
            #     num_hard_neg = min(args.num_two_hop, args.num_neg)  # 不超过总负样本数
            #     # print(f"Selecting top {num_hard_neg} hard negatives from {args.num_neg} negatives")
            #
            #     # 获取困难负样本的索引 [batch_size, num_hard_neg]
            #     hard_indices_selected = hard_neg_indices[:, :num_hard_neg]
            #
            #     # 获取对应的困难负样本分数 [batch_size, num_hard_neg]
            #     hard_neg_scores_selected = hard_neg_scores[:, :num_hard_neg]
            #
            #     # 对困难负样本分数进行归一化和softmax处理
            #     # 方法1: 先L2归一化再softmax
            #     hard_scores_normalized = F.normalize(hard_neg_scores_selected, p=2, dim=1)  # L2归一化
            #     hard_prob = F.softmax(hard_scores_normalized, dim=1)  # softmax得到概率分布
            #
            #     # 方法2: 直接对原始分数进行softmax（可选，取消注释使用）
            #     # hard_prob = F.softmax(hard_neg_scores_selected, dim=1)
            #     # 使用高级索引选择困难负样本的嵌入
            #     # 直接在GPU上创建batch_indices，避免CPU->GPU传输
            #     batch_indices = t.arange(two_hop_neighbor_embeds.size(0), device=two_hop_neighbor_embeds.device).unsqueeze(1).expand(-1, num_hard_neg)
            #     # 得到困难负样本嵌入
            #     two_hop_neighbor_embeds = two_hop_neighbor_embeds[batch_indices, hard_indices_selected]  # [batch_size, num_hard_neg, 128]
            #
            #     hard_loss = self.get_model().batch_bias_hard(drugEmbeds, posEmbeds, two_hop_neighbor_embeds, hard_prob)
            #
            #     regLoss = calcRegLoss(self.model) * args.reg
            #     self.opt.zero_grad()
            #     loss = hard_loss + regLoss
            #
            #     loss.backward()
            #     hard_loss_sum += hard_loss.detach().item()
            #     self.opt.step()

            # 基于二跳邻居的全局负采样损失
            if hasattr(args, 'num_two_hop') and args.num_two_hop > 0:
                print("Computing BPR loss with two-hop neighbors...")
                # 计算正样本分数
                posScores = innerProduct(drugEmbeds.unsqueeze(1), posEmbeds.unsqueeze(1))  # [batch_size, 1]

                # 计算二跳邻居负样本分数
                negScores = innerProduct(drugEmbeds.unsqueeze(1), two_hop_neighbor_embeds)  # [batch_size, num_two_hop]

                # 不使用加权，直接对二跳邻居负样本分数求和
                aggregated_negScore = negScores.sum(dim=1, keepdim=True)  # [batch_size, 1]

                # 计算BPR损失
                scoreDiff = posScores - aggregated_negScore  # [batch_size, 1]
                bpr_loss_value = -(scoreDiff).sigmoid().log().sum() / args.batch

                print(f"Positive scores shape: {posScores.shape}, mean: {posScores.mean():.4f}")
                print(f"Negative scores shape: {negScores.shape}, mean: {negScores.mean():.4f}")
                print(f"Score difference mean: {scoreDiff.mean():.4f}")
                print(f"BPR loss: {bpr_loss_value:.4f}")

                # 正则化损失
                regLoss = calcRegLoss(self.model) * args.reg
                loss = bpr_loss_value + regLoss

                # 反向传播
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 记录损失
                bpr_loss += float(bpr_loss_value)
                reg_loss += float(regLoss)

                print(f"Total loss: {loss:.4f} (BPR: {bpr_loss_value:.4f} + Reg: {regLoss:.4f})")

            # 计算交叉熵损失
            ceLoss = self.get_model().calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate)
            # sslLoss = sslLoss * args.ssl_reg
            regLoss = calcRegLoss(self.model) * args.reg
            # loss = ceLoss + regLoss + sslLoss
            loss = ceLoss + regLoss
            # 优化GPU->CPU传输
            epLoss += loss.detach().item()
            epPreLoss += ceLoss.detach().item()
            # 清空优化器中的所有梯度，使得下一次反向传播时不会受到之前梯度的影响
            self.opt.zero_grad()
            # 反向传播 沿着计算图计算出梯度
            loss.backward()
            # 梯度裁剪，防止梯度爆炸
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=args.clip_grad_norm)
            # 使用计算得到的梯度和学习率
            # 使用优化算法（例如 SGD、Adam、RMSprop 等）来更新模型参数
            self.opt.step()
            #bprLoss = 0
            regLoss = 0
        
        # 统计当前epoch的二跳邻居筛选情况
        total_samples = sum(self.epoch_two_hop_stats.values())
        if total_samples > 0:
            print(f"\n📊 Epoch Two-hop Neighbor Statistics (Total samples: {total_samples}):")
            print(f"  情况1 - 足够的二跳邻居: {self.epoch_two_hop_stats['enough_filtered']} ({self.epoch_two_hop_stats['enough_filtered']/total_samples*100:.1f}%)")
            print(f"  情况2 - 部分二跳邻居: {self.epoch_two_hop_stats['some_filtered']} ({self.epoch_two_hop_stats['some_filtered']/total_samples*100:.1f}%)")
            print(f"  情况3 - 足够的无交互基因: {self.epoch_two_hop_stats['enough_random']} ({self.epoch_two_hop_stats['enough_random']/total_samples*100:.1f}%)")
            print(f"  情况4 - 极端情况重复采样: {self.epoch_two_hop_stats['few_random']} ({self.epoch_two_hop_stats['few_random']/total_samples*100:.1f}%)")
        
        # 累加到总统计中
        for key in self.two_hop_stats:
            self.two_hop_stats[key] += self.epoch_two_hop_stats[key]
            
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        ret['bpr_loss'] = bpr_loss / steps
        ret['regLoss'] = regLoss / steps
        return ret

    # Function to test a single epoch
    # 计算出ACC
    def testEpoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        i = 0
        for tem in tstLoader:
            i += 1
            drugs, genes, labels = tem
            # 将从数据集获得的张量转移到Gpu上面，默认第一个Gpu
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            # predict(self, adj, drugs, genes)
            pre = self.get_model().predict(self.handler.torchBiAdj, drugs, genes)
            # dim=1 指定了 softmax 操作沿着第 1 维（即类别维度）进行
            pre = F.log_softmax(pre, dim=1)
            # 选出可能性最大的类别，优化GPU->CPU传输
            pre = pre.data.max(1, keepdim=True)[1]
            # 批量转移到CPU，减少传输次数
            pre = pre.detach().cpu()
            labels = labels.detach().cpu()
            epAcc = accuracy_score(labels, pre)
        ret = dict()
        ret['Acc'] = epAcc
        return ret

    # Function to load a pre-trained model
    def loadModel(self):
        # 处理DataParallel包装的模型加载
        if self.is_data_parallel:
            self.model.module.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))
        else:
            self.model.load_state_dict(t.load('../Models/' + args.load_model + '.pkl'))
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')

    # Function to save the trained model
    def save_model(self, model_path):
        # 使用 wandb.run.dir 获取当前 WandB 实验的根目录
        model_parent_path = "D:\\桌面\\研\\论文\\实验代码\\DGCL-main\\DGCL-main\\Models"
        # model_parent_path = os.path.join(wandb.run.dir, 'ckl')
        # if not os.path.exists(model_parent_path):
        #     os.mkdir(model_parent_path)
        # # 将模型参数保存到指定的模型路径上（网络层的权重、偏置）
        # # 处理DataParallel包装的模型保存
        # if self.is_data_parallel:
        #     t.save(self.model.module.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))
        # else:
        #     t.save(self.model.state_dict(), '{}/{}_model.pkl'.format(model_parent_path, model_path))


# Main execution block
if __name__ == '__main__':

    matplotlib.use('Agg')
    try:
        import wandb
    except ModuleNotFoundError:
        print("wandb is not installed, skipping related functionality.")


    if args.is_debug is True:
        print("DEBUGGING MODE - Start without wandb")
        # debug模式下不需要wandb
        #wandb.init(mode="disabled")
    # else:
    # 记录配置与项目名
    # wandb.init(project='HC', config=args)
    # '.' 表示将当前目录 将当前目录中的所有代码上传
    # wandb.run.log_code(".")


    use_cuda = args.gpu >= 0 and t.cuda.is_available()
    
    if args.multi_gpu and t.cuda.device_count() > 1:
        gpu_list = [int(x) for x in args.gpu_list.split(',')]
        available_gpus = [gpu for gpu in gpu_list if gpu < t.cuda.device_count()]
        print(f"Multi-GPU mode enabled. Available GPUs: {available_gpus}")
        device = 'cuda:{}'.format(available_gpus[0])  # 主GPU
        args.device = device
        args.available_gpus = available_gpus
    else:
        device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
        args.device = device
        
    if use_cuda:
        t.cuda.set_device(device)
        
    print(f"Primary device: {device}")
    print(f"Total GPU count: {t.cuda.device_count()}")
    
    # 显示GPU内存信息
    if use_cuda:
        for i in range(t.cuda.device_count()):
            gpu_memory = t.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {t.cuda.get_device_name(i)}, {gpu_memory:.1f} GB")

    logger.saveDefault = True

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)
    config = dict()
    results = list()
    aucMax_list = list()
    outputstr_list = list()

    epoch_list = []
    acc_list = []

    iteration_list = []
    end_acc_list = []
    output_str = None
    it_max=0
    aucMax=0
    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed + i
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        if args.data == 'LINCS':
            result = coach.external_test_run()
        else:
            result,output_str,aucMax = coach.run(i)  # 返回最终测试得到的reses['Acc']
            
        # 处理NAN结果：记录但继续下一个iteration
        if np.isnan(result):
            print(f"⚠️ Iteration {i + 1} returned NAN result, but continuing to next iteration...")
            print(f"Current best across all iterations: {it_max}")
        else:
            print(f"✅ Iteration {i + 1} completed successfully with result: {result}")
            
        iteration_list.append(i)
        end_acc_list.append(result)
        results.append(result)
        aucMax_list.append(aucMax)
        outputstr_list.append(output_str)
        
        # 只有非NAN的aucMax才更新it_max
        if not np.isnan(aucMax) and aucMax > it_max:
            it_max = aucMax
            print(f"🎯 New best result updated: {it_max} at iteration {i + 1}")


    plt.plot(iteration_list, end_acc_list)
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.savefig('/home/huangpeng/DGCL-main/dgcl.png')

    # 统计有效结果和NAN结果
    valid_results = [r for r in results if not np.isnan(r)]
    valid_aucMax_list = [a for a in aucMax_list if not np.isnan(a)]
    nan_count = len(results) - len(valid_results)
    
    print(f"\n📊 Training Summary:")
    print(f"Total iterations: {len(results)}")
    print(f"Successful iterations: {len(valid_results)}")
    print(f"NAN iterations: {nan_count}")
    
    if len(valid_results) > 0:
        avg_r = np.mean(np.array(valid_results), axis=0)
        std_r = np.std(valid_results, axis=0)  # 求标准差
        avg_aucMax = np.mean(np.array(valid_aucMax_list), axis=0)
        
        print('test results: ')
        print(results)
        print('best epoch results: ')
        print(outputstr_list)
        print('有效结果平均值: {}'.format(avg_r))
        print('有效结果平均最大值: {}'.format(avg_aucMax))
        print('所有iteration最大值: {}'.format(it_max))
        print('有效结果方差: {}'.format(std_r))
    else:
        print("❌ All iterations resulted in NAN!")
        avg_r = float('nan')
        avg_aucMax = float('nan')
        std_r = float('nan')

    results.append(avg_r)
    results.append(std_r)

    # 计算所有iteration的二跳邻居统计平均值
    if len(coach.iteration_two_hop_stats) > 0:
        print(f"\n🏆 All Iterations Two-hop Neighbor Statistics Summary:")
        print(f"Total valid iterations: {len(coach.iteration_two_hop_stats)}")
        
        # 计算各情况的平均百分比
        avg_stats = {}
        for key in ['enough_filtered', 'some_filtered', 'enough_random', 'few_random']:
            percentages = [stats[key] for stats in coach.iteration_two_hop_stats]
            avg_stats[key] = np.mean(percentages)
        
        situation_names = {
            'enough_filtered': '情况1 - 足够的二跳邻居',
            'some_filtered': '情况2 - 部分二跳邻居', 
            'enough_random': '情况3 - 足够的无交互基因',
            'few_random': '情况4 - 极端情况重复采样'
        }
        
        print("📈 各情况平均占比:")
        for key, avg_percentage in avg_stats.items():
            print(f"  {situation_names[key]}: {avg_percentage:.1f}%")
        
        # 输出详细的iteration统计
        print(f"\n📋 每个Iteration的详细统计:")
        for i, stats in enumerate(coach.iteration_two_hop_stats):
            print(f"  Iteration {i+1}: 情况1={stats['enough_filtered']:.1f}%, 情况2={stats['some_filtered']:.1f}%, 情况3={stats['enough_random']:.1f}%, 情况4={stats['few_random']:.1f}%")
    else:
        print("⚠️ No valid iteration statistics available")

    # results_parent_path = "D:\\桌面\\研\\论文\\实验代码\\DGCL-main\\DGCL-main\\results"
    # # results_parent_path = os.path.join(wandb.run.dir, 'results')
    # if not os.path.exists(results_parent_path):
    #     os.mkdir(results_parent_path)
    # np.savetxt('{}/{}_result.txt'.format(results_parent_path, args.data), np.array(results), delimiter=",", fmt='%f')

    print('result saved!!!')
    #wandb.finish()
