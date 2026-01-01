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
from multiprocess_helper_optimized import init_worker_process, compute_single_pair_optimized
from preprocess_drug_gene_dict import load_drug_gene_dict
import os
import torch.nn.functional as F

from sklearn.metrics import accuracy_score, average_precision_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

import numpy as np
import random
from copy import deepcopy
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from datetime import datetime


# Function to set random seed for reproducibility
# 通过设置一个固定的种子值，我们可以确保每次实验的初始化和随机过程相同，从而使得实验的结果可重复
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
        # 避免 cuDNN 根据硬件自动选择高效但不确定的算法，保证每次运行的算法一致
        # 降低训练速度
        t.backends.cudnn.benchmark = False  # 是否使用 cuDNN 的自动算法优化功能
        # 消除算法本身的不确定性（如某些浮点运算的舍入误差），确保 GPU 计算结果严格一致
        # 可能牺牲部分性能
        t.backends.cudnn.deterministic = True  # 是否强制 cuDNN 使用确定性算法


# Define the Coach class for model training and evaluation
class Coach:
    def __init__(self, handler, drug_gene_dict, log_file=None):
        self.handler = handler
        self.drug_gene_dict = drug_gene_dict
        self.log_file = log_file

        print('DRUG    ', args.drug, 'GENE', args.gene)
        # 数据集长度等价于DGI的长度
        print('NUM  OF   INTERACTIONSSS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()  # 哈希表
        mets = ['Loss', 'preLoss', 'Acc']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

        self.metrics_to_track = ['Acc', 'F1', 'AUC', 'precision', 'recall', 'Auprc']

        # 新增：二跳邻居筛选情况统计
        self.two_hop_stats = {
            'enough_filtered': 0,  # 情况1：有足够的符合条件的二跳邻居
            'some_filtered': 0,  # 情况2：有部分符合条件的二跳邻居，需要重复采样
            'enough_random': 0,  # 情况3：没有二跳邻居，有足够的无交互基因
            'few_random': 0  # 情况4：极端情况，无交互基因不够，需要重复采样
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

        # 新增：从预处理缓存加载基因邻接关系用于二跳邻居查找
        print("Loading gene-gene adjacency matrix from preprocessed cache...")
        drug_gene_dict, gene_neighbors = load_drug_gene_dict()
        if gene_neighbors is not None:
            self.gene_neighbors = gene_neighbors
            print(f"✅ Gene adjacency matrix loaded from cache: {len(gene_neighbors)} genes")
        else:
            raise RuntimeError(
                f"❌ Failed to load gene_neighbors cache for {args.data}!\n"
                f"Please run: python preprocess_drug_gene_dict.py --data {args.data}"
            )

        # 设置混合困难负样本缓存文件路径
        cache_dir = r"/mnt/data/huangpeng/DGCL/DGCL-main/Data/cache"
        if args.data == 'DrugBank':
            # DrugBank使用不含权重信息的文件名
            cache_filename = f"{args.data}_NoWeight_{args.num_two_hop}_r{int(args.one_hop_max_ratio * 100)}_mixed_hard_neg.npz"
        else:
            # 其他数据集（如DGIdb）保持原有文件名格式
            cache_filename = f"{args.data}_{args.num_two_hop}_w1h{args.one_hop_weight}_w2h{args.two_hop_weight}_r{int(args.one_hop_max_ratio * 100)}_mixed_hard_neg.npz"
        self.two_hop_cache_path = os.path.normpath(os.path.join(cache_dir, cache_filename))

        # 尝试加载缓存
        self.two_hop_cache = self.load_two_hop_cache()

        # 如果缓存为空，自动计算并保存
        if self.two_hop_cache is None:
            print("🚀 Cache is empty, computing all mixed hard negatives...")
            self.two_hop_cache = self.precompute_all_mixed_hard_negatives()
            self.save_two_hop_cache(self.two_hop_cache)
            print("✅ Mixed hard negatives computed and cached successfully!")

        print(
            f"⚡ Using cached mixed hard negatives (one-hop weight: {args.one_hop_weight}, two-hop weight: {args.two_hop_weight}, max ratio: {args.one_hop_max_ratio:.1%})!")

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
    def run(self, i):
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

        best_metrics = {met: {'value': float('-inf'), 'epoch': -1} for met in self.metrics_to_track}

        def update_best_metrics(result_dict, epoch_idx):
            for metric in self.metrics_to_track:
                metric_value = result_dict.get(metric)
                if metric_value is None or np.isnan(metric_value):
                    continue
                if metric_value > best_metrics[metric]['value']:
                    best_metrics[metric]['value'] = metric_value
                    best_metrics[metric]['epoch'] = epoch_idx

        aucMax = 0
        bestEpoch = 0

        test_r = {met: float('nan') for met in self.metrics_to_track}

        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch(ep)
            train_loss = reses

            # NAN检测：检查训练结果是否包含NAN
            if any(np.isnan(val) if isinstance(val, (int, float)) else False for val in reses.values()):
                print("❌ NAN detected in training results! Stopping current iteration early...")
                print(f"Training stopped at epoch {ep}")
                print(f"NAN detected in training losses: {reses}")
                nan_metrics = {metric: float('nan') for metric in self.metrics_to_track}
                # 输出到目前为止的最佳结果
                if bestEpoch > 0:
                    output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. Best epoch so far: {bestEpoch}, ACC: {round(aucMax, 4)}'
                    print(output_str)
                    # 返回当前最佳结果，继续下一个iteration
                    return aucMax, output_str, aucMax, nan_metrics
                else:
                    output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. No valid test results yet.'
                    print(output_str)
                    print("➡️ Continuing to next iteration...")
                    # 没有有效结果，返回NAN，但继续下一个iteration
                    return float('nan'), output_str, float('nan'), nan_metrics

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
                    nan_metrics = {metric: float('nan') for metric in self.metrics_to_track}
                    # 输出到目前为止的最佳结果
                    if bestEpoch > 0:
                        output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. Best epoch so far: {bestEpoch}, ACC: {round(aucMax, 4)}'
                        print(output_str)
                        # 返回当前最佳结果，继续下一个iteration
                        return aucMax, output_str, aucMax, nan_metrics
                    else:
                        output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. No valid test results yet.'
                        print(output_str)
                        print("➡️ Continuing to next iteration...")
                        # 没有有效结果，返回NAN，但继续下一个iteration
                        return float('nan'), output_str, float('nan'), nan_metrics

                if reses['Acc'] > aucMax:
                    aucMax = reses['Acc']
                    bestEpoch = ep

                # 记录测试结果
                log(self.makePrint('Test', ep, reses, tstFlag))
                update_best_metrics(reses, ep)
            logs = {'loss_all': train_loss['Loss'], 'loss_pre': train_loss['preLoss'],
                    'test_acc': test_r['Acc']}
            epoch_list.append(ep)
            acc_list.append(test_r['Acc'])
        # wandb.log(logs)
        plt.plot(epoch_list, acc_list)
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        # 这里没有权限来保存图片
        #        plt.savefig('/home/huangpeng/DGCL-main/dgcl{}.png'.format(i))

        reses = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        update_best_metrics(reses, args.epoch)

        self.save_model('{}'.format(config['iteration']))

        # 计算当前iteration的二跳邻居统计
        total_iteration_samples = sum(self.two_hop_stats.values())
        if total_iteration_samples > 0:
            iteration_stats = {}
            print(
                f"\n🔍 Iteration {config['iteration']} Two-hop Neighbor Statistics (Total samples: {total_iteration_samples}):")
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
        best_lines = []
        acc_best_epoch = best_metrics['Acc']['epoch']
        if acc_best_epoch != -1:
            aucMax = best_metrics['Acc']['value']
            bestEpoch = acc_best_epoch
        else:
            aucMax = float('nan')
            bestEpoch = -1
        output_str = f'Best epoch : {bestEpoch} , ACC : {round(aucMax, 4) if not np.isnan(aucMax) else "N/A"}'
        best_lines.append(output_str)

        for metric in self.metrics_to_track:
            if metric == 'Acc':
                continue
            best_value = best_metrics[metric]['value']
            best_epoch = best_metrics[metric]['epoch']
            if best_epoch == -1:
                best_lines.append(f'Best epoch for {metric} : N/A , {metric} : N/A')
            else:
                best_lines.append(
                    f'Best epoch for {metric} : {best_epoch} , {metric} : {round(best_value, 4)}'
                )

        log_text = '\n'.join(best_lines)
        print(log_text)
        if self.log_file:
            self.log_file.write(log_text + '\n')

        iteration_best_metrics = {
            metric: (best_metrics[metric]['value'] if best_metrics[metric]['epoch'] != -1 else float('nan'))
            for metric in self.metrics_to_track
        }

        return reses['Acc'], output_str, aucMax, iteration_best_metrics

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
        加载二跳邻居缓存文件，如果找不到或为空则返回None
        返回: 缓存字典或None
        """
        print(f"🔍 Looking for cache file: {self.two_hop_cache_path}")

        if not os.path.exists(self.two_hop_cache_path):
            print("📁 Cache file not found")
            return None

        try:
            print(f"🔄 Loading cache...")

            # 检查文件大小
            actual_size = os.path.getsize(self.two_hop_cache_path)
            if actual_size == 0:
                print("📁 Cache file is empty")
                return None
            # 从缓存中加载二跳数据
            cache_data = np.load(self.two_hop_cache_path, allow_pickle=True)

            # 验证缓存文件结构
            if 'two_hop_neighbors' not in cache_data or 'params' not in cache_data:
                print("📁 Cache file structure invalid")
                return None

            # 验证缓存参数是否匹配
            cached_params = cache_data['params'].item()
            if args.data == 'DrugBank':
                # DrugBank只验证与权重无关的参数
                current_params = {
                    'data': args.data,
                    'drug': args.drug,
                    'gene': args.gene,
                    'num_two_hop': args.num_two_hop,
                    'one_hop_max_ratio': args.one_hop_max_ratio
                }
            else:
                # 其他数据集验证所有参数，包括权重
                current_params = {
                    'data': args.data,
                    'drug': args.drug,
                    'gene': args.gene,
                    'num_two_hop': args.num_two_hop,
                    'one_hop_weight': args.one_hop_weight,
                    'two_hop_weight': args.two_hop_weight,
                    'one_hop_max_ratio': args.one_hop_max_ratio
                }

            if cached_params != current_params:
                print(f"📁 Cache parameters mismatch:")
                print(f"  Cached: {cached_params}")
                print(f"  Current: {current_params}")
                return None

            cache_dict = cache_data['two_hop_neighbors'].item()
            print(f"✅ Loaded {len(cache_dict)} cached pairs ({actual_size / (1024 * 1024):.2f} MB)")

            return cache_dict

        except Exception as e:
            print(f"❌ Failed to load cache file: {e}")
            return None

    def save_two_hop_cache(self, cache_dict):
        """
        保存二跳邻居缓存到文件
        参数: cache_dict - 缓存字典
        """
        try:
            print(f"💾 Saving two-hop neighbors cache...")
            print(f"📁 Target path: {self.two_hop_cache_path}")

            # 准备保存的数据
            if args.data == 'DrugBank':
                # DrugBank只保存与权重无关的参数
                save_params = {
                    'data': args.data,
                    'drug': args.drug,
                    'gene': args.gene,
                    'num_two_hop': args.num_two_hop,
                    'one_hop_max_ratio': args.one_hop_max_ratio
                }
            else:
                # 其他数据集保存所有参数
                save_params = {
                    'data': args.data,
                    'drug': args.drug,
                    'gene': args.gene,
                    'num_two_hop': args.num_two_hop,
                    'one_hop_weight': args.one_hop_weight,
                    'two_hop_weight': args.two_hop_weight,
                    'one_hop_max_ratio': args.one_hop_max_ratio
                }

            # 添加元数据
            metadata = {
                'total_pairs': len(cache_dict),
                'updated_time': time.time(),
                'version': '1.0',
                'description': f'Two-hop neighbors for {args.data} dataset'
            }

            # 保存缓存文件
            print(f"🔄 Writing cache file...")
            np.savez_compressed(
                self.two_hop_cache_path,
                two_hop_neighbors=cache_dict,
                params=save_params,
                metadata=metadata
            )

            # 验证文件保存
            if os.path.exists(self.two_hop_cache_path):
                file_size = os.path.getsize(self.two_hop_cache_path) / (1024 * 1024)
                print(f"✅ Cache saved successfully!")
                print(f"📊 Cache file size: {file_size:.2f} MB")
                print(f"📈 Cached {len(cache_dict)} unique (drug, gene) pairs")

        except Exception as e:
            print(f"❌ Failed to save cache: {e}")
            import traceback
            traceback.print_exc()

    def precompute_all_mixed_hard_negatives(self):
        """
        预计算所有可能的(drug, gene)对的混合困难负样本
        DrugBank使用多进程优化，DGIdb使用原有单线程方式
        """
        print("🚀 Precomputing all mixed hard negatives (1-hop + 2-hop)...")
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

        unique_pairs_list = list(unique_pairs)
        total_unique_pairs = len(unique_pairs_list)

        print(f"📊 Found {total_unique_pairs} unique (drug, gene) pairs")
        print(
            f"📊 One-hop max ratio: {args.one_hop_max_ratio:.1%} ({int(args.num_two_hop * args.one_hop_max_ratio)} max one-hop)")

        # 统计各种情况的计数
        situation_stats = {
            'mixed_with_one_hop': 0,
            'no_one_hop_available': 0,
            'mixed_with_repeated_two_hop': 0,
            'mixed_with_random': 0,
            'mixed_few_random': 0
        }

        one_hop_count_sum = 0
        two_hop_count_sum = 0

        # 根据数据集选择不同的计算策略
        if args.data == 'DrugBank':
            # DrugBank 使用优化的多进程方案
            print(f"🔥 DrugBank dataset detected - using optimized multiprocess")

            # 减少进程数，避免过多的进程创建开销
            cpu_count = multiprocessing.cpu_count()
            # max_workers = min(max(1, int(cpu_count * 0.5)), 24)  # 使用50%的CPU，最多16进程
            max_workers = 24
            print(f"💻 Using {max_workers} processes (CPU cores: {cpu_count})")

            # 准备共享数据（只传输一次）
            drug_gene_matrix_keys = list(self.handler.trnLoader.dataset.dokmat.keys())

            # 准备轻量级任务参数（不包含大数据）
            # print(f"📦 Preparing optimized tasks for {total_unique_pairs} pairs...")
            # print(f"📊 Data sizes: gene_neighbors={len(self.gene_neighbors)}, drug_gene_matrix={len(drug_gene_matrix_keys)}")

            tasks = []
            for i, (drug_idx, gene_idx) in enumerate(unique_pairs_list):
                task_args = (  # 每个任务使用不同的随机种子（移除大数据传输）
                    drug_idx, gene_idx, args.num_two_hop,
                    args.one_hop_max_ratio, args.gene, 42 + i
                )
                tasks.append(task_args)

            print(f"📈 Starting optimized multiprocess computation...")

            # 使用进程池执行计算（带初始化）
            completed = 0
            last_progress_report = 0

            with ProcessPoolExecutor(
                    max_workers=max_workers,
                    initializer=init_worker_process,
                    initargs=(args.data,)  # 逗号很重要
            ) as executor:
                progress_interval = max(100, total_unique_pairs // 200)  # 每1%或至少每10个报告一次
                print(f"🔧 Initializing worker processes with shared data, 汇报频率：{progress_interval}")

                # 使用 executor.map 提高效率，并设置合理的 chunksize 481
                workers_ = total_unique_pairs // (max_workers * 4)
                chunksize = min(100, workers_)
                results_iterator = executor.map(compute_single_pair_optimized, tasks, chunksize=chunksize)

                # 处理结果
                for i, future_result in enumerate(results_iterator):
                    try:
                        (drug_idx, gene_idx), result, pid = future_result

                        # 统计一跳和二跳邻居数量 (DrugBank)
                        one_hop_count = result['one_hop_count']
                        two_hop_count = len(result['negatives']) - one_hop_count
                        one_hop_count_sum += one_hop_count
                        two_hop_count_sum += two_hop_count

                        # 保存到缓存 (DrugBank)
                        cache_dict[(drug_idx, gene_idx)] = {
                            'negatives': result['negatives'],
                            'one_hop_count': one_hop_count
                        }
                        situation_stats[result['situation_type']] += 1
                        total_pairs += 1
                        completed += 1

                        # 降低进度报告频率
                        if completed - last_progress_report >= progress_interval or completed == total_unique_pairs:
                            progress = completed / total_unique_pairs * 100
                            elapsed = time.time() - start_time
                            if completed > 0:
                                avg_time_per_pair = elapsed / completed
                                eta = avg_time_per_pair * (total_unique_pairs - completed)
                                pairs_per_sec = completed / elapsed if elapsed > 0 else 0
                                print(f" [PID:{pid:5d}] Progress: {completed}/{total_unique_pairs} ({progress:.1f}%) - "
                                      f"{elapsed:.1f}s elapsed, ETA: {eta:.1f}s ({pairs_per_sec:.1f} pairs/s)")
                                last_progress_report = completed

                    except Exception as e:
                        # executor.map 在遇到异常时会中断，这里我们记录原始任务信息
                        drug_idx, gene_idx = unique_pairs_list[i]
                        print(f"❌ Task failed for pair ({drug_idx}, {gene_idx}): {e}")
                        # 使用默认值作为fallback (DrugBank)
                        cache_dict[(drug_idx, gene_idx)] = {
                            'negatives': list(range(args.num_two_hop)),
                            'one_hop_count': 0
                        }
                        situation_stats['mixed_few_random'] += 1
                        total_pairs += 1
                        completed += 1

        else:
            # DGIdb 等其他数据集使用原有单线程方式
            np.random.seed(42)
            progress_interval = 1000
            print(f"📈 Starting single-threaded computation for {total_unique_pairs} pairs...")

            for i, (drug_idx, gene_idx) in enumerate(unique_pairs_list):
                if i % progress_interval == 0 or i == total_unique_pairs - 1:
                    progress = i / total_unique_pairs * 100
                    elapsed = time.time() - start_time
                    if i > 0:
                        avg_time_per_pair = elapsed / i
                        eta = avg_time_per_pair * (total_unique_pairs - i)
                        print(
                            f"  Progress: {i}/{total_unique_pairs} ({progress:.1f}%) - {elapsed:.1f}s elapsed, ETA: {eta:.1f}s")
                    else:
                        print(f"  Progress: {i}/{total_unique_pairs} ({progress:.1f}%) - {elapsed:.1f}s elapsed")

                # 计算这个(drug, gene)对的混合困难负样本
                hard_negatives, weights, situation_type = self._compute_mixed_hard_negatives(
                    drug_idx, gene_idx, args.num_two_hop
                )

                # 统计一跳和二跳邻居数量
                one_hop_count = sum(1 for w in weights if w == args.one_hop_weight)
                two_hop_count = sum(1 for w in weights if w == args.two_hop_weight)
                one_hop_count_sum += one_hop_count
                two_hop_count_sum += two_hop_count

                # 保存到缓存（包含邻居和权重）
                cache_dict[(drug_idx, gene_idx)] = {
                    'negatives': hard_negatives,
                    'weights': weights
                }
                situation_stats[situation_type] += 1
                total_pairs += 1

        elapsed_time = time.time() - start_time
        pairs_per_second = total_pairs / elapsed_time if elapsed_time > 0 else 0
        print(f"✅ Precomputed {total_pairs} unique (drug, gene) pairs in {elapsed_time:.2f}s")
        print(f"⚡ Performance: {pairs_per_second:.1f} pairs/second")

        # 输出统计信息
        print(f"📊 Mixed Hard Negatives Statistics:")
        situation_names = {
            'mixed_with_one_hop': '混合策略 - 包含一跳邻居',
            'no_one_hop_available': '混合策略 - 无一跳邻居可用',
            'mixed_with_repeated_two_hop': '混合策略 - 二跳邻居重复采样',
            'mixed_with_random': '混合策略 - 包含随机样本',
            'mixed_few_random': '混合策略 - 极端情况'
        }
        for situation, count in situation_stats.items():
            if count > 0:  # 只显示有数据的情况
                percentage = count / total_pairs * 100
                print(f"  {situation_names[situation]}: {count} ({percentage:.1f}%)")

        # 输出权重分布统计
        if total_pairs > 0 and (one_hop_count_sum + two_hop_count_sum) > 0:
            avg_one_hop = one_hop_count_sum / total_pairs
            avg_two_hop = two_hop_count_sum / total_pairs
            one_hop_ratio = one_hop_count_sum / (one_hop_count_sum + two_hop_count_sum)
            print(f"📊 Sample Distribution:")
            print(f"  平均一跳样本数: {avg_one_hop:.1f} ({one_hop_ratio:.1%})")
            print(f"  平均二跳样本数: {avg_two_hop:.1f} ({1 - one_hop_ratio:.1%})")
            print(f"  实际一跳比例 vs 最大允许比例: {one_hop_ratio:.1%} vs {args.one_hop_max_ratio:.1%}")

        return cache_dict

    def _compute_mixed_hard_negatives(self, drug_idx, gene_idx, num_neighbors):
        """
        计算混合的困难负样本（优先一跳，不超过指定比例）
        针对不同数据集使用不同的优化策略
        返回: (neighbors_list, weights_list, situation_type)
        """
        drug_gene_matrix = self.handler.trnLoader.dataset.dokmat
        # DGIdb 等其他数据集使用原有稳定的算法
        return self._compute_mixed_hard_negatives_original(drug_idx, gene_idx, num_neighbors, drug_gene_matrix)

    def _compute_mixed_hard_negatives_original(self, drug_idx, gene_idx, num_neighbors, drug_gene_matrix):
        """
        原始版本的混合困难负样本计算（用于DGIdb等稳定数据集）
        """
        # 获取一跳邻居
        one_hop = self.gene_neighbors.get(gene_idx, set())

        # 过滤一跳邻居（确保与药物无交互）
        filtered_one_hop = []
        for candidate_gene in one_hop:
            if (drug_idx, candidate_gene) not in drug_gene_matrix:
                filtered_one_hop.append(candidate_gene)

        # 获取二跳邻居
        two_hop = set()
        for one_hop_gene in one_hop:
            second_hop = self.gene_neighbors.get(one_hop_gene, set())
            for two_hop_gene in second_hop:
                if two_hop_gene != gene_idx and two_hop_gene not in one_hop:
                    two_hop.add(two_hop_gene)

        # 过滤二跳邻居（确保与药物无交互）
        filtered_two_hop = []
        for candidate_gene in two_hop:
            if (drug_idx, candidate_gene) not in drug_gene_matrix:
                filtered_two_hop.append(candidate_gene)

        # 计算一跳邻居的最大允许数量（不超过指定比例）
        max_one_hop = int(num_neighbors * args.one_hop_max_ratio)

        # 确定实际使用的一跳邻居数量（不超过可用数量和最大允许数量）
        actual_one_hop = min(len(filtered_one_hop), max_one_hop)
        actual_two_hop = num_neighbors - actual_one_hop

        selected_negatives = []
        weights = []

        # 采样一跳邻居（优先但不强求）
        if actual_one_hop > 0 and len(filtered_one_hop) > 0:
            if len(filtered_one_hop) >= actual_one_hop:
                one_hop_samples = np.random.choice(filtered_one_hop, size=actual_one_hop, replace=False)
            else:
                # 这种情况理论上不会发生，因为actual_one_hop已经考虑了可用数量
                one_hop_samples = np.random.choice(filtered_one_hop, size=actual_one_hop, replace=True)

            selected_negatives.extend(one_hop_samples.tolist())
            weights.extend([args.one_hop_weight] * actual_one_hop)  # 一跳邻居使用设定的权重倍数
            situation_type = 'mixed_with_one_hop'
        else:
            situation_type = 'no_one_hop_available'

        # 采样二跳邻居填充剩余位置
        if actual_two_hop > 0:
            if len(filtered_two_hop) >= actual_two_hop:
                two_hop_samples = np.random.choice(filtered_two_hop, size=actual_two_hop, replace=False)
            elif len(filtered_two_hop) > 0:
                two_hop_samples = np.random.choice(filtered_two_hop, size=actual_two_hop, replace=True)
                situation_type = 'mixed_with_repeated_two_hop'
            else:
                # 没有二跳邻居，使用随机无交互基因
                all_genes = set(range(args.gene))
                drug_connected_genes = set()

                for (drug, gene) in drug_gene_matrix.keys():
                    if drug == drug_idx:
                        drug_connected_genes.add(gene)

                no_interaction_genes = list(all_genes - drug_connected_genes)

                if len(no_interaction_genes) >= actual_two_hop:
                    two_hop_samples = np.random.choice(no_interaction_genes, size=actual_two_hop, replace=False)
                    situation_type = 'mixed_with_random'
                else:
                    two_hop_samples = np.random.choice(no_interaction_genes, size=actual_two_hop, replace=True)
                    situation_type = 'mixed_few_random'

            selected_negatives.extend(two_hop_samples.tolist())
            weights.extend([args.two_hop_weight] * actual_two_hop)  # 二跳邻居使用设定的权重倍数

        return selected_negatives, weights, situation_type

    # def build_gene_adjacency_matrix(self):
    #     """
    #     构建基因-基因邻接矩阵 一跳邻居
    #     基于药物-基因交互关系：如果两个基因都与同一个药物有交互，则认为它们间接相关
    #
    #     已弃用：改为从预处理缓存加载gene_neighbors
    #     """
    #     # 获取药物-基因交互矩阵 (稀疏矩阵格式)
    #     drug_gene_matrix = self.handler.trnLoader.dataset.dokmat
    #
    #     # 构建基因-基因邻接字典
    #     self.gene_neighbors = {}
    #
    #     # 初始化每个基因的邻居集合
    #     for gene in range(args.gene):
    #         self.gene_neighbors[gene] = set()
    #
    #     # 遍历所有药物，找到与每个药物交互的基因
    #     drug_gene_dict = {}
    #     for (drug, gene) in drug_gene_matrix.keys():
    #         if drug not in drug_gene_dict:
    #             drug_gene_dict[drug] = set()
    #         drug_gene_dict[drug].add(gene)
    #
    #     # 基于药物构建基因间的邻接关系
    #     for drug, genes in drug_gene_dict.items():
    #         genes_list = list(genes)
    #         # 对于每对基因，如果它们与同一药物交互，则它们是邻居
    #         for i in range(len(genes_list)):
    #             for j in range(i + 1, len(genes_list)):
    #                 gene1, gene2 = genes_list[i], genes_list[j]
    #                 self.gene_neighbors[gene1].add(gene2)
    #                 self.gene_neighbors[gene2].add(gene1)
    #
    #     print(f"Gene adjacency built: {len(self.gene_neighbors)} genes")

    def get_mixed_hard_negatives(self, drugs_batch, genes_batch, num_neighbors=None, current_epoch=None,
                                 batch_idx=None):
        """
        获取混合困难负样本和对应权重（从缓存读取）

        参数:
        drugs_batch: [batch_size] - 药物索引批次
        genes_batch: [batch_size] - 基因索引批次
        num_neighbors: int - 每个基因返回的困难负样本数量
        current_epoch: int - 当前epoch编号
        batch_idx: int - 当前batch索引

        返回:
        hard_negatives: [batch_size, num_neighbors] - 混合困难负样本基因索引
        weights: [batch_size, num_neighbors] - 对应的权重
        """
        if num_neighbors is None:
            num_neighbors = args.num_two_hop

        batch_size = len(genes_batch)
        hard_negatives = []
        weights = []

        cache_hits = 0
        cache_misses = 0

        for i, gene_idx in enumerate(genes_batch):
            gene_idx = gene_idx.item() if hasattr(gene_idx, 'item') else int(gene_idx)
            drug_idx = drugs_batch[i].item() if hasattr(drugs_batch[i], 'item') else int(drugs_batch[i])

            key = (drug_idx, gene_idx)

            if key in self.two_hop_cache:
                # 从缓存读取混合困难负样本信息
                cached_data = self.two_hop_cache[key]

                # 检查是否为新的混合格式
                if isinstance(cached_data, dict) and 'negatives' in cached_data:
                    cached_negatives = cached_data['negatives']

                    if 'one_hop_count' in cached_data:

                        # DrugBank的新格式：动态重建权重
                        one_hop_count = cached_data['one_hop_count']  # one_hop_count 3
                        num_negatives = len(cached_negatives)
                        reconstructed_weights = ([args.one_hop_weight] * one_hop_count) + \
                                                ([args.two_hop_weight] * (num_negatives - one_hop_count))
                        weights.append(reconstructed_weights)
                    elif 'weights' in cached_data:
                        # DGIdb的格式：直接使用权重
                        weights.append(cached_data['weights'])
                    else:
                        # 兼容更旧的格式
                        weights.append([args.two_hop_weight] * len(cached_negatives))

                    hard_negatives.append(cached_negatives)
                else:
                    # 旧格式兼容：只有邻居列表，设置默认权重
                    cached_negatives = cached_data
                    reconstructed_weights = [args.two_hop_weight] * len(cached_negatives)
                    print("⚠️  Using legacy cache format without weight information")
                    hard_negatives.append(cached_negatives)
                    weights.append(reconstructed_weights)

                cache_hits += 1
            else:
                # 缓存未命中，报错
                cache_misses += 1
                raise KeyError(
                    f"❌ Cache miss for (drug={drug_idx}, gene={gene_idx})!\n"
                    f"This key is not in the mixed hard negatives cache.\n"
                    f"Please regenerate the cache file with mixed hard negatives."
                )

        # 输出缓存命中情况（仅在第一个epoch的第一个batch显示）
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            hit_rate = cache_hits / total_requests * 100
            # 在第一个epoch的第一个batch中输出缓存命中率
            if current_epoch == 0 and batch_idx == 0:
                print(f"📊 Mixed Hard Negatives Cache: {cache_hits}/{total_requests} hits ({hit_rate:.1f}% hit rate)")

        return t.tensor(hard_negatives, dtype=t.long), t.tensor(weights, dtype=t.float)

    def compute_real_time_statistics(self, drugs_batch, genes_batch, num_neighbors=5):
        """
        实时计算二跳邻居的四种情况统计（仅在最后5个epoch使用）
        """
        stats = {
            'enough_filtered': 0,
            'some_filtered': 0,
            'enough_random': 0,
            'few_random': 0
        }

        drug_gene_matrix = self.handler.trnLoader.dataset.dokmat

        for i, gene_idx in enumerate(genes_batch):
            gene_idx = gene_idx.item() if hasattr(gene_idx, 'item') else int(gene_idx)
            drug_idx = drugs_batch[i].item() if hasattr(drugs_batch[i], 'item') else int(drugs_batch[i])

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

            # 判断属于哪种情况
            if len(filtered_two_hop) >= num_neighbors:
                stats['enough_filtered'] += 1
            elif len(filtered_two_hop) > 0:
                stats['some_filtered'] += 1
            else:
                # 检查是否有足够的无交互基因
                all_genes = set(range(args.gene))
                drug_connected_genes = set()

                for (drug, gene) in drug_gene_matrix.keys():
                    if drug == drug_idx:
                        drug_connected_genes.add(gene)

                no_interaction_genes = list(all_genes - drug_connected_genes)

                if len(no_interaction_genes) >= num_neighbors:
                    stats['enough_random'] += 1
                else:
                    stats['few_random'] += 1

        return stats

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
        print(f"Estimated memory needed: {(total_neg_samples * num_genes * 4) / 1024 ** 3:.2f} GB")

        if args.multi_gpu and torch.cuda.device_count() > 1:
            # 双GPU并行方案
            gpu_list = [int(x) for x in args.gpu_list.split(',')]
            available_gpus = [gpu for gpu in gpu_list if gpu < torch.cuda.device_count()][:2]  # 只使用前两个GPU
            print(f"Using dual GPUs : {available_gpus}")

            # 检查GPU显存状态
            for gpu_id in available_gpus:
                gpu_memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3
                gpu_memory_allocated = torch.cuda.memory_allocated(gpu_id) / 1024 ** 3
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
    def trainEpoch(self, current_epoch):
        self.model.train()
        trnLoader = self.handler.trnLoader
        if current_epoch == 0:
            print("开始生成普通负样本")
        # 这一步如果在DataHandler中就执行会牺牲随机性
        trnLoader.dataset.positive_genes_dict = self.drug_gene_dict
        trnLoader.dataset.negSampling()  # 这一步非常消耗时间，改多线程生成

        # 重置epoch级别的二跳邻居统计
        self.epoch_two_hop_stats = {
            'enough_filtered': 0,
            'some_filtered': 0,
            'enough_random': 0,
            'few_random': 0
        }

        hard_loss_sum = 0
        epLoss, epPreLoss = 0, 0
        bprLoss, bpr_loss, reg_loss, regLoss, im_loss = 0, 0, 0, 0, 0
        hard_loss, common_loss = 0, 0

        # 数据集长度
        len__ = trnLoader.dataset.__len__()
        steps = len__ // args.batch  # 步数=长度/批次数
        for i, tem in enumerate(trnLoader):
            data = deepcopy(self.handler.torchBiAdj).cuda()
            drugs, genes, labels, negs = tem
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            negs = negs.long().cuda()
            # 得到嵌入用于计算负采样损失
            usrEmbeds, itmEmbeds = self.get_model().forward_gcn(data)
            # 获取正样本和负样本的嵌入
            drugEmbeds = usrEmbeds[drugs]  # 药物嵌入 4096 128
            posEmbeds = itmEmbeds[genes]  # 正样本嵌入 4096 128
            # 获取混合困难负样本和对应权重
            mixed_hard_negatives, neg_weights = self.get_mixed_hard_negatives(
                drugs.cpu(), genes.cpu(), num_neighbors=args.num_two_hop,
                current_epoch=current_epoch, batch_idx=i
            )
            mixed_hard_negatives = mixed_hard_negatives.cuda()
            neg_weights = neg_weights.cuda()
            mixed_hard_neg_embeds = itmEmbeds[mixed_hard_negatives]  # [batch_size, num_two_hop, 128]

            if current_epoch == 0 and i == 0:
                print(f"Mixed hard negatives shape: {mixed_hard_neg_embeds.shape}")
                print(f"Negative weights shape: {neg_weights.shape}")
                print(f"One-hop weight multiplier: {args.one_hop_weight}")
                print(f"Two-hop weight multiplier: {args.two_hop_weight}")
                print(f"One-hop max ratio: {args.one_hop_max_ratio:.1%}")

            # 统计实际权重分布 - 只在第一个batch输出
            one_hop_samples = (neg_weights == args.one_hop_weight).sum().item()
            two_hop_samples = (neg_weights == args.two_hop_weight).sum().item()
            total_samples = neg_weights.numel()
            actual_one_hop_ratio = one_hop_samples / total_samples if total_samples > 0 else 0
            actual_two_hop_ratio = two_hop_samples / total_samples if total_samples > 0 else 0
            if i == 0:  # 只在第一个batch输出
                print(
                    f"Batch weight distribution: {one_hop_samples} one-hop ({actual_one_hop_ratio:.1%}), {two_hop_samples} two-hop ({actual_two_hop_ratio:.1%})")

            negEmbeds = itmEmbeds[negs]  # 负样本嵌入 4096 100 128

            # 基于混合困难负样本的加权负采样损失
            if hasattr(args, 'num_two_hop') and args.num_two_hop > 0:
                if current_epoch == 0 and i == 0:
                    print("Computing weighted BPR loss with mixed hard negatives...")
                # 计算正样本分数
                posScores = innerProduct(drugEmbeds.unsqueeze(1), posEmbeds.unsqueeze(1))  # [batch_size, 1]

                # 计算混合困难负样本分数
                hard_negScores = innerProduct(drugEmbeds.unsqueeze(1),
                                              mixed_hard_neg_embeds)  # [batch_size, num_two_hop]
                # 普通负样本分数
                negScores = innerProduct(drugEmbeds.unsqueeze(1), negEmbeds)  # [batch_size, num_two_hop]

                # 应用权重并求和
                weighted_negScores = hard_negScores * neg_weights  # [batch_size, num_two_hop]
                aggregated_negScore = weighted_negScores.sum(dim=1, keepdim=True)  # [batch_size, 1]

                # 计算加权BPR损失
                scoreDiff1 = posScores - aggregated_negScore  # [batch_size, 1]
                hard_loss_value = -(scoreDiff1).sigmoid().log().sum() / args.batch

                # sum() 不带任何参数时，会对张量的所有元素求和，直接返回一个标量张量（0维张量）
                scoreDiff2 = posScores - negScores  # [batch_size, 1]
                neg_loss_value = -(scoreDiff2).sigmoid().log().sum() / args.batch

                # bpr_loss_value = hard_loss_value
                # bpr_loss_value=hard_loss_value+neg_loss_value*args.common_neg_weight
                if current_epoch == 0 and i == 0:
                    print(f"Positive scores shape: {posScores.shape}, mean: {posScores.mean():.4f}")
                    print(f"Raw negative scores shape: {hard_negScores.shape}, mean: {negScores.mean():.4f}")
                    print(f"Weighted negative scores mean: {weighted_negScores.mean():.4f}")
                    print(f"Aggregated negative score mean: {aggregated_negScore.mean():.4f}")
                    print(f"Average weight per sample: {neg_weights.mean():.4f}")
                    print(f"Score difference mean: {scoreDiff1.mean():.4f}")
                    # print(f"Weighted BPR loss: {bpr_loss_value:.4f}")

                # 梯度累积但分别控制 - 避免损失值差异过大的影响
                regLoss = calcRegLoss(self.model) * args.reg

                # 清零梯度，准备累积
                self.opt.zero_grad()
                # 计算总损失用于记录
                # total_loss = hard_loss_value + neg_loss_value * args.common_neg_weight + regLoss
                total_loss = neg_loss_value * args.common_neg_weight + regLoss

                # 统一参数更新（基于累积的梯度）
                total_loss.backward()
                self.opt.step()

                # 记录损失
                # bpr_loss += float(bpr_loss_value)
                hard_loss += hard_loss_value
                common_loss += neg_loss_value
                reg_loss += float(regLoss)

                # 只在第一个batch输出分离式损失信息
                if i == 0 and (current_epoch == 0):
                    print(f"🔄 Gradient Accumulation Strategy:")
                    print(f"  Hard loss: {hard_loss_value:.4f}")
                    print(f"  Common neg loss: {neg_loss_value:.4f} (ratio: {neg_loss_value / hard_loss_value:.1f}x)")
                    print(f"  Weighted common loss: {neg_loss_value * args.common_neg_weight:.4f}")
                    print(f"  Total loss: {total_loss:.4f}")
                    print(f"  Reg loss (split): {regLoss:.4f} (0.5 each)")

            # 计算交叉熵损失
            ceLoss, sslLoss = self.get_model().calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate)
            sslLoss_reg = sslLoss * args.ssl_reg
            regLoss = calcRegLoss(self.model) * args.reg
            loss = ceLoss + regLoss
            # loss = ceLoss + regLoss + sslLoss_reg
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
            # bprLoss = 0
            regLoss = 0
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['sslLoss'] = sslLoss / steps
        ret['preLoss'] = epPreLoss / steps
        ret['common_neg_loss'] = common_loss / steps
        ret['hard_neg_loss'] = hard_loss / steps
        ret['regLoss'] = reg_loss / steps
        return ret

    # Function to test a single epoch
    # 计算出ACC
    def testEpoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        labels_list = []
        probs_list = []  # 新增：用于存储预测概率
        i = 0
        for tem in tstLoader:  # for i, tem in enumerate(trnLoader)
            i += 1
            drugs, genes, labels = tem
            # 将从数据集获得的张量转移到Gpu上面，默认第一个Gpu
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            # predict(self, adj, drugs, genes)
            pre_logits = self.get_model().predict(self.handler.torchBiAdj, drugs, genes)
            # dim=1 指定了 softmax 操作沿着第 1 维（即类别维度）进行
            pre = F.log_softmax(pre_logits, dim=1)
            # 选出可能性最大的类别，优化GPU->CPU传输
            pre = pre.data.max(1, keepdim=True)[1]
            # 批量转移到CPU，减少传输次数
            pre = pre.detach().cpu()
            labels = labels.detach().cpu()
            epAcc = accuracy_score(labels, pre)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, pre, average='weighted', zero_division=0)
            # precision, recall, f1, _ = precision_recall_fscore_support(labels, pre, average='binary')
            labels_list.append(labels.cpu().numpy())
            probs = F.softmax(pre_logits, dim=1)  # 使用softmax获取概率
            probs_list.append(probs.cpu().detach().numpy())

        all_labels = np.concatenate(labels_list)
        all_probs = np.vstack(probs_list)  # 所有预测概率
        auprc = 0

        # 计算AUC，根据数据集类型
        if args.data == 'DGIdb':
            # 多分类AUC（one-vs-rest）
            # 对于不平衡数据集，通常推荐使用 'macro'
            # 原因：你关心的是所有类别（包括少数类）的表现是否均衡。如果使用
            # 'weighted'，模型可能在多数类上表现好就拉高整体分数，掩盖了对少数类的糟糕预测。
            try:
                auc_score = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='weighted')
                num_classes = all_probs.shape[1]
                y_true_bin = label_binarize(all_labels, classes=range(num_classes))
                auprc = average_precision_score(y_true_bin, all_probs, average='macro')
            except ValueError as e:
                print(f"计算多分类AUC时出错: {e}")
                auc_score = 0.0
        elif args.data == 'DrugBank':
            # 二分类AUC
            try:
                auc_score = roc_auc_score(all_labels, all_probs[:, 1])  # 使用正类的概率
                auprc = average_precision_score(all_labels, all_probs[:, 1])
            except ValueError as e:
                print(f"计算二分类AUC时出错: {e}")
                auc_score = 0.0
        else:
            # 默认尝试二分类
            try:
                auc_score = roc_auc_score(all_labels, all_probs[:, 1] if all_probs.shape[1] > 1 else all_probs)
            except:
                auc_score = 0.0

        # 绘制散点密度图
        # self.plot_scatter(all_labels, all_predictions, epoch)

        # 调用 plot_tSNE，并传递保存路径
        # self.plot_tSNE(all_features, all_labels, epoch, self.plot_save_path)

        ret = {'Acc': epAcc, 'F1': f1, 'AUC': auc_score,
               'precision': precision, 'recall': recall, 'Auprc': auprc}
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
    # else:
    # wandb.init(project='HC', config=args)
    # wandb.run.log_code(".")

    use_cuda = args.gpu >= 0 and t.cuda.is_available()

    if args.multi_gpu and t.cuda.device_count() > 1:
        gpu_list = [int(x) for x in args.gpu_list.split(',')]
        available_gpus = [gpu for gpu in gpu_list if gpu < t.cuda.device_count()]
        print(f"Multi-GPU mode enabled. Available GPUs: {available_gpus}")
        device = 'cuda:{}'.format(available_gpus[0])
        args.device = device
        args.available_gpus = available_gpus
    else:
        device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
        args.device = device

    if use_cuda:
        t.cuda.set_device(device)

    print(f"Primary device: {device}")
    print(f"Total GPU count: {t.cuda.device_count()}")

    if use_cuda:
        for i in range(t.cuda.device_count()):
            gpu_memory = t.cuda.get_device_properties(i).total_memory / 1024 ** 3
            print(f"GPU {i}: {t.cuda.get_device_name(i)}, {gpu_memory:.1f} GB")

    logger.saveDefault = True

    log_dir = 'Logs'
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{args.data}_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)

    log_file = open(log_filepath, 'w')
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    drug_gene_dict, _ = load_drug_gene_dict(args.data)
    if drug_gene_dict is None:
        raise RuntimeError("读取drug_gene_dict缓存失败!!!!")

    log('Load Data')

    hyperparameters_info = (
        "\n" + "=" * 60 + "\n" +
        "🔧 实验超参数配置信息\n" +
        "=" * 60 + "\n" +
        f"📊 数据集 (data): {args.data}\n" +
        f"📈 学习率 (lr): {args.lr}\n" +
        f"🎲 全局负样本数量 (num_neg): {args.num_neg}\n" +
        f"🔗 二跳邻居数量 (num_two_hop): {args.num_two_hop}\n" +
        f"📏 一跳最大比例 (one_hop_max_ratio): {args.one_hop_max_ratio:.1%} ({int(args.num_two_hop * args.one_hop_max_ratio)} max samples)\n" +
        f"⚖️  一跳权重倍数 (one_hop_weight): {args.one_hop_weight}\n" +
        f"⚖️  二跳权重倍数 (two_hop_weight): {args.two_hop_weight}\n" +
        f"⚖️  普通负样本权重 (common_neg_weight): {args.common_neg_weight}\n" +
        "=" * 60 + "\n"
    )
    print(hyperparameters_info)
    log_file.write(hyperparameters_info)

    coach = Coach(handler, drug_gene_dict, log_file)
    config = dict()
    results = list()
    aucMax_list = list()
    outputstr_list = list()

    epoch_list = []
    acc_list = []

    iteration_list = []
    end_acc_list = []
    output_str = None
    it_max = 0
    aucMax = 0
    overall_iteration_best = {met: [] for met in coach.metrics_to_track}

    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed + i
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        if args.data == 'LINCS':
            result = coach.external_test_run()
        else:
            result, output_str, aucMax, iteration_best_metrics = coach.run(i)  # 返回最终测试得到的reses['Acc']

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
        for metric in coach.metrics_to_track:
            metric_value = iteration_best_metrics.get(metric, float('nan'))
            overall_iteration_best[metric].append(metric_value)

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

    if len(valid_results) > 0:
        avg_r = np.mean(np.array(valid_results), axis=0)
        std_r = np.std(valid_results, axis=0)  # 求标准差
        avg_aucMax = np.mean(np.array(valid_aucMax_list), axis=0)

        summary_log = []
        summary_log.append('test results: ')
        summary_log.append(str(results))
        summary_log.append('best epoch results: ')
        summary_log.append(str(outputstr_list))
        summary_log.append('有效结果平均值: {}'.format(avg_r))
        summary_log.append('有效结果平均最大值: {}'.format(avg_aucMax))
        summary_log.append('所有iteration最大值: {}'.format(it_max))
        summary_log.append('有效结果方差: {}'.format(std_r))
        summary_text = '\n'.join(summary_log)
        print(summary_text)
        log_file.write('\n' + summary_text + '\n')

    summary_log = ['\n📊 Iteration-level best metric summary:']
    has_valid_metric = False
    for metric, values in overall_iteration_best.items():
        valid_vals = [val for val in values if not np.isnan(val)]
        if len(valid_vals) == 0:
            summary_log.append(f'  {metric}: 无有效迭代结果')
            continue
        has_valid_metric = True
        avg_best = np.mean(valid_vals)
        max_best = np.max(valid_vals)
        rounded_vals = [round(val, 4) for val in valid_vals]
        summary_log.append(f'  {metric}:')
        summary_log.append(f'    平均最佳 = {avg_best:.4f}')
        summary_log.append(f'    最高最佳 = {max_best:.4f}')
        summary_log.append(f'    所有迭代最佳值 = {rounded_vals}')
    if not has_valid_metric:
        summary_log.append("❌ All iterations resulted in NAN!")

    summary_text = '\n'.join(summary_log)
    print(summary_text)
    log_file.write(summary_text + '\n')

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
            print(
                f"  Iteration {i + 1}: 情况1={stats['enough_filtered']:.1f}%, 情况2={stats['some_filtered']:.1f}%, 情况3={stats['enough_random']:.1f}%, 情况4={stats['few_random']:.1f}%")
    else:
        print("⚠️ No valid iteration statistics available")


    # 保存实验结果到log文件
    def save_results_to_log(overall_iteration_best, coach):
        """
        将实验结果保存到log文件
        文件名格式: MMDD_HHMM_DatasetName.txt
        文件路径: log目录下
        """
        now = datetime.now()
        log_filename = now.strftime("%m%d_%H%M") + f"_{args.data}.txt"
        # 获取当前脚本所在目录路径
        log_dir = os.path.join(os.path.dirname(__file__), '..', 'log')

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_filepath = os.path.join(log_dir, log_filename)

        with open(log_filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("实验结果记录\n")
            f.write("=" * 60 + "\n")
            f.write(f"数据集: {args.data}\n")
            f.write(f"时间: {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("超参数配置\n")
            f.write("=" * 60 + "\n")
            f.write(f"学习率 (lr): {args.lr}\n")
            f.write(f"全局负样本数量 (num_neg): {args.num_neg}\n")
            f.write(f"二跳邻居数量 (num_two_hop): {args.num_two_hop}\n")
            f.write(
                f"一跳最大比例 (one_hop_max_ratio): {args.one_hop_max_ratio:.1%} ({int(args.num_two_hop * args.one_hop_max_ratio)} max samples)\n")
            f.write(f"一跳权重倍数 (one_hop_weight): {args.one_hop_weight}\n")
            f.write(f"二跳权重倍数 (two_hop_weight): {args.two_hop_weight}\n")
            f.write(f"普通负样本权重 (common_neg_weight): {args.common_neg_weight}\n")
            f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("评估指标结果\n")
            f.write("=" * 60 + "\n")
            f.write("\n")

            for metric, values in overall_iteration_best.items():
                valid_vals = [val for val in values if not np.isnan(val)]
                if len(valid_vals) == 0:
                    f.write(f"{metric}:\n")
                    f.write(f"  平均最佳: N/A\n")
                    f.write(f"  最高最佳: N/A\n")
                    f.write(f"  所有迭代最佳值: []\n")
                else:
                    avg_best = np.mean(valid_vals)
                    max_best = np.max(valid_vals)
                    rounded_vals = [round(val, 4) for val in valid_vals]
                    f.write(f"{metric}:\n")
                    f.write(f"  平均最佳 = {avg_best:.4f}\n")
                    f.write(f"  最高最佳 = {max_best:.4f}\n")
                    f.write(f"  所有迭代最佳值 = {rounded_vals}\n")
                f.write("\n")

            f.write("=" * 60 + "\n")

        print(f"✅ 实验结果已保存到: {log_filepath}")
        return log_filepath


    save_results_to_log(overall_iteration_best, coach)

    log_file.close()
    print(f"📝 日志文件已关闭: {log_filepath}")

    # results_parent_path = "D:\\桌面\\研\\论文\\实验代码\\DGCL-main\\DGCL-main\\results"
    # # results_parent_path = os.path.join(wandb.run.dir, 'results')
    # if not os.path.exists(results_parent_path):
    #     os.mkdir(results_parent_path)
    # np.savetxt('{}/{}_result.txt'.format(results_parent_path, args.data), np.array(results), delimiter=",", fmt='%f')

    print('result saved!!!')
    # wandb.finish()
