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


# Function to set random seed for reproducibility 2675
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


# Function to log all error cases from all iterations
def log_all_error_cases(all_iteration_errors, log_file, log_filepath):
    """
    将所有iteration的错误案例导出为CSV文件，并在日志中记录统计信息

    参数:
    all_iteration_errors: 列表，每个元素包含一个iteration的错误信息
    log_file: 已打开的日志文件对象
    log_filepath: 日志文件路径（用于生成CSV文件路径）
    """
    import csv

    # 生成CSV文件路径（与日志文件同目录，同名但扩展名为.csv）
    csv_filepath = log_filepath.replace('.txt', '_error_cases.csv')

    # 统计信息
    total_errors = sum(len(iter_info['error_cases']) for iter_info in all_iteration_errors)
    avg_errors = total_errors / len(all_iteration_errors) if all_iteration_errors else 0

    # 确定类别数量（从第一个有错误案例的iteration中获取）
    num_classes = 0
    for iter_info in all_iteration_errors:
        if len(iter_info['error_cases']) > 0:
            num_classes = len(iter_info['error_cases'][0].get('prob_distribution', []))
            break

    # 写入CSV文件
    try:
        with open(csv_filepath, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)

            # 动态生成CSV表头（包含概率分布列）
            header = ['Iteration', 'Best_Epoch', 'Best_ACC', '药物ID', '基因ID', 
                     '预测标签', '真实标签', '最大概率', '预测类概率', '真实类概率']
            if num_classes > 0:
                header.extend([f'类别{i}概率' for i in range(num_classes)])
            csv_writer.writerow(header)

            # 遍历每个iteration
            for iter_info in all_iteration_errors:
                iteration = iter_info['iteration']
                best_epoch = iter_info['best_epoch']
                best_acc = iter_info['best_acc']
                error_cases = iter_info['error_cases']

                if len(error_cases) > 0:
                    # 写入该iteration的所有错误案例
                    for case in error_cases:
                        row = [
                            iteration,
                            best_epoch,
                            f"{best_acc:.4f}",
                            case['drug'],
                            case['gene'],
                            case['predicted'],
                            case['actual'],
                            f"{case.get('max_prob', 0):.4f}",
                            f"{case.get('predicted_prob', 0):.4f}",
                            f"{case.get('actual_prob', 0):.4f}"
                        ]
                        # 添加完整概率分布
                        prob_dist = case.get('prob_distribution', [])
                        row.extend([f"{p:.4f}" for p in prob_dist])
                        csv_writer.writerow(row)
                else:
                    # 如果没有错误案例，写入一行说明
                    row = [iteration, best_epoch, f"{best_acc:.4f}"] + ['N/A'] * (len(header) - 3)
                    csv_writer.writerow(row)

                # 在每个iteration后添加空行（用于分隔）
                csv_writer.writerow([])

        print(f"✅ 错误案例已导出到CSV文件: {csv_filepath}")

    except Exception as e:
        print(f"❌ 导出CSV文件失败: {e}")
        import traceback
        traceback.print_exc()
        csv_filepath = None

    # 在日志文件中记录统计信息
    error_log = [
        "\n" + "=" * 60,
        "🔍 错误案例分析（每个Iteration的最佳Epoch）",
        "=" * 60 + "\n"
    ]

    # 添加CSV文件路径信息
    if csv_filepath:
        error_log.append(f"📄 详细错误案例已导出到CSV文件:")
        error_log.append(f"   {csv_filepath}\n")

    # 收集所有错误案例的概率统计
    all_max_probs = []
    all_predicted_probs = []
    all_actual_probs = []
    
    # 为每个iteration添加简要统计
    for iter_info in all_iteration_errors:
        iteration = iter_info['iteration']
        best_epoch = iter_info['best_epoch']
        best_acc = iter_info['best_acc']
        error_cases = iter_info['error_cases']

        error_log.append(f"\nIteration {iteration} - Best Epoch: {best_epoch} - ACC: {best_acc:.4f}")
        error_log.append(f"  错误案例数: {len(error_cases)}")
        
        # 统计该iteration的概率信息
        if len(error_cases) > 0:
            iter_max_probs = [case.get('max_prob', 0) for case in error_cases]
            iter_pred_probs = [case.get('predicted_prob', 0) for case in error_cases]
            iter_actual_probs = [case.get('actual_prob', 0) for case in error_cases]
            
            all_max_probs.extend(iter_max_probs)
            all_predicted_probs.extend(iter_pred_probs)
            all_actual_probs.extend(iter_actual_probs)
            
            error_log.append(f"  平均最大概率: {np.mean(iter_max_probs):.4f}")
            error_log.append(f"  平均预测类概率: {np.mean(iter_pred_probs):.4f}")
            error_log.append(f"  平均真实类概率: {np.mean(iter_actual_probs):.4f}")

    # 总体统计信息
    error_log.append(f"\n{'=' * 60}")
    error_log.append("📊 总体统计")
    error_log.append(f"{'=' * 60}")
    error_log.append(f"总Iteration数: {len(all_iteration_errors)}")
    error_log.append(f"总错误案例数: {total_errors}")
    error_log.append(f"平均每个Iteration错误数: {avg_errors:.2f}")
    
    # 添加概率分布统计
    if len(all_max_probs) > 0:
        error_log.append(f"\n📈 概率分布统计（所有错误案例）:")
        error_log.append(f"  平均最大概率: {np.mean(all_max_probs):.4f} ± {np.std(all_max_probs):.4f}")
        error_log.append(f"  平均预测类概率: {np.mean(all_predicted_probs):.4f} ± {np.std(all_predicted_probs):.4f}")
        error_log.append(f"  平均真实类概率: {np.mean(all_actual_probs):.4f} ± {np.std(all_actual_probs):.4f}")
        error_log.append(f"  最大概率范围: [{np.min(all_max_probs):.4f}, {np.max(all_max_probs):.4f}]")
        error_log.append(f"  预测类概率范围: [{np.min(all_predicted_probs):.4f}, {np.max(all_predicted_probs):.4f}]")
        error_log.append(f"  真实类概率范围: [{np.min(all_actual_probs):.4f}, {np.max(all_actual_probs):.4f}]")
    
    error_log.append("=" * 60 + "\n")

    error_text = '\n'.join(error_log)
    print(error_text)
    log_file.write(error_text + '\n')


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

        # 全局最佳ACC（跨所有iteration与epoch）
        self.best_global_acc = float('-inf')
        self.best_global_info = None

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
            # 标量指标按浮点数输出，列表/数组等复杂类型直接转为字符串
            if np.isscalar(val):
                ret += '%s = %.4f, ' % (metric, val)
                tem = name + metric
                if save and tem in self.metrics:
                    self.metrics[tem].append(val)
            else:
                # 对于如 PerClassAcc 这样的列表型结果，只做打印展示，不参与历史曲线统计
                ret += '%s = %s, ' % (metric, str(val))
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

    def random_sample_nonzero(self, arr_list, k, pad_value=-1):
        result_list = []
        for row in arr_list:
            row_np = np.array(row)
            valid_indices = np.where(row_np != -1)[0]
            if len(valid_indices) == 0:
                result_row = [pad_value] * k
            elif len(valid_indices) < k:
                repeats_needed = k // len(valid_indices) + 1
                repeated_elements = np.tile(row_np[valid_indices], repeats_needed)
                result_row = repeated_elements[:k].tolist()
            else:
                result_row = np.random.choice(row_np[valid_indices], k, replace=False).tolist()
            result_list.append(result_row)
        return result_list

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

        self.best_metrics = {met: {'value': float('-inf'), 'epoch': -1} for met in self.metrics_to_track}

        def update_best_metrics(result_dict, epoch_idx):
            for metric in self.metrics_to_track:
                metric_value = result_dict.get(metric)
                if metric_value is None or np.isnan(metric_value):
                    continue
                if metric_value > self.best_metrics[metric]['value']:
                    self.best_metrics[metric]['value'] = metric_value
                    self.best_metrics[metric]['epoch'] = epoch_idx

        aucMax = 0
        bestEpoch = 0
        best_errors_in_iteration = []  # 新增：跟踪iteration内最佳epoch的错误案例
        best_acc_in_iteration = 0  # 新增：跟踪iteration内最佳ACC
        best_per_class_acc_in_iteration = None  # 新增：跟踪iteration内最佳epoch的按类别ACC

        test_r = {met: float('nan') for met in self.metrics_to_track}

        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch(ep, i)
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
                    if self.log_file:
                        self.log_file.write(output_str + '\n')
                    # 返回当前最佳结果，继续下一个iteration
                    return aucMax, output_str, aucMax, nan_metrics, [], bestEpoch, aucMax
                else:
                    output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. No valid test results yet.'
                    print(output_str)
                    if self.log_file:
                        self.log_file.write(output_str + '\n')
                    print("➡️ Continuing to next iteration...")
                    # 没有有效结果，返回NAN，但继续下一个iteration
                    return float('nan'), output_str, float('nan'), nan_metrics, [], -1, 0.0

            # 记录训练结果
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses, error_cases = self.testEpoch()  # 接收错误案例
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
                        if self.log_file:
                            self.log_file.write(output_str + '\n')
                        # 返回当前最佳结果，继续下一个iteration
                        return aucMax, output_str, aucMax, nan_metrics, best_errors_in_iteration, bestEpoch, best_acc_in_iteration
                    else:
                        output_str = f'🛑 Iteration stopped due to NAN at epoch {ep}. No valid test results yet.'
                        print(output_str)
                        if self.log_file:
                            self.log_file.write(output_str + '\n')
                        print("➡️ Continuing to next iteration...")
                        # 没有有效结果，返回NAN，但继续下一个iteration
                        return float('nan'), output_str, float('nan'), nan_metrics, [], -1, 0.0

                if reses['Acc'] > aucMax:
                    aucMax = reses['Acc']
                    bestEpoch = ep

                # 更新iteration内的最佳结果
                if reses['Acc'] > best_acc_in_iteration:
                    best_acc_in_iteration = reses['Acc']
                    best_errors_in_iteration = error_cases
                    best_per_class_acc_in_iteration = reses.get('PerClassAcc', None)

                # 更新跨所有iteration与epoch的全局最佳ACC模型
                if reses['Acc'] > self.best_global_acc:
                    self.best_global_acc = reses['Acc']
                    self.best_global_info = {
                        'iteration': config.get('iteration', i + 1),
                        'epoch': ep,
                        'Acc': reses['Acc']
                    }
                    # 始终将全局最佳ACC模型保存为 best_acc.pkl
                    self.save_model('best_acc')

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

        reses, final_error_cases = self.testEpoch()
        log(self.makePrint('Test', args.epoch, reses, True))
        update_best_metrics(reses, args.epoch)

        # 如果最终测试的ACC更高，更新最佳错误案例以及全局最佳模型
        if reses['Acc'] > best_acc_in_iteration:
            best_acc_in_iteration = reses['Acc']
            best_errors_in_iteration = final_error_cases
            best_per_class_acc_in_iteration = reses.get('PerClassAcc', None)

        if reses['Acc'] > self.best_global_acc:
            self.best_global_acc = reses['Acc']
            self.best_global_info = {
                'iteration': config.get('iteration', i + 1),
                'epoch': args.epoch,
                'Acc': reses['Acc']
            }
            self.save_model('best_acc')

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
        acc_best_epoch = self.best_metrics['Acc']['epoch']
        if acc_best_epoch != -1:
            aucMax = self.best_metrics['Acc']['value']
            bestEpoch = acc_best_epoch
        else:
            aucMax = float('nan')
            bestEpoch = -1
        output_str = f'Best epoch : {bestEpoch} , ACC : {round(aucMax, 4) if not np.isnan(aucMax) else "N/A"}'
        best_lines.append(output_str)

        for metric in self.metrics_to_track:
            if metric == 'Acc':
                continue
            best_value = self.best_metrics[metric]['value']
            best_epoch = self.best_metrics[metric]['epoch']
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
            metric: (self.best_metrics[metric]['value'] if self.best_metrics[metric]['epoch'] != -1 else float('nan'))
            for metric in self.metrics_to_track
        }

        return (
            reses['Acc'],
            output_str,
            aucMax,
            iteration_best_metrics,
            best_errors_in_iteration,
            bestEpoch,
            best_acc_in_iteration,
            best_per_class_acc_in_iteration
        )

    # Function to prepare the model and optimizer
    def prepareModel(self):
        self.model = Model().cuda()
        mode_msg = "模型模式：联合编码文本 + 结构特征" if getattr(self.model, 'use_joint_encoding', False) \
            else ("模型模式：实体级LLM文本嵌入 + 结构特征" if getattr(self.model, 'use_text_features', False) \
                 else "模型模式：仅使用结构嵌入")
        print(mode_msg)
        log(mode_msg)
        if self.log_file:
            self.log_file.write(mode_msg + '\n')
        self.is_data_parallel = False

        # 如果启用多GPU并且有多个GPU可用，使用DataParallel
        if args.multi_gpu and t.cuda.device_count() > 1:
            gpu_list = [int(x) for x in args.gpu_list.split(',')]
            available_gpus = [gpu for gpu in gpu_list if gpu < t.cuda.device_count()]
            if len(available_gpus) > 1:
                print(f"Wrapping model with DataParallel on GPUs: {available_gpus}")
                self.model = t.nn.DataParallel(self.model, device_ids=available_gpus)
                self.is_data_parallel = True

        if getattr(self.get_model(), 'use_joint_encoding', False):
            self.get_model().train_pair_to_row = getattr(self.handler, 'train_pair_to_row', None)
            self.get_model().test_pair_to_row = getattr(self.handler, 'test_pair_to_row', None)

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
    def trainEpoch(self, current_epoch, iteration_count):
        self.model.train()
        trnLoader = self.handler.trnLoader
        if current_epoch == 0:
            print("开始生成普通负样本")
            print(iteration_count)
        # 这一步如果在DataHandler中就执行会牺牲随机性
        trnLoader.dataset.positive_genes_dict = self.drug_gene_dict
        trnLoader.dataset.negSampling()  # 这一步非常消耗时间，改多线程生成
        if (current_epoch == 0 and (iteration_count == 0)):
            print("make_neg_begin")
            trnLoader.dataset.negMul_gene()
            print("make_neg_gene_over")
            trnLoader.dataset.negMul_drug()
            print("make_neg_drug_over")
            trnLoader.dataset.padded_matrix()
            print("make_neg_over")
            print(args.num_neg)
            print(args.num_neg_mul)

        num_neg_mul = getattr(args, 'num_neg_mul', 0)
        enable_drug_local = (args.data == 'DGIdb') or (getattr(args, 'is_use', 0) != 0)

        def _to_numpy(arr):
            if isinstance(arr, np.ndarray):
                return arr
            if t.is_tensor(arr):
                return arr.detach().cpu().numpy()
            return np.asarray(arr)

        # 重置epoch级别的二跳邻居统计
        self.epoch_two_hop_stats = {
            'enough_filtered': 0,
            'some_filtered': 0,
            'enough_random': 0,
            'few_random': 0
        }

        epLoss, epPreLoss = 0, 0
        bprLoss, bpr_loss, reg_loss, regLoss, im_loss = 0, 0, 0, 0, 0
        hard_loss, common_loss = 0, 0
        loss_mult_loss = 0

        # 数据集长度
        len__ = trnLoader.dataset.__len__()
        steps = len__ // args.batch  # 步数=长度/批次数
        for i, tem in enumerate(trnLoader):
            data = deepcopy(self.handler.torchBiAdj).cuda()
            if len(tem) == 8:
                drugs, genes, labels, negs, negs_mul_genes_labels, mul_genes, negs_mul_drugs_labels, mul_drugs = tem
            else:
                drugs, genes, labels, negs = tem
                negs_mul_genes_labels = mul_genes = negs_mul_drugs_labels = mul_drugs = None
            drugs = drugs.long().cuda()
            genes = genes.long().cuda()
            labels = labels.long().cuda()
            negs = negs.long().cuda()
            # 得到嵌入用于计算负采样损失
            usrEmbeds, itmEmbeds = self.get_model().forward_gcn(data)
            # 获取正样本和负样本的嵌入
            drugEmbeds = usrEmbeds[drugs]  # 药物嵌入 4096 128
            posEmbeds = itmEmbeds[genes]  # 正样本嵌入 4096 128

            local_negatives_ready = (
                    num_neg_mul != 0 and
                    mul_genes is not None and
                    negs_mul_genes_labels is not None
            )

            if local_negatives_ready:
                labels_np = _to_numpy(labels)
                mul_genes_np = _to_numpy(mul_genes)
                negs_mul_genes_labels_np = _to_numpy(negs_mul_genes_labels)
                bool_matrix_gene = (negs_mul_genes_labels_np == labels_np[:, None])
                inverted_bool_gene = np.logical_not(bool_matrix_gene)
                negs_mul_genes_list = [
                    row[mask] for row, mask in zip(mul_genes_np, inverted_bool_gene)
                ]
                negs_mul_genes_list = self.random_sample_nonzero(negs_mul_genes_list, num_neg_mul, -1)
                negs_mul_genes = t.tensor(negs_mul_genes_list, device=drugs.device).long()

                negs_mul_drugs = None
                if enable_drug_local and mul_drugs is not None and negs_mul_drugs_labels is not None:
                    mul_drugs_np = _to_numpy(mul_drugs)
                    negs_mul_drugs_labels_np = _to_numpy(negs_mul_drugs_labels)
                    bool_matrix_drug = (negs_mul_drugs_labels_np == labels_np[:, None])
                    inverted_bool_drug = np.logical_not(bool_matrix_drug)
                    negs_mul_drugs_list = [
                        row[mask] for row, mask in zip(mul_drugs_np, inverted_bool_drug)
                    ]
                    negs_mul_drugs_list = self.random_sample_nonzero(negs_mul_drugs_list, num_neg_mul, -1)
                    negs_mul_drugs = t.tensor(negs_mul_drugs_list, device=drugs.device).long()

                zero_item = itmEmbeds.new_zeros(1, itmEmbeds.shape[1])
                itmEmbeds_local = t.cat((itmEmbeds, zero_item), dim=0)
                sentinel_item_idx = itmEmbeds_local.size(0) - 1
                negs_mul_genes = negs_mul_genes.clone()
                negs_mul_genes[negs_mul_genes < 0] = sentinel_item_idx
                negEmbeds_gene_local = itmEmbeds_local[negs_mul_genes]

                usrEmbeds_local = usrEmbeds
                negEmbeds_drug_local = None
                if negs_mul_drugs is not None:
                    zero_user = usrEmbeds.new_zeros(1, usrEmbeds.shape[1])
                    usrEmbeds_local = t.cat((usrEmbeds, zero_user), dim=0)
                    sentinel_usr_idx = usrEmbeds_local.size(0) - 1
                    negs_mul_drugs = negs_mul_drugs.clone()
                    negs_mul_drugs[negs_mul_drugs < 0] = sentinel_usr_idx
                    negEmbeds_drug_local = usrEmbeds_local[negs_mul_drugs]

                ancEmbeds = usrEmbeds[drugs]
                posScores_local = innerProduct(ancEmbeds.unsqueeze(1), posEmbeds.unsqueeze(1))
                negScores_gene_local = innerProduct(posEmbeds.unsqueeze(1), negEmbeds_gene_local)

                # 局部负样本损失：从BPR改为BCE
                # BPR损失（已注释）:
                scoreDiff_local = posScores_local - negScores_gene_local
                if negEmbeds_drug_local is not None:
                    negScores_drug_local = innerProduct(ancEmbeds.unsqueeze(1), negEmbeds_drug_local)
                    scoreDiff_local = scoreDiff_local - negScores_drug_local
                local_bpr_loss = -(scoreDiff_local).sigmoid().log().sum() / args.batch

                # BCE损失（标准公式）：
                # if negEmbeds_drug_local is not None:
                #     negScores_drug_local = innerProduct(ancEmbeds.unsqueeze(1), negEmbeds_drug_local)
                #     # 合并gene和drug负样本：[batch_size, num_neg_mul*2]
                #     all_neg_scores = t.cat([negScores_gene_local, negScores_drug_local], dim=1)
                #     local_bce_loss = -t.mean(F.logsigmoid(posScores_local) + F.logsigmoid(-all_neg_scores))
                # else:
                #     # 只有gene负样本
                #     local_bce_loss = -t.mean(F.logsigmoid(posScores_local) + F.logsigmoid(-negScores_gene_local))

                loss_mult_loss += float(local_bpr_loss)

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
                negScores = innerProduct(drugEmbeds.unsqueeze(1), negEmbeds)

                # 困难负样本损失：先计算分数差，再对差值加权（一跳和二跳使用不同权重）
                # posScores会自动广播: [batch_size, 1] -> [batch_size, num_two_hop]
                scoreDiff1 = posScores - hard_negScores  # [batch_size, num_two_hop]

                # 将权重乘到差值上（区分一跳和二跳）
                # 一跳样本：(s_pos - s_neg) × 2.0
                # 二跳样本：(s_pos - s_neg) × 1.2
                weighted_scoreDiff1 = scoreDiff1 * neg_weights  # [batch_size, num_two_hop]

                hard_loss_value = -(weighted_scoreDiff1).sigmoid().log().sum() / args.batch

                # BCE损失（标准公式）：
                # hard_loss_value = -t.mean(F.logsigmoid(posScores) + F.logsigmoid(-hard_negScores))

                # sum() 不带任何参数时，会对张量的所有元素求和，直接返回一个标量张量（0维张量）
                # 普通负样本损失：从BPR改为BCE
                # BPR损失
                scoreDiff2 = posScores - negScores  # [batch_size, 1]
                neg_loss_value = -(scoreDiff2).sigmoid().log().sum() / args.batch

                # BCE损失：
                # neg_loss_value = -t.mean(F.logsigmoid(posScores) + F.logsigmoid(-negScores))

                if not t.isfinite(hard_loss_value):
                    warn_msg = (
                        f"困难负采样NAN [Warn] Epoch {current_epoch} batch {i}: "
                        f"non-finite hard loss detected ({hard_loss_value.item()}). "
                        f"Forcing hard_loss_value=0."
                    )
                    print(warn_msg)
                    if self.log_file:
                        self.log_file.write(warn_msg + '\n')
                    hard_loss_value = hard_loss_value.new_tensor(0.0)

                if not t.isfinite(neg_loss_value):
                    warn_msg = (
                        f"普通负采样NAN,[Warn] Epoch {current_epoch} batch {i}: "
                        f"non-finite common neg loss detected ({neg_loss_value.item()}). "
                        f"Forcing neg_loss_value=0."
                    )
                    print(warn_msg)
                    if self.log_file:
                        self.log_file.write(warn_msg + '\n')
                    neg_loss_value = neg_loss_value.new_tensor(0.0)

                # bpr_loss_value = hard_loss_value
                # bpr_loss_value=hard_loss_value+neg_loss_value*args.common_neg_weight
                if current_epoch == 0 and i == 0:
                    print(f"Positive scores shape: {posScores.shape}, mean: {posScores.mean():.4f}")
                    print(f"Hard negative scores shape: {hard_negScores.shape}, mean: {hard_negScores.mean():.4f}")
                    print(f"Common negative scores shape: {negScores.shape}, mean: {negScores.mean():.4f}")
                    print(f"Score difference (scoreDiff1) mean: {scoreDiff1.mean():.4f}")
                    print(f"Weighted score difference mean: {weighted_scoreDiff1.mean():.4f}")
                    print(f"Average weight per sample: {neg_weights.mean():.4f}")
                    print(f"Hard loss: {hard_loss_value:.4f}, Common loss: {neg_loss_value:.4f}")
                    print(f"Hard loss: {hard_loss_value:.4f}, Common loss: {neg_loss_value:.4f}")

                # 梯度累积但分别控制 - 避免损失值差异过大的影响
                regLoss = calcRegLoss(self.model) * args.reg

                # 清零梯度，准备累积
                self.opt.zero_grad()
                # 计算总损失用于记录
                if args.data == 'DGIdb':
                    total_loss = hard_loss_value + neg_loss_value + regLoss + local_bpr_loss
                else:
                    total_loss = hard_loss_value + neg_loss_value + regLoss

                # total_loss = neg_loss_value * args.common_neg_weight + regLoss

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
            regLoss = calcRegLoss(self.model) * args.reg
            # loss = ceLoss + regLoss
            loss = ceLoss + regLoss + sslLoss
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

            # 记录损失
            # bpr_loss += float(bpr_loss_value)
            hard_loss += hard_loss_value
            common_loss += neg_loss_value
            reg_loss += float(regLoss)

        ret = dict()
        ret['Loss'] = epLoss / steps
        # ret['sslLoss'] = sslLoss / steps
        ret['preLoss'] = epPreLoss / steps
        ret['common_neg_loss'] = common_loss / steps
        ret['hard_neg_loss'] = hard_loss / steps
        ret['regLoss'] = reg_loss / steps
        if num_neg_mul != 0:
            ret['local_neg_loss'] = loss_mult_loss / steps
        return ret

    def testEpoch(self):
        self.model.eval()
        tstLoader = self.handler.tstLoader
        labels_list = []
        probs_list = []  # 新增：用于存储预测概率
        error_cases = []  # 新增：存储错误案例
        i = 0
        num_classes = None
        correct_per_class = None
        total_per_class = None
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
            pre_cpu = pre.squeeze().detach().cpu()
            labels_cpu = labels.detach().cpu()
            drugs_cpu = drugs.detach().cpu()
            genes_cpu = genes.detach().cpu()

            # 初始化按类别统计的数组
            if num_classes is None:
                num_classes = pre_logits.size(1)
                correct_per_class = np.zeros(num_classes, dtype=np.int64)
                total_per_class = np.zeros(num_classes, dtype=np.int64)

            # 计算概率分布（用于错误案例分析）
            probs = F.softmax(pre_logits, dim=1)  # 使用softmax获取概率
            probs_cpu = probs.cpu().detach().numpy()

            # 收集错误案例（包含概率分布信息）
            incorrect_mask = (pre_cpu != labels_cpu)
            for idx in range(len(labels_cpu)):
                if incorrect_mask[idx]:
                    prob_dist = probs_cpu[idx]
                    max_prob = np.max(prob_dist)
                    predicted_class = pre_cpu[idx].item()
                    actual_class = labels_cpu[idx].item()
                    
                    error_cases.append({
                        'drug': drugs_cpu[idx].item(),
                        'gene': genes_cpu[idx].item(),
                        'predicted': predicted_class,
                        'actual': actual_class,
                        'max_prob': max_prob,
                        'predicted_prob': prob_dist[predicted_class],
                        'actual_prob': prob_dist[actual_class],
                        'prob_distribution': prob_dist.tolist()
                    })

            # 统计每个类别的正确数和总数
            for cls in range(num_classes):
                cls_mask = (labels_cpu == cls)
                cls_count = cls_mask.sum().item()
                if cls_count > 0:
                    total_per_class[cls] += cls_count
                    correct_per_class[cls] += (pre_cpu[cls_mask] == cls).sum().item()

            epAcc = accuracy_score(labels_cpu, pre_cpu)
            # zero_division=0  用于处理 “分母为零”导致指标无法计算 的情况
            if args.data == 'DGIdb':
                precision, recall, f1, _ = precision_recall_fscore_support(labels_cpu, pre_cpu, average='weighted', zero_division=0)
            else:
                precision, recall, f1, _ = precision_recall_fscore_support(labels_cpu, pre_cpu, average='binary', zero_division=0)
            labels_list.append(labels_cpu.numpy())
            probs_list.append(probs_cpu)

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

        # 计算按类别的准确率列表
        if num_classes is not None:
            per_class_acc = [
                (correct_per_class[c] / total_per_class[c]) if total_per_class[c] > 0 else float('nan')
                for c in range(num_classes)
            ]
        else:
            per_class_acc = []

        ret = {
            'Acc': epAcc,
            'F1': f1,
            'AUC': auc_score,
            'precision': precision,
            'recall': recall,
            'Auprc': auprc,
            'PerClassAcc': per_class_acc
        }
        return ret, error_cases

    # Function to load a pre-trained model
    def loadModel(self):
        """
        加载已经训练好的模型权重。
        - 目录结构（相对项目根目录）: Models/ckl/<dataset>/
        - 文件名: <args.load_model>.pkl  （例如: best_acc.pkl）
        """
        # 计算模型目录（相对路径 + 绝对路径，仅绝对路径用于实际读写和日志）
        code_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(code_dir, '..'))
        rel_model_dir = os.path.join('Models', 'ckl', str(args.data))
        abs_model_dir = os.path.join(project_root, rel_model_dir)

        # 构造检查点文件名
        ckpt_name = args.load_model
        if not ckpt_name.endswith('.pkl'):
            ckpt_name = ckpt_name + '.pkl'
        ckpt_path = os.path.join(abs_model_dir, ckpt_name)

        # 实际加载
        state_dict = t.load(ckpt_path, map_location=args.device)
        if self.is_data_parallel:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

        load_msg = f"Model Loaded from: {os.path.abspath(ckpt_path)}"
        print(load_msg)
        log(load_msg)
        if self.log_file:
            self.log_file.write(load_msg + '\n')

    # Function to save the trained model
    def save_model(self, model_path):
        """
        保存当前模型为给定名称的检查点。
        - 相对目录: Models/ckl/<dataset>/
        - 实际保存文件名: <model_path>.pkl  （示例: best_acc.pkl, iter1_best.pkl）
        注意: 代码内部使用相对目录构造，日志中输出绝对路径。
        """
        code_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(code_dir, '..'))
        rel_model_dir = os.path.join('Models', 'ckl', str(args.data))
        abs_model_dir = os.path.join(project_root, rel_model_dir)
        os.makedirs(abs_model_dir, exist_ok=True)

        ckpt_name = model_path
        if not ckpt_name.endswith('.pkl'):
            ckpt_name = ckpt_name + '.pkl'
        ckpt_path = os.path.join(abs_model_dir, ckpt_name)

        # 处理DataParallel包装的模型保存
        if self.is_data_parallel:
            t.save(self.model.module.state_dict(), ckpt_path)
        else:
            t.save(self.model.state_dict(), ckpt_path)

        save_msg = f"Model checkpoint saved to: {os.path.abspath(ckpt_path)}"
        print(save_msg)
        log(save_msg)
        if self.log_file:
            self.log_file.write(save_msg + '\n')


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

    # 日志根目录固定为指定路径，并在其下按数据集名称创建子目录
    base_log_dir = "/mnt/data/huangpeng/DGCL/DGCL-main/log"
    dataset_log_dir = os.path.join(base_log_dir, str(args.data))
    os.makedirs(dataset_log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    log_filename = f"{timestamp}_{args.data}.txt"
    log_filepath = os.path.join(dataset_log_dir, log_filename)

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
            f"📐 嵌入维度 (latdim): {args.latdim}\n" +
            f"🧩 GNN 层数 (gnn_layer): {args.gnn_layer}\n" +
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
    best_epochs_per_metric = {met: [] for met in coach.metrics_to_track}
    all_iteration_errors = []  # 新增：存储每个iteration的错误案例（可通过超参数控制是否启用）

    for i in range(args.iteration):
        print('{}-th iteration'.format(i + 1))
        seed = args.seed
        config['seed'] = seed
        config['iteration'] = i + 1
        set_seed(seed)
        if args.data == 'LINCS':
            result = coach.external_test_run()
            best_errors = []
            best_epoch = -1
            best_acc = 0.0
        else:
            (
                result,
                output_str,
                aucMax,
                iteration_best_metrics,
                best_errors,
                best_epoch,
                best_acc,
                best_per_class_acc
            ) = coach.run(i)  # 返回最终测试得到的reses['Acc']

        # 保存当前iteration的错误案例信息（可关闭）
        if getattr(args, 'enable_error_logging', False):
            all_iteration_errors.append({
                'iteration': i + 1,
                'best_epoch': best_epoch,
                'best_acc': best_acc,
                'error_cases': best_errors
            })

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
        # 记录每个iteration在best_epoch上的按类别ACC和各指标best值
        if 'per_class_acc_per_iteration' not in locals():
            per_class_acc_per_iteration = []
            best_metric_values_per_iteration = []
        per_class_acc_per_iteration.append(best_per_class_acc)
        best_metric_values_per_iteration.append(iteration_best_metrics)
        for metric in coach.metrics_to_track:
            best_val = iteration_best_metrics[metric]
            best_ep = coach.best_metrics[metric]['epoch'] if best_val is not None and not np.isnan(best_val) else -1
            overall_iteration_best[metric].append(best_val)
            best_epochs_per_metric[metric].append(best_ep)

        # 只有非NAN的aucMax才更新it_max
        if not np.isnan(aucMax) and aucMax > it_max:
            it_max = aucMax
            best_iteration_index = i
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
        epochs = best_epochs_per_metric[metric]
        valid_pairs = [(v, e) for v, e in zip(values, epochs) if not np.isnan(v)]
        max_best_val, max_best_epoch = max(valid_pairs, key=lambda item: item[0])

        summary_log.append(f'  {metric}:')
        summary_log.append(f'    平均最佳 = {avg_best:.4f}')
        summary_log.append(f'    最高最佳 = {max_best_val:.4f} (at epoch {max_best_epoch})')
        # 格式化列表输出
        formatted_values = [f'{v:.4f} (epoch {e})' for v, e in zip(values, epochs)]
        summary_log.append(f"    所有迭代最佳值 = [{', '.join(formatted_values)}]")
    if not has_valid_metric:
        summary_log.append("❌ All iterations resulted in NAN!")

    summary_text = '\n'.join(summary_log)
    print(summary_text)
    log_file.write(summary_text + '\n')

    # 在所有iteration中，输出：
    # 1）全局best_epoch上，各类别的ACC（一个list）
    # 2）每个类别在所有iteration中的全局最佳ACC（一个list）
    if getattr(args, 'enable_error_logging', False) and len(all_iteration_errors) > 0:
        try:
            if 'best_iteration_index' in locals():
                best_iter_idx = best_iteration_index
            else:
                # 如果未显式记录best_iteration_index，则默认选择第一个有效iteration
                valid_indices = [idx for idx, a in enumerate(aucMax_list) if not np.isnan(a)]
                best_iter_idx = valid_indices[0] if valid_indices else None

            if best_iter_idx is not None:
                best_iter_info = all_iteration_errors[best_iter_idx]
                best_epoch_global = best_iter_info['best_epoch']

                best_per_class_acc_list = (
                    per_class_acc_per_iteration[best_iter_idx]
                    if 'per_class_acc_per_iteration' in locals() else None
                )

                if best_per_class_acc_list is not None:
                    # ① 全局best_epoch上，各类别ACC列表
                    print(f"\nPer-class ACC at global best_epoch (iteration {best_iter_idx + 1}, epoch {best_epoch_global}):")
                    print(list(best_per_class_acc_list))

                # ② 每个类别在所有iteration中的全局最佳ACC
                if 'per_class_acc_per_iteration' in locals() and len(per_class_acc_per_iteration) > 0:
                    # 找到第一个非空的按类别ACC列表，确定类别数
                    first_valid = None
                    for per_cls in per_class_acc_per_iteration:
                        if per_cls is not None:
                            first_valid = per_cls
                            break
                    if first_valid is not None:
                        num_classes = len(first_valid)
                        global_best_per_class = [float('-inf')] * num_classes

                        for per_cls in per_class_acc_per_iteration:
                            if per_cls is None:
                                continue
                            for c, v in enumerate(per_cls):
                                if v is None or np.isnan(v):
                                    continue
                                if v > global_best_per_class[c]:
                                    global_best_per_class[c] = v

                        # 将仍为 -inf 的位置置为 NaN，表示该类别没有有效记录
                        global_best_per_class = [
                            (val if val != float('-inf') else float('nan'))
                            for val in global_best_per_class
                        ]

                        print("\nGlobal best ACC for each class across all iterations:")
                        print(global_best_per_class)
        except Exception as e:
            print(f"⚠️ Failed to compute per-class ACC and best metric lists: {e}")

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

        with open(log_filepath, 'a', encoding='utf-8') as f:
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
            f.write(f"嵌入维度 (latdim): {args.latdim}\n")
            f.write(f"GNN 层数 (gnn_layer): {args.gnn_layer}\n")
            f.write(f"全局负样本数量 (num_neg): {args.num_neg}\n")
            f.write(f"二跳邻居数量 (num_two_hop): {args.num_two_hop}\n")
            f.write(
                f"一跳最大比例 (one_hop_max_ratio): {args.one_hop_max_ratio:.1%} ({int(args.num_two_hop * args.one_hop_max_ratio)} max samples)\n")
            f.write(f"一跳权重倍数 (one_hop_weight): {args.one_hop_weight}\n")
            f.write(f"二跳权重倍数 (two_hop_weight): {args.two_hop_weight}\n")
            f.write(f"普通负样本权重 (common_neg_weight): {args.common_neg_weight}\n")
            text_mode = "启用LLM文本嵌入" if args.use_llm_embeddings else "仅使用结构嵌入"
            f.write(f"文本嵌入模式: {text_mode}\n")
            f.write("\n")

            f.write("=" * 60 + "\n")
            f.write("评估指标结果\n")
            f.write("=" * 60 + "\n")
            f.write("\n")

            for metric, values in overall_iteration_best.items():
                epochs = best_epochs_per_metric[metric]
                if not values or all(np.isnan(v) for v in values):
                    f.write(f"\n{metric}:\n  数据不足或全为NAN\n")
                    continue

                valid_pairs = [(v, e) for v, e in zip(values, epochs) if not np.isnan(v)]

                if not valid_pairs:
                    f.write(f"\n{metric}:\n  无有效数据\n")
                    continue

                valid_values = [p[0] for p in valid_pairs]
                mean_best = np.mean(valid_values)
                max_best_val, max_best_epoch = max(valid_pairs, key=lambda item: item[0])

                f.write(f"\n{metric}:\n")
                f.write(f"  平均最佳 = {mean_best:.4f}\n")
                f.write(f"  最高最佳 = {max_best_val:.4f} (在 epoch {max_best_epoch})\n")
                # 格式化列表输出，将值和epoch合并
                formatted_values = [f'{v:.4f} (epoch {e})' for v, e in zip(values, epochs)]
                f.write(f"  所有迭代最佳值 = [{', '.join(formatted_values)}]\n")

            print(f"✅ 实验结果已保存到: {log_filepath}")
            return log_filepath


    save_results_to_log(overall_iteration_best, coach)

    # 输出错误案例到日志和CSV文件（可通过超参数控制）
    if getattr(args, 'enable_error_logging', False) and len(all_iteration_errors) > 0:
        log_all_error_cases(all_iteration_errors, log_file, log_filepath)

    log_file.close()
    print(f"📝 日志文件已关闭: {log_filepath}")

    # results_parent_path = "D:\\桌面\\研\\论文\\实验代码\\DGCL-main\\DGCL-main\\results"
    # # results_parent_path = os.path.join(wandb.run.dir, 'results')
    # if not os.path.exists(results_parent_path):
    #     os.mkdir(results_parent_path)
    # np.savetxt('{}/{}_result.txt'.format(results_parent_path, args.data), np.array(results), delimiter=",", fmt='%f')

    print('result saved!!!')
    # wandb.finish()
