"""
优化版本的多进程辅助模块 - 使用预缓存的药物-基因字典和基因邻接关系
解决O(N)遍历瓶颈，实现真正的高性能多进程计算
"""
import numpy as np
import os
import pickle

# 全局变量，在进程初始化时设置一次
_gene_neighbors = None
_drug_gene_dict = None  # 使用预缓存的药物-基因字典


def init_worker_process(data_name):
    """
    初始化工作进程的全局数据
    从缓存加载药物-基因字典和基因邻接关系
    
    参数:
        data_name: 数据集名称，用于定位缓存文件
    """
    global _gene_neighbors, _drug_gene_dict
    
    # 加载预缓存的数据
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'cache')
    cache_filename = f"dict_{data_name}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if not os.path.exists(cache_path):
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ 缓存文件不存在: {cache_path}\n"
            f"{'='*80}\n"
            f"请先运行预处理脚本:\n"
            f"  python preprocess_drug_gene_dict.py\n"
            f"{'='*80}\n"
        )
        raise FileNotFoundError(error_msg)
    
    # 加载缓存
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
    except Exception as e:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ 缓存文件加载失败: {cache_path}\n"
            f"错误信息: {str(e)}\n"
            f"{'='*80}\n"
            f"请重新运行预处理脚本:\n"
            f"  python preprocess_drug_gene_dict.py\n"
            f"{'='*80}\n"
        )
        raise RuntimeError(error_msg) from e
    
    # 验证缓存格式和内容
    if not isinstance(cache_data, dict):
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ 缓存格式错误: 期望字典类型\n"
            f"实际类型: {type(cache_data)}\n"
            f"{'='*80}\n"
            f"请重新运行预处理脚本:\n"
            f"  python preprocess_drug_gene_dict.py\n"
            f"{'='*80}\n"
        )
        raise ValueError(error_msg)
    
    # 检查并加载 drug_gene_dict
    if 'drug_gene_dict' not in cache_data:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ 缓存中缺少 'drug_gene_dict' 数据\n"
            f"缓存包含的键: {list(cache_data.keys())}\n"
            f"{'='*80}\n"
            f"请重新运行预处理脚本:\n"
            f"  python preprocess_drug_gene_dict.py\n"
            f"{'='*80}\n"
        )
        raise KeyError(error_msg)
    
    _drug_gene_dict = cache_data['drug_gene_dict']
    
    # 检查并加载 gene_neighbors
    if 'gene_neighbors' not in cache_data:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ 缓存中缺少 'gene_neighbors' 数据\n"
            f"缓存包含的键: {list(cache_data.keys())}\n"
            f"{'='*80}\n"
            f"请重新运行预处理脚本:\n"
            f"  python preprocess_drug_gene_dict.py\n"
            f"{'='*80}\n"
        )
        raise KeyError(error_msg)
    
    _gene_neighbors = cache_data['gene_neighbors']
    
    # 验证数据不为空
    if _drug_gene_dict is None or len(_drug_gene_dict) == 0:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ drug_gene_dict 数据为空或无效\n"
            f"{'='*80}\n"
            f"请检查数据集并重新运行预处理脚本:\n"
            f"  python preprocess_drug_gene_dict.py\n"
            f"{'='*80}\n"
        )
        raise ValueError(error_msg)
    
    if _gene_neighbors is None or len(_gene_neighbors) == 0:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ gene_neighbors 数据为空或无效\n"
            f"{'='*80}\n"
            f"请检查数据集并重新运行预处理脚本:\n"
            f"  python preprocess_drug_gene_dict.py\n"
            f"{'='*80}\n"
        )
        raise ValueError(error_msg)
    
    # 数据加载成功（不输出信息，避免24个进程同时输出造成I/O竞争）


def compute_single_pair_optimized(task_args):
    """
    极致优化版本的单对计算函数
    使用预缓存的药物-基因字典实现O(1)查询
    """
    drug_idx, gene_idx, num_neighbors, one_hop_weight, two_hop_weight, one_hop_max_ratio, gene_count, random_seed = task_args
    
    # 设置随机种子
    np.random.seed(random_seed)
    
    # 使用全局数据（已经在进程初始化时设置）
    global _gene_neighbors, _drug_gene_dict
    
    # 检查全局数据是否已初始化
    if _drug_gene_dict is None or _gene_neighbors is None:
        error_msg = (
            f"\n{'='*80}\n"
            f"❌ 工作进程未正确初始化！\n"
            f"drug_gene_dict: {'已加载' if _drug_gene_dict is not None else '未加载'}\n"
            f"gene_neighbors: {'已加载' if _gene_neighbors is not None else '未加载'}\n"
            f"{'='*80}\n"
        )
        raise RuntimeError(error_msg)
    
    # 🚀 关键优化：O(1)查询替代O(N)遍历
    drug_connected_genes = _drug_gene_dict.get(drug_idx, set())
    
    # 获取一跳邻居并直接过滤
    one_hop = _gene_neighbors.get(gene_idx, set())
    filtered_one_hop = [g for g in one_hop if g not in drug_connected_genes]
    
    # 优化二跳邻居计算
    two_hop = set()
    for one_hop_gene in one_hop:
        second_hop = _gene_neighbors.get(one_hop_gene, set())
        valid_two_hop = second_hop - {gene_idx} - one_hop - drug_connected_genes
        two_hop.update(valid_two_hop)
    
    filtered_two_hop = list(two_hop)
    
    # 计算采样数量
    max_one_hop = int(num_neighbors * one_hop_max_ratio)
    actual_one_hop = min(len(filtered_one_hop), max_one_hop)
    actual_two_hop = num_neighbors - actual_one_hop
    
    selected_negatives = []
    weights = []
    
    # 采样一跳邻居
    if actual_one_hop > 0 and len(filtered_one_hop) > 0:
        if len(filtered_one_hop) >= actual_one_hop:
            one_hop_samples = np.random.choice(filtered_one_hop, size=actual_one_hop, replace=False)
        else:
            one_hop_samples = np.random.choice(filtered_one_hop, size=actual_one_hop, replace=True)
        
        selected_negatives.extend(one_hop_samples.tolist())
        weights.extend([one_hop_weight] * actual_one_hop)
        situation_type = 'mixed_with_one_hop'
    else:
        situation_type = 'no_one_hop_available'
    
    # 采样二跳邻居
    if actual_two_hop > 0:
        if len(filtered_two_hop) >= actual_two_hop:
            two_hop_samples = np.random.choice(filtered_two_hop, size=actual_two_hop, replace=False)
        elif len(filtered_two_hop) > 0:
            two_hop_samples = np.random.choice(filtered_two_hop, size=actual_two_hop, replace=True)
            situation_type = 'mixed_with_repeated_two_hop'
        else:
            # 使用随机无交互基因
            all_genes = set(range(gene_count))
            no_interaction_genes = list(all_genes - drug_connected_genes)
            
            if len(no_interaction_genes) >= actual_two_hop:
                two_hop_samples = np.random.choice(no_interaction_genes, size=actual_two_hop, replace=False)
                situation_type = 'mixed_with_random'
            else:
                two_hop_samples = np.random.choice(no_interaction_genes, size=actual_two_hop, replace=True)
                situation_type = 'mixed_few_random'
        
        selected_negatives.extend(two_hop_samples.tolist())
        weights.extend([two_hop_weight] * actual_two_hop)
    
    return (drug_idx, gene_idx), {
        'negatives': selected_negatives,
        'weights': weights,
        'situation_type': situation_type
    }, os.getpid()


def compute_batch_pairs_optimized(batch_tasks):
    """
    批量处理多个对，进一步减少进程间通信
    """
    results = {}
    for task in batch_tasks:
        key, result, _ = compute_single_pair_optimized(task)
        results[key] = result
    return results
