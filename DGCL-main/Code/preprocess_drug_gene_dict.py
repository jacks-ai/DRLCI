#!/usr/bin/env python3
"""
预处理药物-基因字典脚本
将O(N)遍历优化为O(1)查询，大幅提升多进程性能
同时缓存基因邻接关系（gene_neighbors）
"""

import os
import pickle
import time
from DataHandler import DataHandler
from Params import args

def preprocess_drug_gene_dict():
    """
    预处理并缓存药物-基因字典和基因邻接关系
    """
    print("🚀 开始预处理药物-基因字典和基因邻接关系...")
    start_time = time.time()
    
    # 初始化数据处理器
    print("📊 加载数据集...")
    handler = DataHandler()
    
    # 必须先调用LoadData()才能访问trnLoader
    handler.LoadData()
    
    # 获取药物-基因交互矩阵
    train_dataset = handler.trnLoader.dataset
    drug_gene_matrix = train_dataset.dokmat
    
    # 从数据中获取药物和基因数量
    all_drugs = set()
    all_genes = set()
    for (drug_idx, gene_idx) in drug_gene_matrix.keys():
        all_drugs.add(drug_idx)
        all_genes.add(gene_idx)
    
    drug_count_total = len(all_drugs)
    gene_count_total = len(all_genes)
    
    print(f"📈 数据集信息:")
    print(f"  数据集: {args.data}")
    print(f"  药物数量: {drug_count_total}")
    print(f"  基因数量: {gene_count_total}")
    print(f"  交互关系数量: {len(drug_gene_matrix)}")
    
    # ========== 1. 构建药物-基因字典 ==========
    print("🔧 构建药物-基因字典...")
    drug_gene_dict = {}
    
    for (drug_idx, gene_idx) in drug_gene_matrix.keys():
        if drug_idx not in drug_gene_dict:
            drug_gene_dict[drug_idx] = set()
        drug_gene_dict[drug_idx].add(gene_idx)
    
    # 统计信息
    drug_count = len(drug_gene_dict)
    total_interactions = sum(len(genes) for genes in drug_gene_dict.values())
    avg_interactions = total_interactions / drug_count if drug_count > 0 else 0
    
    print(f"📊 药物-基因字典统计:")
    print(f"  有交互的药物数量: {drug_count}")
    print(f"  总交互关系数量: {total_interactions}")
    print(f"  平均每个药物的交互基因数: {avg_interactions:.1f}")
    
    # ========== 2. 构建基因邻接关系 (gene_neighbors) ==========
    print("🔧 构建基因-基因邻接关系...")
    gene_neighbors = {}
    
    # 初始化每个基因的邻居集合
    for gene in range(args.gene):
        gene_neighbors[gene] = set()
    
    # 基于药物构建基因间的邻接关系
    for drug, genes in drug_gene_dict.items():
        genes_list = list(genes)
        # 对于每对基因，如果它们与同一药物交互，则它们是邻居
        for i in range(len(genes_list)):
            for j in range(i + 1, len(genes_list)):
                gene1, gene2 = genes_list[i], genes_list[j]
                gene_neighbors[gene1].add(gene2)
                gene_neighbors[gene2].add(gene1)
    
    # 统计基因邻接关系
    total_gene_edges = sum(len(neighbors) for neighbors in gene_neighbors.values()) // 2
    avg_neighbors = sum(len(neighbors) for neighbors in gene_neighbors.values()) / len(gene_neighbors) if len(gene_neighbors) > 0 else 0
    
    print(f"📊 基因邻接关系统计:")
    print(f"  基因数量: {len(gene_neighbors)}")
    print(f"  基因间边数: {total_gene_edges}")
    print(f"  平均每个基因的邻居数: {avg_neighbors:.1f}")
    
    # ========== 3. 保存到缓存文件 ==========
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'cache')
    os.makedirs(cache_dir, exist_ok=True)
    
    cache_filename = f"dict_{args.data}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    print(f"💾 保存数据到缓存: {cache_path}")
    
    # 保存两个字典到一个文件
    cache_data = {
        'drug_gene_dict': drug_gene_dict,
        'gene_neighbors': gene_neighbors,
        'metadata': {
            'data': args.data,
            'drug_count': drug_count_total,
            'gene_count': gene_count_total,
            'interaction_count': total_interactions,
            'gene_edge_count': total_gene_edges,
            'timestamp': time.time()
        }
    }
    
    with open(cache_path, 'wb') as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # 验证缓存文件
    file_size = os.path.getsize(cache_path) / (1024 * 1024)  # MB
    print(f"✅ 缓存文件大小: {file_size:.2f} MB")
    
    elapsed_time = time.time() - start_time
    print(f"⚡ 预处理完成，耗时: {elapsed_time:.2f}s")
    
    # ========== 4. 测试加载速度 ==========
    print("🧪 测试缓存加载速度...")
    test_start = time.time()
    
    with open(cache_path, 'rb') as f:
        test_data = pickle.load(f)
    
    test_time = time.time() - test_start
    print(f"📈 缓存加载耗时: {test_time:.3f}s")
    
    # 验证数据正确性
    if (len(test_data['drug_gene_dict']) == len(drug_gene_dict) and 
        len(test_data['gene_neighbors']) == len(gene_neighbors)):
        print("✅ 缓存数据验证通过")
        print(f"  药物-基因字典: {len(test_data['drug_gene_dict'])} 个药物")
        print(f"  基因邻接关系: {len(test_data['gene_neighbors'])} 个基因")
    else:
        print("❌ 缓存数据验证失败")
        return False
    
    print(f"\n🎯 优化效果预估:")
    print(f"  原方案: 每个pair遍历{len(drug_gene_matrix)}条记录 (O(N))")
    print(f"  新方案: 每个pair只需O(1)查询")
    print(f"  预期性能提升: 100-1000倍")
    
    return cache_path

def load_drug_gene_dict(data_name=None):
    """
    加载预处理的药物-基因字典和基因邻接关系
    
    返回:
        tuple: (drug_gene_dict, gene_neighbors) 或 (None, None)
    """
    if data_name is None:
        data_name = args.data
    
    cache_dir = os.path.join(os.path.dirname(__file__), '..', 'Data', 'cache')
    cache_filename = f"dict_{data_name}.pkl"
    cache_path = os.path.join(cache_dir, cache_filename)
    
    if not os.path.exists(cache_path):
        print(f"❌ 缓存文件不存在: {cache_path}")
        print("请先运行预处理脚本: python preprocess_drug_gene_dict.py")
        return None, None
    
    print(f"📂 加载药物-基因字典和基因邻接关系缓存: {cache_path}")
    start_time = time.time()
    
    with open(cache_path, 'rb') as f:
        cache_data = pickle.load(f)
    
    load_time = time.time() - start_time
    
    # 检查缓存格式
    if isinstance(cache_data, dict) and 'drug_gene_dict' in cache_data and 'gene_neighbors' in cache_data:
        # 新格式：包含两个字典
        drug_gene_dict = cache_data['drug_gene_dict']
        gene_neighbors = cache_data['gene_neighbors']
        print(f"✅ 加载完成，耗时: {load_time:.3f}s")
        print(f"  药物-基因字典: {len(drug_gene_dict)} 个药物")
        print(f"  基因邻接关系: {len(gene_neighbors)} 个基因")
        
        # 显示元数据信息
        if 'metadata' in cache_data:
            metadata = cache_data['metadata']
            print(f"📊 缓存元数据:")
            print(f"  数据集: {metadata.get('data', 'N/A')}")
            print(f"  交互关系数: {metadata.get('interaction_count', 'N/A')}")
            print(f"  基因间边数: {metadata.get('gene_edge_count', 'N/A')}")
        
        return drug_gene_dict, gene_neighbors
    else:
        # 旧格式：只有药物-基因字典
        print(f"⚠️  检测到旧格式缓存文件（仅包含药物-基因字典）")
        print(f"   建议重新运行预处理脚本以获取基因邻接关系")
        drug_gene_dict = cache_data
        print(f"✅ 字典加载完成，耗时: {load_time:.3f}s，包含{len(drug_gene_dict)}个药物")
        return drug_gene_dict, None

if __name__ == "__main__":
    print("=" * 80)
    print("药物-基因字典和基因邻接关系预处理工具")
    print("=" * 80)
    print()
    
    # 检查参数
    print(f"当前数据集: {args.data}")
    print()
    
    # 执行预处理
    cache_path = preprocess_drug_gene_dict()
    
    if cache_path:
        print("\n" + "=" * 80)
        print("✨ 预处理完成！")
        print("=" * 80)
        print()
        print("📋 下一步:")
        print("1. 修改 multiprocess_helper_optimized.py 使用这个缓存")
        print("2. 两个数据结构都已缓存:")
        print("   - drug_gene_dict: 药物-基因交互字典")
        print("   - gene_neighbors: 基因邻接关系")
        print("3. 预期性能提升: 0.3 pairs/s → 10-30 pairs/s")
        print("4. 预期总时间: 58小时 → 0.6-2小时")
        print()
        print(f"💾 缓存文件位置: {cache_path}")
        print()
        print("📖 使用方法:")
        print("   from preprocess_drug_gene_dict import load_drug_gene_dict")
        print("   drug_gene_dict, gene_neighbors = load_drug_gene_dict()")
    else:
        print("\n❌ 预处理失败，请检查错误信息")
