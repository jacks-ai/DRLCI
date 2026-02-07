"""
分析 BioLinkBERT 联合编码的 token 长度分布
格式: [CLS] Drug: {drug_desc} [SEP] Gene: {gene_desc} [SEP]
"""
import json
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from pathlib import Path
import sys
from tqdm import tqdm

# 添加项目路径
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.append(str(CODE_DIR))

from Params import args
from DataHandler import DataHandler

# ==================== 路径配置 ====================
DATA_ROOT = PROJECT_ROOT / 'Data' / args.data
MODEL_CACHE = Path(r"/mnt/data/huangpeng/DGCL/mymodel/BioLinkBERT")
DRUG_DESC_CSV = DATA_ROOT / 'drug_text' / 'mixed_drug_descriptions.csv'
GENE_DESC_JSON = DATA_ROOT / 'gene_text' / 'gene_embeddings_txt.json'
GLOBAL_IDS_JSON = DATA_ROOT / 'global_ids.json'


def load_global_ids():
    """加载全局ID映射"""
    with open(GLOBAL_IDS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['drug_ids'], data['gene_ids']


def load_drug_descriptions():
    """加载药物描述"""
    df = pd.read_csv(DRUG_DESC_CSV, encoding='utf-8')
    return dict(zip(df['DrugID'].astype(str), df['LLM_Text']))


def load_gene_descriptions():
    """加载基因描述"""
    with open(GENE_DESC_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_training_data():
    """使用DataHandler加载训练数据"""
    print(f"\n[加载训练数据] 数据集: {args.data}")
    
    handler = DataHandler()
    handler.LoadData()
    
    train_drugs = []
    train_genes = []
    train_labels = []
    
    for batch in handler.trnLoader:
        if len(batch) == 4:
            drugs, genes, labels, _ = batch
        else:
            drugs, genes, labels = batch[:3]
        
        train_drugs.extend(drugs.numpy())
        train_genes.extend(genes.numpy())
        train_labels.extend(labels.numpy())
    
    return np.array(train_drugs), np.array(train_genes), np.array(train_labels)


def analyze_individual_lengths(tokenizer, drug_ids, gene_ids, drug_descriptions, 
                               gene_descriptions, drug_ids_global, gene_ids_global):
    """
    分析 drug 和 gene 描述的原始长度分布（不截断）
    """
    print(f"\n{'='*80}")
    print(f"📊 Drug 和 Gene 原始描述长度分析（不截断）")
    print(f"{'='*80}")
    
    drug_lengths = []
    gene_lengths = []
    
    for drug_idx, gene_idx in tqdm(zip(drug_ids, gene_ids), total=len(drug_ids), desc="分析原始长度"):
        # 获取全局ID
        drug_id = drug_ids_global[drug_idx]
        gene_id = gene_ids_global[gene_idx]
        
        # 获取完整描述文本（不截断）
        drug_desc = drug_descriptions.get(drug_id, f"Drug {drug_id}")
        gene_desc = gene_descriptions.get(gene_id, f"Gene {gene_id}")
        
        # Tokenize 完整描述
        drug_tokens = tokenizer(drug_desc, add_special_tokens=False)
        gene_tokens = tokenizer(gene_desc, add_special_tokens=False)
        
        drug_lengths.append(len(drug_tokens['input_ids']))
        gene_lengths.append(len(gene_tokens['input_ids']))
    
    drug_lengths = np.array(drug_lengths)
    gene_lengths = np.array(gene_lengths)
    
    # Drug 描述统计
    print(f"\n{'─'*80}")
    print(f"Drug 描述 Token 长度统计（完整文本，不截断）:")
    print(f"{'─'*80}")
    print(f"  最小值: {drug_lengths.min()}")
    print(f"  最大值: {drug_lengths.max()}")
    print(f"  平均值: {drug_lengths.mean():.2f}")
    print(f"  中位数: {np.median(drug_lengths):.2f}")
    print(f"  标准差: {drug_lengths.std():.2f}")
    
    print(f"\n分位点分布:")
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
    for p in percentiles:
        value = np.percentile(drug_lengths, p)
        print(f"  {p:5.1f}%: {value:.0f}")
    
    # Gene 描述统计
    print(f"\n{'─'*80}")
    print(f"Gene 描述 Token 长度统计（完整文本，不截断）:")
    print(f"{'─'*80}")
    print(f"  最小值: {gene_lengths.min()}")
    print(f"  最大值: {gene_lengths.max()}")
    print(f"  平均值: {gene_lengths.mean():.2f}")
    print(f"  中位数: {np.median(gene_lengths):.2f}")
    print(f"  标准差: {gene_lengths.std():.2f}")
    
    print(f"\n分位点分布:")
    for p in percentiles:
        value = np.percentile(gene_lengths, p)
        print(f"  {p:5.1f}%: {value:.0f}")
    
    return drug_lengths, gene_lengths


def analyze_joint_token_lengths(tokenizer, drug_ids, gene_ids, drug_descriptions, 
                                 gene_descriptions, drug_ids_global, gene_ids_global):
    """
    分析联合编码的 token 长度分布（与训练脚本完全一致）
    使用 text + text_pair 方式，格式: [CLS] Drug: {drug_desc} [SEP] Gene: {gene_desc} [SEP]
    """
    print(f"\n{'='*80}")
    print(f"📊 联合编码 Token 长度分析（训练时实际使用，text_pair方式）")
    print(f"{'='*80}")
    print(f"格式: tokenizer(text='Drug: {{drug_desc}}', text_pair='Gene: {{gene_desc}}')")
    print(f"实际: [CLS] Drug: {{drug_desc}} [SEP] Gene: {{gene_desc}} [SEP]")
    
    token_lengths = []
    
    for drug_idx, gene_idx in tqdm(zip(drug_ids, gene_ids), total=len(drug_ids), desc="分析联合编码"):
        # 获取全局ID
        drug_id = drug_ids_global[drug_idx]
        gene_id = gene_ids_global[gene_idx]
        
        # 获取描述文本（不截断，让tokenizer的truncation参数处理）
        drug_desc = drug_descriptions.get(drug_id, f"Drug {drug_id}")
        gene_desc = gene_descriptions.get(gene_id, f"Gene {gene_id}")
        
        # 使用 text_pair 方式（与训练脚本完全一致）
        tokens = tokenizer(
            text=f"Drug: {drug_desc}",
            text_pair=f"Gene: {gene_desc}",
            add_special_tokens=True,  # 自动添加 [CLS] 和 [SEP]
            truncation=False  # 不截断，计算完整长度
        )
        token_lengths.append(len(tokens['input_ids']))
    
    token_lengths = np.array(token_lengths)
    
    # 计算统计信息
    print(f"\n样本数: {len(token_lengths)}")
    
    print(f"\n{'─'*80}")
    print(f"Token 长度统计:")
    print(f"{'─'*80}")
    print(f"  最小值: {token_lengths.min()}")
    print(f"  最大值: {token_lengths.max()}")
    print(f"  平均值: {token_lengths.mean():.2f}")
    print(f"  中位数: {np.median(token_lengths):.2f}")
    print(f"  标准差: {token_lengths.std():.2f}")
    
    print(f"\n分位点分布:")
    percentiles = [10, 25, 50, 75, 90, 95, 99, 99.5, 99.9]
    for p in percentiles:
        value = np.percentile(token_lengths, p)
        print(f"  {p:5.1f}%: {value:.0f}")
    
    # 分析超过不同长度阈值的比例
    print(f"\n超过长度阈值的样本比例:")
    thresholds = [128, 256, 384, 512, 768, 1024]
    for threshold in thresholds:
        count = (token_lengths > threshold).sum()
        ratio = count / len(token_lengths)
        print(f"  > {threshold:4d}: {ratio*100:6.2f}% ({count:6d} 样本)")
    
    return token_lengths


def main():
    print("=" * 80)
    print("BioLinkBERT 联合编码 Token 长度分析")
    print("=" * 80)
    print(f"数据集: {args.data}")
    
    # 1. 加载 tokenizer
    print("\n[1/5] 加载 BioLinkBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)
    print(f"  ✓ Tokenizer 词汇表大小: {len(tokenizer)}")
    
    # 2. 加载全局数据
    print("\n[2/5] 加载全局数据...")
    drug_ids_global, gene_ids_global = load_global_ids()
    print(f"  ✓ 药物数: {len(drug_ids_global)}, 基因数: {len(gene_ids_global)}")
    
    # 3. 加载描述文本
    print("\n[3/5] 加载描述文本...")
    drug_descriptions = load_drug_descriptions()
    gene_descriptions = load_gene_descriptions()
    print(f"  ✓ 药物描述数: {len(drug_descriptions)}")
    print(f"  ✓ 基因描述数: {len(gene_descriptions)}")
    
    # 4. 加载训练数据
    print("\n[4/6] 加载训练数据...")
    train_drugs, train_genes, train_labels = load_training_data()
    print(f"  ✓ 训练样本数: {len(train_labels)}")
    
    # 5. 分析原始描述长度（不截断）
    print("\n[5/6] 分析原始描述长度分布（不截断）...")
    drug_lengths, gene_lengths = analyze_individual_lengths(
        tokenizer, train_drugs, train_genes,
        drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global
    )
    
    # 6. 分析联合编码的 token 长度（训练时实际使用）
    print("\n[6/6] 分析联合编码 token 长度分布（训练时实际使用）...")
    token_lengths = analyze_joint_token_lengths(
        tokenizer, train_drugs, train_genes,
        drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global
    )
    
    # 综合分析与建议
    print(f"\n{'='*80}")
    print(f"📈 综合分析与建议")
    print(f"{'='*80}")
    
    # 原始描述长度分析
    print(f"\n原始描述长度对比:")
    print(f"  Drug 平均长度: {drug_lengths.mean():.0f} tokens")
    print(f"  Gene 平均长度: {gene_lengths.mean():.0f} tokens")
    print(f"  Drug 95% 分位点: {np.percentile(drug_lengths, 95):.0f} tokens")
    print(f"  Gene 95% 分位点: {np.percentile(gene_lengths, 95):.0f} tokens")
    
    # tokenizer 的 truncation 策略说明
    print(f"\n{'─'*80}")
    print(f"Tokenizer truncation 策略:")
    print(f"{'─'*80}")
    print(f"使用 text_pair 时，tokenizer 会智能截断：")
    print(f"  1. 优先保留较短的序列")
    print(f"  2. 从较长的序列中截断")
    print(f"  3. 确保总长度 <= max_length")
    print(f"  4. 特殊token ([CLS], [SEP]) 会自动添加并计入长度")
    
    # max_length 推荐
    print(f"\n{'─'*80}")
    print(f"max_length 设置建议:")
    print(f"{'─'*80}")
    
    # 推荐的 max_length（覆盖95%样本）
    p95 = np.percentile(token_lengths, 95)
    p99 = np.percentile(token_lengths, 99)
    
    print(f"\n联合编码分位点:")
    print(f"  95% 分位点: {p95:.0f} (覆盖 95% 样本)")
    print(f"  99% 分位点: {p99:.0f} (覆盖 99% 样本)")
    print(f"  推荐值 (95%, 向上取整到64倍数): {int(np.ceil(p95 / 64) * 64)}")
    print(f"  推荐值 (99%, 向上取整到64倍数): {int(np.ceil(p99 / 64) * 64)}")
    
    # 计算当前设置的覆盖率
    current_max_length = 512  # 训练脚本中的默认值
    coverage = (token_lengths <= current_max_length).sum() / len(token_lengths)
    truncated_count = (token_lengths > current_max_length).sum()
    
    print(f"\n当前训练脚本设置: max_length={current_max_length}")
    print(f"  覆盖率: {coverage*100:.2f}%")
    print(f"  被截断样本数: {truncated_count} ({(1-coverage)*100:.2f}%)")
    
    # 不同 max_length 的对比
    print(f"\n不同 max_length 设置对比:")
    test_lengths = [192, 256, 384, 512]
    for test_len in test_lengths:
        cov = (token_lengths <= test_len).sum() / len(token_lengths)
        trunc = (token_lengths > test_len).sum()
        print(f"  max_length={test_len}: 覆盖率 {cov*100:.2f}%, 截断 {trunc} 样本")
    
    if coverage < 0.95:
        print(f"\n⚠️  警告: 当前设置会截断超过 5% 的样本，建议增大 max_length 到 {int(np.ceil(p95 / 64) * 64)}")
    elif coverage < 0.99:
        print(f"\n✓ 当前设置覆盖了 95% 以上样本，但仍有 {(1-coverage)*100:.2f}% 被截断")
        print(f"  如需更高覆盖率，建议增大到 {int(np.ceil(p99 / 64) * 64)}")
    else:
        print(f"\n✓ 当前设置合理，覆盖了 99% 以上的样本")
        print(f"  💡 可以考虑降低到 256 以提升训练速度（覆盖率: {(token_lengths <= 256).sum() / len(token_lengths)*100:.2f}%）")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
