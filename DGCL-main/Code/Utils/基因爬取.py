import mygene
import pandas as pd

# NCBI（National Center for Biotechnology Information，美国国家生物技术信息中心）

# 1. 实例化查询对象
mg = mygene.MyGeneInfo()

# 2. 从CSV文件读取基因ID
csv_path = '/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/transductive/train.csv'
df = pd.read_csv(csv_path, header=None)

# 提取第二列（索引为1）的所有唯一基因ID
gene_ids_list = list(set(df[1].astype(str).tolist()))

print(f"从CSV文件中提取了 {len(gene_ids_list)} 个唯一基因ID")
print(f"基因ID范围: {gene_ids_list[:5]} ... {gene_ids_list[-5:]}")

# 3. 批量查询
# scopes='entrezgene': 告诉库我们输入的是数字ID
# fields='symbol,name,summary': 我们需要 基因符号、全名 和 功能简介
print("\n正在从 NCBI 查询基因信息...")
results = mg.querymany(gene_ids_list, scopes='entrezgene', fields='symbol,name,summary', species='human')

# 4. 格式化输出为自然语言文本 (用于 LLM 输入)
gene_texts = {}

for res in results:
    query_id = res['query']

    # 获取字段，如果缺失则填默认值
    symbol = res.get('symbol', 'Unknown Gene')
    name = res.get('name', 'Unknown Name')
    summary = res.get('summary', 'No functional description available for this gene.')

    # 【关键步骤】构造 Prompt 友好的文本格式
    # 这种格式包含了名字和具体的生物学功能，Llama-3-OpenBioLLM 读起来会非常顺畅
    text_for_llm = f"Gene symbol: {symbol}. Gene name: {name}. Function: {summary}"

    gene_texts[query_id] = text_for_llm

# 5. 保存结果到文件
import json
output_path = '/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb_gene_embeddings_text.json'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(gene_texts, f, ensure_ascii=False, indent=2)

print(f"\n已保存 {len(gene_texts)} 个基因的文本描述到 {output_path}")

# 6. 打印结果示例
print("\n--- 转换结果示例 ---")
for i, (gid, text) in enumerate(list(gene_texts.items())[:5]):
    print(f"ID [{gid}]: {text[:150]}...")