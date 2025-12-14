import json
import csv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import os

model_cache_dir = r"F:\my_models\Llama3-OpenBioLLM-8B"
train_csv_path = r"d:\桌面\研\论文\实验代码\DGCL-main\DGCL-main\Data\DGIdb\transductive\train.csv"
gene_mapping_path = r"d:\桌面\研\论文\实验代码\DGCL-main\DGCL-main\Data\DGIdb\DGIdb_gene_embeddings_text.json"
output_path = r"d:\桌面\研\论文\实验代码\DGCL-main\DGCL-main\Data\DGIdb\DGIdb_gene_emd.npy"

print("=" * 80)
print("开始生成基因嵌入")
print("=" * 80)

print("\n[1/4] 从train.csv读取基因ID...")
gene_ids = []
with open(train_csv_path, 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        gene_id = row[1]
        if gene_id not in gene_ids:
            gene_ids.append(gene_id)

print(f"✓ 读取到 {len(gene_ids)} 个唯一基因ID")
print(f"  基因ID示例: {gene_ids[:5]}")

print("\n[2/4] 从JSON映射表读取基因描述...")
with open(gene_mapping_path, 'r', encoding='utf-8') as f:
    gene_descriptions = json.load(f)

print(f"✓ 映射表中包含 {len(gene_descriptions)} 个基因")

gene_texts = {}
missing_genes = []
for gene_id in gene_ids:
    if gene_id in gene_descriptions:
        gene_texts[gene_id] = gene_descriptions[gene_id]
    else:
        missing_genes.append(gene_id)
        gene_texts[gene_id] = f"Gene ID: {gene_id}"

if missing_genes:
    print(f"⚠ 警告: {len(missing_genes)} 个基因在映射表中未找到，使用默认描述")
    print(f"  缺失基因示例: {missing_genes[:5]}")

print("\n[3/4] 加载Llama-3-OpenBioLLM模型...")
print(f"  从本地路径加载: {model_cache_dir}")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_cache_dir,
        local_files_only=True,
        trust_remote_code=True
    )
    print("✓ Tokenizer加载完成")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_cache_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto"
    )
    print("✓ 模型加载完成")
    print(f"  隐藏层维度: {model.config.hidden_size}")
    
    model.eval()
except Exception as e:
    print(f"✗ 模型加载失败: {e}")
    print(f"  请确保模型已下载到: {model_cache_dir}")
    raise

print("\n[4/4] 生成基因文本嵌入...")
embeddings = []
embedding_dim = model.config.hidden_size

print(f"  嵌入维度: {embedding_dim}")
print(f"  处理基因数: {len(gene_ids)}")

with torch.no_grad():
    for i, gene_id in enumerate(tqdm(gene_ids, desc="生成嵌入")):
        text = gene_texts[gene_id]
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        outputs = model(
            **inputs,
            output_hidden_states=True
        )
        
        last_hidden_state = outputs.hidden_states[-1]
        sentence_embedding = last_hidden_state[:, 0, :].cpu().numpy()
        
        embeddings.append(sentence_embedding[0])

embeddings_array = np.array(embeddings, dtype=np.float32)

print(f"\n✓ 生成完成")
print(f"  嵌入形状: {embeddings_array.shape}")

print(f"\n保存嵌入到: {output_path}")
np.save(output_path, embeddings_array)
print("✓ 嵌入已保存")

mapping_output_path = output_path.replace('.npy', '_gene_ids.json')
gene_id_mapping = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
with open(mapping_output_path, 'w', encoding='utf-8') as f:
    json.dump(gene_id_mapping, f, indent=2)
print(f"✓ 基因ID映射已保存到: {mapping_output_path}")

print("\n" + "=" * 80)
print("✓ 所有步骤完成！")
print("=" * 80)
