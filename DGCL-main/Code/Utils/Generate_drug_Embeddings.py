import json
import csv
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import argparse
import os

def main(args):
    print("=" * 80)
    print("开始生成药物文本嵌入")
    print("=" * 80)

    print("\n[1/5] 从 train.csv 读取药物ID...")
    drug_ids = []
    with open(args.train_csv_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            drug_id = row[0]  # 药物ID在第一列
            if drug_id not in drug_ids:
                drug_ids.append(drug_id)

    print(f"✓ 读取到 {len(drug_ids)} 个唯一药物ID")
    print(f"  药物ID示例: {drug_ids[:5]}")

    print("\n[2/5] 从JSON文件读取药物描述...")
    with open(args.drug_mapping_path, 'r', encoding='utf-8') as f:
        drug_descriptions = json.load(f)

    print(f"✓ 映射表中包含 {len(drug_descriptions)} 个药物")

    drug_texts = {}
    missing_drugs = []
    for drug_id in drug_ids:
        if drug_id in drug_descriptions:
            drug_texts[drug_id] = drug_descriptions[drug_id]
        else:
            missing_drugs.append(drug_id)
            drug_texts[drug_id] = f"Drug ID: {drug_id}"

    if missing_drugs:
        print(f"⚠ 警告: {len(missing_drugs)} 个药物在映射表中未找到，将使用默认描述")
        print(f"  缺失药物示例: {missing_drugs[:10]}")
    else:
        print("✓ 所有药物都在映射表中找到")

    print("\n[3/5] 加载Llama-3-OpenBioLLM模型...")
    print(f"  从本地路径加载: {args.model_cache_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_cache_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        print("✓ Tokenizer加载完成")
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_cache_dir,
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
        print(f"  请确保模型已下载到: {args.model_cache_dir}")
        raise

    print("\n[4/5] 生成药物文本嵌入...")
    embedding_dim = model.config.hidden_size
    print(f"  嵌入维度: {embedding_dim}")
    print(f"  处理药物数: {len(drug_ids)}")

    embeddings = {}
    batch_size = args.batch_size
    print(f"  批处理大小: {batch_size}")

    all_texts = [drug_texts[did] for did in drug_ids]

    with torch.no_grad():
        for i in tqdm(range(0, len(all_texts), batch_size), desc="生成嵌入"):
            batch_texts = all_texts[i:i+batch_size]
            batch_drug_ids = drug_ids[i:i+batch_size]
            
            inputs = tokenizer(
                batch_texts,
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
            
            sequence_lengths = (inputs['attention_mask'].sum(dim=1) - 1).to('cpu')
            # 提取最后一个有效token的嵌入
            batch_embeddings = last_hidden_state[torch.arange(last_hidden_state.shape[0], device='cpu'), sequence_lengths, :].cpu().float().numpy()

            for drug_id, embedding in zip(batch_drug_ids, batch_embeddings):
                embeddings[drug_id] = embedding

    ordered_embeddings = np.array([embeddings[did] for did in drug_ids], dtype=np.float32)

    print(f"\n✓ 生成完成")
    print(f"  嵌入形状: {ordered_embeddings.shape}")

    print(f"\n[5/5] 保存嵌入和ID映射...")
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  已创建输出目录: {output_dir}")

    np.save(args.output_path, ordered_embeddings)
    print(f"✓ 嵌入已保存到: {args.output_path}")

    mapping_output_path = args.output_path.replace('.npy', '_drug_ids.json')
    drug_id_mapping = {drug_id: idx for idx, drug_id in enumerate(drug_ids)}
    with open(mapping_output_path, 'w', encoding='utf-8') as f:
        json.dump(drug_id_mapping, f, indent=2)
    print(f"✓ 药物ID映射已保存到: {mapping_output_path}")

    print("\n" + "=" * 80)
    print("✓ 所有步骤完成！")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用Llama-3-OpenBioLLM模型生成药物文本嵌入")
    
    parser.add_argument('--model_cache_dir', type=str, 
                        default=r"/mnt/data/huangpeng/DGCL/DGCL-main/hf_cache/Llama3-OpenBioLLM-8B",
                        help='预训练模型的本地缓存目录')
                        
    parser.add_argument('--train_csv_path', type=str, 
                        default='/mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/transductive/test.csv',
                        help='包含药物ID的训练数据CSV文件路径')

    parser.add_argument('--drug_mapping_path', type=str, 
                        default=r"/mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/drug_text_test/drug_text_test.json",
                        help='药物ID到文本描述的JSON映射文件路径')

    parser.add_argument('--output_path', type=str, 
                        default=r"/mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/drug_text_test/drugbank_emd_test.npy",
                        help='输出的药物嵌入.npy文件路径')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='批处理大小')

    args = parser.parse_args()
    main(args)
