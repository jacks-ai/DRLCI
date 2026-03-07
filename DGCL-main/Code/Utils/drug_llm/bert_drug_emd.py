import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import pandas as pd
import os
import hashlib
from types import SimpleNamespace
from pathlib import Path
import sys

# 添加父目录到路径以导入Params
CODE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE_DIR))
from Params import args as train_args

PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = PROJECT_ROOT / 'Data'
DEFAULT_MODEL_CACHE = Path(r"/mnt/data/huangpeng/DGCL/mymodel/BioLinkBERT")

# 根据数据集动态选择微调后最佳 BioLinkBERT 权重（只包含 bert_state_dict）
DATASET_NAME = getattr(train_args, 'data', 'DrugBank')
if DATASET_NAME == 'DrugBank':
    # DrugBank 数据集使用你指定的最新微调权重
    FINE_TUNED_CKPT = Path(
        "/mnt/data/huangpeng/DGCL/DGCL-main/Code/bert/best_biolinkbert_only_DrugBank_0303_020934.pt"
    )
else:
    # 其他数据集仍然回退到原来的默认权重路径
    FINE_TUNED_CKPT = Path(
        "/mnt/data/huangpeng/DGCL/DGCL-main/Code/bert/best_biolinkbert_only_0302_123530.pt"
    )


def main(args):
    print("=" * 80)
    print("开始生成药物文本嵌入 (BioLinkBERT)")
    print("=" * 80)
    print("当前路径配置：")
    print(f"  模型缓存路径: {args.model_cache_dir}")
    print(f"  全局ID文件:  {args.global_ids_path}")
    print(f"  文本映射文件: {args.drug_mapping_path}")
    print(f"  输出Numpy文件: {args.output_path}")
    print(f"  批处理大小: {args.batch_size}")

    print("\n[1/5] 从全局ID文件读取药物ID...")
    try:
        with open(args.global_ids_path, 'r', encoding='utf-8') as f:
            global_ids_data = json.load(f)
        drug_ids = global_ids_data['drug_ids']
        expected_checksum = global_ids_data.get('drug_ids_checksum')
        computed_checksum = hashlib.sha1('\n'.join(drug_ids).encode('utf-8')).hexdigest()
        print(f"  校验 global_ids drug checksum: expected={expected_checksum}, computed={computed_checksum}")
        if expected_checksum and expected_checksum != computed_checksum:
            raise ValueError("global_ids drug checksum mismatch，顺序可能已被破坏")
    except Exception as e:
        print(f"✗ 读取全局ID文件失败: {e}")
        raise

    print(f"✓ 读取到 {len(drug_ids)} 个唯一药物ID")
    print(f"  药物ID示例: {drug_ids[:5]}")

    print("\n[2/5] 从JSON文件读取药物描述...")
    drug_descriptions = load_drug_descriptions(args.drug_mapping_path)

    print(f"✓ 映射表中包含 {len(drug_descriptions)} 个药物")

    drug_texts = {}
    missing_drugs = []
    for drug_id in drug_ids:
        if drug_id in drug_descriptions:
            drug_texts[drug_id] = drug_descriptions[drug_id]
        else:
            missing_drugs.append(drug_id)
            # 如果在描述文件里找不到，就用一个默认的简单描述
            drug_texts[drug_id] = f"DrugBank ID: {drug_id}"

    if missing_drugs:
        print(f"⚠ 警告: {len(missing_drugs)} 个药物在映射表中未找到，将使用默认描述")
        print(f"  缺失药物示例: {missing_drugs[:10]}")
    else:
        print("✓ 所有药物都在映射表中找到")

    print("\n[3/5] 加载BioLinkBERT模型...")
    print(f"  从本地路径加载: {args.model_cache_dir}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_cache_dir,
            local_files_only=True,
            trust_remote_code=True
        )
        print("✓ Tokenizer加载完成")
        
        model = AutoModel.from_pretrained(
            args.model_cache_dir,
            local_files_only=True,
            trust_remote_code=True
        )

        # 如果存在微调后的BioLinkBERT权重，则加载
        if FINE_TUNED_CKPT.is_file():
            print(f"  检测到微调权重: {FINE_TUNED_CKPT}")
            try:
                ckpt = torch.load(FINE_TUNED_CKPT, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("bert_state_dict", ckpt)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                print(f"  ✓ 已加载微调权重 (missing={len(missing)}, unexpected={len(unexpected)})")
            except Exception as e:
                print(f"  ⚠ 加载微调权重失败，将继续使用原始预训练模型: {e}")
        else:
            print(f"  未找到微调权重文件，将使用原始预训练BioLinkBERT")
        
        # 将模型移到GPU
        device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        print(f"✓ 模型加载完成，设备: {device}")
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
            batch_texts = all_texts[i:i + batch_size]
            batch_drug_ids = drug_ids[i:i + batch_size]

            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            
            # 使用[CLS] token
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
            
            # Mean Pooling（排除Padding，只对有效token求平均）
            # attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [batch, seq_len, 1]
            # masked_embeddings = outputs.last_hidden_state * attention_mask  # 屏蔽padding
            # sum_embeddings = masked_embeddings.sum(dim=1)  # [batch, hidden_dim]
            # sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # 防止除零
            # mean_embeddings = (sum_embeddings / sum_mask).cpu().float().numpy()

            for drug_id, embedding in zip(batch_drug_ids, cls_embeddings):
                embeddings[drug_id] = embedding

    # 根据 drug_ids 的顺序重新排列嵌入向量
    ordered_embeddings = np.array([embeddings[did] for did in drug_ids], dtype=np.float32)

    print(f"\n✓ 生成完成")
    print(f"  嵌入形状: {ordered_embeddings.shape}")

    print(f"\n[5/5] 保存嵌入...")
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 保存前先删除旧文件（如果存在）
    if os.path.exists(args.output_path):
        os.remove(args.output_path)
        print(f"  已删除旧的嵌入文件")
    
    np.save(args.output_path, ordered_embeddings)
    print(f"✓ 嵌入已保存到: {args.output_path}")

    print("\n" + "=" * 80)
    print("✓ 所有步骤完成！")
    print("=" * 80)


def load_drug_descriptions(mapping_path):
    """
    支持 JSON 或 CSV 的药物描述映射加载
    """
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"找不到药物描述文件: {mapping_path}")

    if mapping_path.suffix.lower() == '.json':
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    if mapping_path.suffix.lower() == '.csv':
        df = pd.read_csv(mapping_path, encoding='utf-8')
        candidate_cols = [('DrugID', 'LLM_Text'), ('DrugID', 'LLM_Input_Text')]
        for id_col, text_col in candidate_cols:
            if id_col in df.columns and text_col in df.columns:
                return {
                    str(row[id_col]): row[text_col]
                    for _, row in df[[id_col, text_col]].dropna().iterrows()
                }
        raise ValueError(f"CSV 文件 {mapping_path} 中找不到支持的列组合 {candidate_cols}")

    raise ValueError(f"不支持的药物描述文件格式: {mapping_path.suffix}")


def build_default_args():
    dataset = getattr(train_args, 'data', 'DrugBank')
    dataset_dir = DATA_ROOT / dataset
    drug_text_dir = dataset_dir / 'drug_text'

    global_ids_path = dataset_dir / 'global_ids_2.json'

    mapping_candidates = [
        drug_text_dir / 'mixed_drug_descriptions.csv',
        drug_text_dir / 'drug_text.json',
    ]
    for candidate in mapping_candidates:
        if candidate.exists():
            drug_mapping_path = candidate
            break
    else:
        drug_mapping_path = mapping_candidates[0]

    # 添加bert前缀和cls后缀
    output_path = drug_text_dir / f"ft_bert_{dataset.lower()}_drug_emd_cls.npy"

    return SimpleNamespace(
        model_cache_dir=str(DEFAULT_MODEL_CACHE),
        global_ids_path=str(global_ids_path),
        drug_mapping_path=str(drug_mapping_path),
        output_path=str(output_path),
        batch_size=8,  # BioLinkBERT可以使用更大的batch size
    )


if __name__ == "__main__":
    args = build_default_args()
    main(args)
