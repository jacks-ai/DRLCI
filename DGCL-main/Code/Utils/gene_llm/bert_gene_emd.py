import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os
from pathlib import Path
from types import SimpleNamespace
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
    print("开始生成基因文本嵌入 (BioLinkBERT)")
    print("=" * 80)
    print("当前路径配置：")
    print(f"  模型缓存路径: {args.model_cache_dir}")
    print(f"  全局ID文件:  {args.global_ids_path}")
    print(f"  基因描述文件: {args.gene_mapping_path}")
    print(f"  输出Numpy文件: {args.output_path}")
    mapping_output_path = args.output_path.replace('.npy', '_gene_ids.json')
    print(f"  基因ID映射输出: {mapping_output_path}")
    print(f"  批处理大小: {args.batch_size}")

    print("\n[1/5] 从全局ID文件读取基因ID...")
    try:
        print(f"  读取全局ID文件: {args.global_ids_path}")
        with open(args.global_ids_path, 'r', encoding='utf-8') as f:
            global_ids_data = json.load(f)
        gene_ids = global_ids_data['gene_ids']
    except Exception as e:
        print(f"✗ 读取全局ID文件失败: {e}")
        raise

    print(f"✓ 读取到 {len(gene_ids)} 个唯一基因ID")
    print(f"  基因ID示例: {gene_ids[:5]}")

    print("\n[2/5] 从基因描述文件读取文本...")
    gene_descriptions = load_gene_descriptions(args.gene_mapping_path)

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
        print(f"⚠ 警告: {len(missing_genes)} 个基因在映射表中未找到，将使用默认描述")
        print(f"  缺失基因示例: {missing_genes[:10]}")
    else:
        print("✓ 所有基因都在映射表中找到")

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
                ckpt = torch.load(FINE_TUNED_CKPT, map_location="cpu",weights_only=False)
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

    print("\n[4/5] 生成基因文本嵌入...")
    embedding_dim = model.config.hidden_size
    print(f"  嵌入维度: {embedding_dim}")
    print(f"  处理基因数: {len(gene_ids)}")

    embeddings = {}
    batch_size = args.batch_size
    print(f"  批处理大小: {batch_size}")

    all_texts = [gene_texts[gid] for gid in gene_ids]

    with torch.no_grad():
        for i in tqdm(range(0, len(all_texts), batch_size), desc="生成嵌入"):
            batch_texts = all_texts[i:i + batch_size]
            batch_gene_ids = gene_ids[i:i + batch_size]

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
            
            # Mean Pooling（考虑attention mask，更好地利用所有token信息）
            # attention_mask = inputs['attention_mask'].unsqueeze(-1)  # [batch, seq_len, 1]
            # masked_embeddings = outputs.last_hidden_state * attention_mask  # 屏蔽padding
            # sum_embeddings = masked_embeddings.sum(dim=1)  # [batch, hidden_dim]
            # sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # 防止除零
            # mean_embeddings = (sum_embeddings / sum_mask).cpu().float().numpy()

            for gene_id, embedding in zip(batch_gene_ids, cls_embeddings):
                embeddings[gene_id] = embedding

# 最后基因文本嵌入顺序与.global_ids_path中的索引顺序完全一致（按照字符串的值排序），与datahandler无关
    ordered_embeddings = np.array([embeddings[gid] for gid in gene_ids], dtype=np.float32)

    print(f"\n✓ 生成完成")
    print(f"  嵌入形状: {ordered_embeddings.shape}")

    print(f"\n[5/5] 保存嵌入和ID映射...")
    output_dir = os.path.dirname(args.output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  已创建输出目录: {output_dir}")

    # 保存前先删除旧文件（如果存在）
    if os.path.exists(args.output_path):
        os.remove(args.output_path)
        print(f"  已删除旧的嵌入文件")
    
    np.save(args.output_path, ordered_embeddings)
    print(f"✓ 嵌入已保存到: {args.output_path}")

    # 保存ID映射前也先删除旧文件
    if os.path.exists(mapping_output_path):
        os.remove(mapping_output_path)
        print(f"  已删除旧的ID映射文件")
    
    gene_id_mapping = {gene_id: idx for idx, gene_id in enumerate(gene_ids)}
    with open(mapping_output_path, 'w', encoding='utf-8') as f:
        json.dump(gene_id_mapping, f, indent=2)
    print(f"✓ 基因ID映射已保存到: {mapping_output_path}")

    print("\n" + "=" * 80)
    print("✓ 所有步骤完成！")
    print("=" * 80)


def load_gene_descriptions(mapping_path):
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"找不到基因描述文件: {mapping_path}")
    print(f"  读取基因描述文件: {mapping_path}")

    if mapping_path.suffix.lower() == '.json':
        with open(mapping_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    if mapping_path.suffix.lower() == '.csv':
        import pandas as pd
        df = pd.read_csv(mapping_path, encoding='utf-8')
        if 'GeneID' in df.columns and 'LLM_Text' in df.columns:
            return {
                str(row['GeneID']): row['LLM_Text']
                for _, row in df[['GeneID', 'LLM_Text']].dropna().iterrows()
            }
        raise ValueError("CSV 文件缺少 'GeneID' 或 'LLM_Text' 列")

    raise ValueError(f"不支持的基因描述文件格式: {mapping_path.suffix}")


def build_default_args():
    dataset = getattr(train_args, 'data', 'DrugBank')
    dataset_dir = DATA_ROOT / dataset
    gene_text_dir = dataset_dir / 'gene_text'

    # global_ids_2.json 位于对应数据集目录下，例如
    # Data/DGIdb/global_ids_2.json 或 Data/DrugBank/global_ids_2.json
    global_ids_path = dataset_dir / 'global_ids_2.json'
    mapping_candidates = [
        gene_text_dir / 'gene_embeddings_txt.json',
        gene_text_dir / 'gene_text.json',
    ]
    for candidate in mapping_candidates:  #  定义候选文件列表
        if candidate.exists():
            gene_mapping_path = candidate
            break
    else:
        gene_mapping_path = mapping_candidates[0]

    # 添加bert前缀和cls后缀
    output_path = gene_text_dir / f"ft_bert_{dataset.lower()}_gene_emd_cls.npy"

    return SimpleNamespace(
        model_cache_dir=str(DEFAULT_MODEL_CACHE),
        global_ids_path=str(global_ids_path),
        gene_mapping_path=str(gene_mapping_path),
        output_path=str(output_path),
        batch_size=8,  # BioLinkBERT可以使用更大的batch size
    )


if __name__ == "__main__":
    args = build_default_args()
    main(args)
