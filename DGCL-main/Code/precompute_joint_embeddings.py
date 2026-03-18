"""
联合编码预计算脚本：与 train_biolinkbert_classifier 完全一致
- 输入：train.csv / test.csv 中成对出现的 (drug_id, gene_id)
- 编码：BioLinkBERT tokenizer(text="Drug: {desc}", text_pair="Gene: {desc}")，取 [CLS] hidden state
- 输出：joint_embeddings_train.npy、joint_embeddings_test.npy，行序与 CSV 行序一致，无需额外 joint_pairs 文件
- precompute_joint_embeddings.py：直接读 CSV 里的 drug_id / gene_id（字符串），用这些 ID 去查描述文本。
对 train/test 的每行都算（即使重复出现同一对也会重复计算），强调“与训练脚本一致、按行直接用”
"""
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

CODE_DIR = Path(__file__).resolve().parents[2]  # Code/
sys.path.insert(0, str(CODE_DIR))
from Params import args as train_args

# 与 bert_drug_emd / bert_gene_emd 一致的服务器路径
PROJECT_ROOT = CODE_DIR.parent
DATA_ROOT = PROJECT_ROOT / "Data"
SERVER_DATA_ROOT = Path(r"/mnt/data/huangpeng/DGCL/DGCL-main/Data")  # 服务器上 Data 目录，与 Params/bert 一致
DEFAULT_MODEL_CACHE = Path(r"/mnt/data/huangpeng/DGCL/mymodel/BioLinkBERT")

DATASET_NAME = getattr(train_args, "data", "DrugBank")
if DATASET_NAME == "DrugBank":
    FINE_TUNED_CKPT = Path(
        "/mnt/data/huangpeng/DGCL/DGCL-main/Code/bert/best_biolinkbert_only_DrugBank_0303_020934.pt"
    )
else:
    FINE_TUNED_CKPT = Path(
        "/mnt/data/huangpeng/DGCL/DGCL-main/Code/bert/best_biolinkbert_only_0302_123530.pt"
    )

# 与 train_biolinkbert_classifier 一致的截断长度
DRUG_DESC_MAX_CHARS = 292
GENE_DESC_MAX_CHARS = 220
MAX_LENGTH = 512


def load_drug_descriptions(mapping_path):
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"找不到药物描述文件: {mapping_path}")
    if mapping_path.suffix.lower() == ".json":
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if mapping_path.suffix.lower() == ".csv":
        df = pd.read_csv(mapping_path, encoding="utf-8")
        for id_col, text_col in [("DrugID", "LLM_Text"), ("DrugID", "LLM_Input_Text")]:
            if id_col in df.columns and text_col in df.columns:
                return {
                    str(row[id_col]): row[text_col]
                    for _, row in df[[id_col, text_col]].dropna().iterrows()
                }
        raise ValueError(f"CSV 中找不到支持的列组合")
    raise ValueError(f"不支持的格式: {mapping_path.suffix}")


def load_gene_descriptions(mapping_path):
    mapping_path = Path(mapping_path)
    if not mapping_path.exists():
        raise FileNotFoundError(f"找不到基因描述文件: {mapping_path}")
    if mapping_path.suffix.lower() == ".json":
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    if mapping_path.suffix.lower() == ".csv":
        df = pd.read_csv(mapping_path, encoding="utf-8")
        if "GeneID" in df.columns and "LLM_Text" in df.columns:
            return {
                str(row["GeneID"]): row["LLM_Text"]
                for _, row in df[["GeneID", "LLM_Text"]].dropna().iterrows()
            }
        raise ValueError("CSV 缺少 GeneID 或 LLM_Text")
    raise ValueError(f"不支持的格式: {mapping_path.suffix}")


def load_pairs_from_csv(csv_path):
    """读取无表头 CSV：drug_id, gene_id, label。返回 (drug_ids, gene_ids) 均为字符串列表，与行序一致。"""
    df = pd.read_csv(
        csv_path, header=None, names=["d_nodes", "g_nodes", "relations"], dtype={"d_nodes": str, "g_nodes": str}
    )
    drug_ids = df["d_nodes"].astype(str).tolist()
    gene_ids = df["g_nodes"].astype(str).tolist()
    return drug_ids, gene_ids


def encode_pairs_and_save(
    drug_ids, gene_ids, drug_descriptions, gene_descriptions,
    tokenizer, model, device, output_path, batch_size=16,
):
    """
    对 (drug_ids[i], gene_ids[i]) 做联合编码，取 [CLS]，按行序写入 .npy。
    与 train_biolinkbert_classifier 的 prompt 与截断一致。
    """
    drug_texts = []
    gene_texts = []
    for did, gid in zip(drug_ids, gene_ids):
        drug_desc = drug_descriptions.get(did, f"Drug {did}")[:DRUG_DESC_MAX_CHARS]
        gene_desc = gene_descriptions.get(gid, f"Gene {gid}")[:GENE_DESC_MAX_CHARS]
        drug_texts.append(f"Drug: {drug_desc}")
        gene_texts.append(f"Gene: {gene_desc}")

    all_embeddings = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(drug_texts), batch_size), desc="联合编码"):
            end = min(start + batch_size, len(drug_texts))
            batch_drug = drug_texts[start:end]
            batch_gene = gene_texts[start:end]

            encoding = tokenizer(
                text=batch_drug,
                text_pair=batch_gene,
                max_length=MAX_LENGTH,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}
            outputs = model(**encoding)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_embeddings.append(cls_emb)

    out = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, out)
    return out.shape


def build_default_args():
    dataset = getattr(train_args, "data", "DGIdb")
    # 使用服务器路径，与 bert_drug_emd / bert_gene_emd 一致（Params 中嵌入路径也为 /mnt/.../Data/...）
    dataset_dir = SERVER_DATA_ROOT / dataset
    transductive_dir = dataset_dir / "transductive"
    drug_text_dir = dataset_dir / "drug_text"
    gene_text_dir = dataset_dir / "gene_text"

    train_csv = transductive_dir / "train.csv"
    test_csv = transductive_dir / "test.csv"

    drug_candidates = [
        drug_text_dir / "mixed_drug_descriptions.csv",
        drug_text_dir / "drug_text.json",
    ]
    drug_mapping = next((p for p in drug_candidates if p.exists()), drug_candidates[0])

    gene_candidates = [
        gene_text_dir / "gene_embeddings_txt.json",
        gene_text_dir / "gene_text.json",
        dataset_dir / f"{dataset}_gene_embeddings_txt_train.json",
    ]
    gene_mapping = next((p for p in gene_candidates if p.exists()), gene_candidates[0])

    return {
        "dataset": dataset,
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "drug_mapping_path": str(drug_mapping),
        "gene_mapping_path": str(gene_mapping),
        "output_dir": str(transductive_dir),
        "model_cache_dir": str(DEFAULT_MODEL_CACHE),
        "batch_size": 16,
    }


def main():
    from types import SimpleNamespace
    a = SimpleNamespace(**build_default_args())

    print("=" * 80)
    print("联合编码预计算（与 train_biolinkbert_classifier 一致）")
    print("=" * 80)
    print(f"  数据集: {a.dataset}")
    print(f"  train.csv: {a.train_csv}")
    print(f"  test.csv:  {a.test_csv}")
    print(f"  输出目录: {a.output_dir}")
    print(f"  药物描述: {a.drug_mapping_path}")
    print(f"  基因描述: {a.gene_mapping_path}")
    print(f"  batch_size: {a.batch_size}")

    print("\n[1/5] 读取 train/test CSV 中的 (drug_id, gene_id) 对...")
    train_drugs, train_genes = load_pairs_from_csv(a.train_csv)
    test_drugs, test_genes = load_pairs_from_csv(a.test_csv)
    print(f"  ✓ 训练对数: {len(train_drugs)}, 测试对数: {len(test_drugs)}")

    print("\n[2/5] 加载药物/基因描述...")
    drug_descriptions = load_drug_descriptions(a.drug_mapping_path)
    gene_descriptions = load_gene_descriptions(a.gene_mapping_path)
    print(f"  ✓ 药物描述数: {len(drug_descriptions)}, 基因描述数: {len(gene_descriptions)}")

    print("\n[3/5] 加载 BioLinkBERT...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(
        a.model_cache_dir, local_files_only=True, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        a.model_cache_dir, local_files_only=True, trust_remote_code=True
    )
    if Path(FINE_TUNED_CKPT).is_file():
        try:
            ckpt = torch.load(FINE_TUNED_CKPT, map_location="cpu", weights_only=False)
            state = ckpt.get("bert_state_dict", ckpt)
            model.load_state_dict(state, strict=False)
            print(f"  ✓ 已加载微调权重: {FINE_TUNED_CKPT}")
        except Exception as e:
            print(f"  ⚠ 微调权重加载失败，使用预训练: {e}")
    else:
        print(f"  未找到微调权重，使用预训练 BioLinkBERT")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    print(f"  隐藏维度: {model.config.hidden_size}")

    print("\n[4/5] 生成训练集联合嵌入（行序与 train.csv 一致）...")
    train_path = Path(a.output_dir) / "joint_embeddings_train.npy"
    train_shape = encode_pairs_and_save(
        train_drugs, train_genes, drug_descriptions, gene_descriptions,
        tokenizer, model, device, str(train_path), batch_size=a.batch_size,
    )
    print(f"  ✓ 已保存: {train_path}, shape={train_shape}")

    print("\n[5/5] 生成测试集联合嵌入（行序与 test.csv 一致）...")
    test_path = Path(a.output_dir) / "joint_embeddings_test.npy"
    test_shape = encode_pairs_and_save(
        test_drugs, test_genes, drug_descriptions, gene_descriptions,
        tokenizer, model, device, str(test_path), batch_size=a.batch_size,
    )
    print(f"  ✓ 已保存: {test_path}, shape={test_shape}")

    print("\n" + "=" * 80)
    print("说明：无需 joint_pairs 文件。train.csv 第 i 行 → joint_embeddings_train.npy[i]；test.csv 同理。")
    print("=" * 80)


if __name__ == "__main__":
    main()
