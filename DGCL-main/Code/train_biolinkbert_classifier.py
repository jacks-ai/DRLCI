"""
BioLinkBERT + MLP 分类器训练脚本
方案: 特征提取 + 下游分类器 (端到端训练，不冻结)
采用 InfoNCE 对比学习损失优化困难负样本
"""
import json
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from pathlib import Path
from tqdm import tqdm
import sys
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.sparse import csr_matrix
import time
from datetime import datetime
from torch.amp import autocast, GradScaler
import argparse

# ==================== 解析命令行参数 ====================
def parse_and_extract_custom_args():
    learning_rate = 4e-5
    data = 'DGIdb'
    gpu = 0

    new_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--learning_rate' and i + 1 < len(sys.argv):
            learning_rate = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--data' and i + 1 < len(sys.argv):
            data = sys.argv[i + 1]
            new_argv.extend([sys.argv[i], sys.argv[i + 1]])
            i += 2
        elif sys.argv[i] == '--gpu' and i + 1 < len(sys.argv):
            gpu = int(sys.argv[i + 1])
            new_argv.extend([sys.argv[i], sys.argv[i + 1]])
            i += 2
        else:
            new_argv.append(sys.argv[i])
            i += 1

    sys.argv = new_argv
    return learning_rate, data, gpu

CUSTOM_LEARNING_RATE, DATASET_NAME, GPU_ID = parse_and_extract_custom_args()

CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.append(str(CODE_DIR))

from DataHandler import DataHandler

# ==================== 路径配置 ====================
timestamp = datetime.now().strftime("%m%d_%H%M%S")
DATA_ROOT = PROJECT_ROOT / 'Data' / DATASET_NAME
MODEL_CACHE = Path(r"/mnt/data/huangpeng/DGCL/mymodel/BioLinkBERT")
DRUG_TEXT_DIR = DATA_ROOT / 'drug_text'
DRUG_DESC_CANDIDATES = [
    DRUG_TEXT_DIR / 'mixed_drug_descriptions.csv',
    DRUG_TEXT_DIR / 'drug_text.json',
]
for _candidate in DRUG_DESC_CANDIDATES:
    if _candidate.exists():
        DRUG_DESC_PATH = _candidate
        break
else:
    DRUG_DESC_PATH = DRUG_DESC_CANDIDATES[0]
GENE_DESC_JSON = DATA_ROOT / 'gene_text' / 'gene_embeddings_txt.json'
GLOBAL_IDS_JSON = DATA_ROOT / 'global_ids_DGIdb.json'
SERVER_DATA_ROOT = Path(r"/mnt/data/huangpeng/DGCL/DGCL-main/Data")
RAG_GENE_CACHE_DIR = SERVER_DATA_ROOT / DATASET_NAME / "rag"
TRAIN_GENE_EMBED_CACHE = RAG_GENE_CACHE_DIR / "pretrained_train_gene_cls.npz"
ALL_GENE_EMBED_CACHE = RAG_GENE_CACHE_DIR / "pretrained_all_gene_cls.npz"
HARD_NEG_TOP_K = 5
TRAIN_HARD_NEG_CACHE = RAG_GENE_CACHE_DIR / "train_hard_neg_similar_varlen.npz"
BERT_SAVE_DIR = SERVER_DATA_ROOT / DATASET_NAME / "rag" / "bert"
GENE_ENTITY_MAX_CHARS = 220
GENE_ENCODE_BATCH = 32
GENE_ENCODE_MAX_LENGTH = 256
MAX_SIM_MATRIX_BYTES = int(3.0 * (1024**3))

LOG_DIR = Path("/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/rag/bert/log")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"training_log_{timestamp}.txt"

# ==================== 超参数配置 ====================
TRAIN_CONFIG = {
    'batch_size': 16,
    'learning_rate': CUSTOM_LEARNING_RATE,
    'num_epochs': 15,
    'warmup_ratio': 0.1,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    'save_steps': 500,
    'eval_steps': 500,
    'fp16': True,
    'max_length': 512,

    # --- 新增的 InfoNCE 损失超参数 ---
    'contrastive_weight': 1.0,             # 损失权重
    'contrastive_temperature': 0.1,        # InfoNCE 温度超参数 τ
    'contrastive_use_cosine_weight': False, # 是否额外保留困难样本的先验权重
    'contrastive_neg_chunk_size': 8,       # 负样本分块处理，防止显存溢出
    'gradient_checkpointing': True,
}

PROMPT_TEMPLATE_TEXT = "Drug: {drug_desc}"
PROMPT_TEMPLATE_TEXT_PAIR = "Gene: {gene_desc}"

INTERACTION_TYPES = [
    "Agonist", "Antagonist", "Antibody", "Modulator", "Blocker", "Binder",
    "Potentiator", "Cofactor", "Ligand", "Inhibitor", "Activator",
    "Partial agonist", "Positive modulator", "Allosteric modulator"
]

def log_and_print(message, log_file=None):
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')

# ==================== 模型定义 ====================
class ClassifierLayer(nn.Module):
    def __init__(self, input_dim=1024, num_classes=14):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(128, num_classes)

    def forward(self, embeds):
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        ret = self.lin2(embeds)
        return ret

class BioLinkBERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=14):
        super().__init__()
        self.bert = bert_model
        hidden_dim = bert_model.config.hidden_size
        self.classifier = ClassifierLayer(input_dim=hidden_dim, num_classes=num_classes)

    @staticmethod
    def segment_mean_pool(hidden_states, attention_mask, token_type_ids, segment_id):
        m = attention_mask.bool() & (token_type_ids == segment_id)
        m = m.unsqueeze(-1).to(hidden_states.dtype)
        denom = m.sum(dim=1).clamp(min=1e-6)
        return (hidden_states * m).sum(dim=1) / denom

    def forward_backbone(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        last = outputs.last_hidden_state
        cls_embed = last[:, 0, :]
        drug_pool = self.segment_mean_pool(last, attention_mask, token_type_ids, 0)
        gene_pool = self.segment_mean_pool(last, attention_mask, token_type_ids, 1)
        return cls_embed, drug_pool, gene_pool

    def forward(self, input_ids, attention_mask, token_type_ids=None, return_pair_pools=False):
        cls_embed, drug_pool, gene_pool = self.forward_backbone(
            input_ids, attention_mask, token_type_ids
        )
        logits = self.classifier(cls_embed)
        if return_pair_pools:
            return logits, drug_pool, gene_pool
        return logits

# ==================== 数据集定义 ====================
class DrugGeneDataset(Dataset):
    def __init__(self, drug_ids, gene_ids, labels, drug_descriptions, gene_descriptions,
                 drug_ids_global, gene_ids_global, tokenizer, max_length,
                 hard_neg_offsets=None, hard_neg_gene_idx=None, hard_neg_cos_sim=None):
        self.drug_ids = drug_ids
        self.gene_ids = gene_ids
        self.labels = labels
        self.drug_descriptions = drug_descriptions
        self.gene_descriptions = gene_descriptions
        self.drug_ids_global = drug_ids_global
        self.gene_ids_global = gene_ids_global
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hard_neg_offsets = hard_neg_offsets
        self.hard_neg_gene_idx = hard_neg_gene_idx
        self.hard_neg_cos_sim = hard_neg_cos_sim

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        drug_idx = int(self.drug_ids[idx])
        gene_idx = int(self.gene_ids[idx])
        label = self.labels[idx]

        drug_id = self.drug_ids_global[drug_idx]
        gene_id = self.gene_ids_global[gene_idx]

        drug_desc = self.drug_descriptions.get(drug_id, f"Drug {drug_id}")[:292]
        _gd = lookup_gene_description(self.gene_descriptions, gene_id)
        gene_desc = (_gd if _gd is not None else f"Gene {gene_id}")[:220]

        encoding = self.tokenizer(
            text=f"Drug: {drug_desc}",
            text_pair=f"Gene: {gene_desc}",
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding['token_type_ids'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
            'drug_idx': torch.tensor(drug_idx, dtype=torch.long),
            'gene_idx': torch.tensor(gene_idx, dtype=torch.long),
        }
        if self.hard_neg_offsets is not None:
            lo = int(self.hard_neg_offsets[idx])
            hi = int(self.hard_neg_offsets[idx + 1])
            gns = self.hard_neg_gene_idx[lo:hi]
            if self.hard_neg_cos_sim is not None:
                coss = np.asarray(self.hard_neg_cos_sim[lo:hi], dtype=np.float32)
            else:
                coss = np.ones(len(gns), dtype=np.float32)
            pad_g = np.full((HARD_NEG_TOP_K,), -1, dtype=np.int64)
            pad_w = np.zeros((HARD_NEG_TOP_K,), dtype=np.float32)
            L = min(int(len(gns)), HARD_NEG_TOP_K)
            if L > 0:
                pad_g[:L] = np.asarray(gns[:L], dtype=np.int64)
                pad_w[:L] = coss[:L]
            result['hard_neg_genes'] = torch.from_numpy(pad_g)
            result['hard_neg_weight'] = torch.from_numpy(pad_w)
        else:
            result['hard_neg_genes'] = torch.full((HARD_NEG_TOP_K,), -1, dtype=torch.long)
            result['hard_neg_weight'] = torch.zeros(HARD_NEG_TOP_K, dtype=torch.float32)

        return result

# ==================== 数据加载函数 ====================
def load_global_ids():
    path_resolved = str(GLOBAL_IDS_JSON.resolve()) if hasattr(GLOBAL_IDS_JSON, 'resolve') else str(GLOBAL_IDS_JSON)
    print(f"[索引文件] 读取路径: {path_resolved}")
    with open(GLOBAL_IDS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['drug_ids'], data['gene_ids']

def load_drug_descriptions():
    mapping_path = Path(DRUG_DESC_PATH)
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

def load_gene_descriptions():
    with open(GENE_DESC_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)

def lookup_gene_description(gene_descriptions, gid):
    if gid in gene_descriptions:
        return gene_descriptions[gid]
    s = str(gid)
    if s in gene_descriptions:
        return gene_descriptions[s]
    try:
        ik = int(s)
        if ik in gene_descriptions:
            return gene_descriptions[ik]
        if str(ik) in gene_descriptions:
            return gene_descriptions[str(ik)]
    except ValueError:
        pass
    return None

def verify_global_ids_vs_transductive_csv(drug_ids_global, gene_ids_global, log_file=None):
    train_csv = DATA_ROOT / "transductive" / "train.csv"
    test_csv = DATA_ROOT / "transductive" / "test.csv"
    if not train_csv.is_file() or not test_csv.is_file():
        return
    df_tr = pd.read_csv(train_csv, header=None)
    df_te = pd.read_csv(test_csv, header=None)
    df = pd.concat([df_tr, df_te], ignore_index=True)
    drugs_csv = sorted(pd.unique(df[0].astype(str)).tolist())
    genes_raw = pd.unique(df[1]).tolist()
    genes_csv = [str(x) for x in sorted(genes_raw, key=lambda x: int(x))]
    ok_d = drugs_csv == list(drug_ids_global)
    ok_g = genes_csv == list(gene_ids_global)
    if not ok_d or not ok_g:
        log_and_print("  ⚠ 索引可能有误，请检查", log_file)

def audit_gene_text_sample(gene_ids_global, gene_descriptions, log_file=None, k=5):
    pass

def load_training_and_test_data():
    print(f"\n[加载数据] 数据集: {DATASET_NAME}")
    handler = DataHandler()
    handler.LoadData()

    train_drugs, train_genes, train_labels = [], [], []
    for batch in handler.trnLoader:
        if len(batch) == 4:
            drugs, genes, labels, _ = batch
        else:
            drugs, genes, labels = batch[:3]
        train_drugs.extend(drugs.numpy())
        train_genes.extend(genes.numpy())
        train_labels.extend(labels.numpy())

    train_drugs = np.array(train_drugs)
    train_genes = np.array(train_genes)
    train_labels = np.array(train_labels)

    test_drugs, test_genes, test_labels = [], [], []
    for batch in handler.tstLoader:
        drugs, genes, labels = batch
        test_drugs.extend(drugs.numpy())
        test_genes.extend(genes.numpy())
        test_labels.extend(labels.numpy())

    test_drugs = np.array(test_drugs)
    test_genes = np.array(test_genes)
    test_labels = np.array(test_labels)

    return train_drugs, train_genes, train_labels, test_drugs, test_genes, test_labels

def ensure_precomputed_train_gene_embeddings(train_genes, gene_ids_global, gene_descriptions, device, log_file=None):
    cache_path = TRAIN_GENE_EMBED_CACHE
    if cache_path.is_file():
        return
    unique_gene_idx = np.unique(train_genes.astype(np.int64))
    unique_gene_idx.sort()
    texts = []
    for gidx in unique_gene_idx:
        gid = gene_ids_global[int(gidx)]
        desc = lookup_gene_description(gene_descriptions, gid)
        if desc is None: desc = f"Gene {gid}"
        texts.append(f"Gene: {str(desc)[:GENE_ENTITY_MAX_CHARS]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)
    enc_model = AutoModel.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True).to(device)
    enc_model.eval()

    all_rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), GENE_ENCODE_BATCH), desc="基因实体编码"):
            end = min(start + GENE_ENCODE_BATCH, len(texts))
            batch_texts = texts[start:end]
            encoding = tokenizer(batch_texts, max_length=GENE_ENCODE_MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}
            out = enc_model(**encoding)
            all_rows.append(out.last_hidden_state[:, 0, :].cpu().float().numpy())

    embeddings = np.concatenate(all_rows, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, gene_indices=unique_gene_idx.astype(np.int64), embeddings=embeddings)
    del enc_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def ensure_precomputed_all_gene_embeddings(gene_ids_global, gene_descriptions, device, log_file=None):
    cache_path = ALL_GENE_EMBED_CACHE
    if cache_path.is_file():
        return
    texts = []
    for gid in gene_ids_global:
        desc = lookup_gene_description(gene_descriptions, gid)
        if desc is None: desc = f"Gene {gid}"
        texts.append(f"Gene: {str(desc)[:GENE_ENTITY_MAX_CHARS]}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)
    enc_model = AutoModel.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True).to(device)
    enc_model.eval()

    all_rows = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), GENE_ENCODE_BATCH), desc="全基因实体编码"):
            end = min(start + GENE_ENCODE_BATCH, len(texts))
            encoding = tokenizer(texts[start:end], max_length=GENE_ENCODE_MAX_LENGTH, padding=True, truncation=True, return_tensors="pt")
            encoding = {k: v.to(device) for k, v in encoding.items()}
            all_rows.append(enc_model(**encoding).last_hidden_state[:, 0, :].cpu().float().numpy())

    embeddings = np.concatenate(all_rows, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, embeddings=embeddings)
    del enc_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def _transductive_pair_csr(num_drugs, num_genes, drug_ids_global, gene_ids_global, log_file=None):
    train_csv = DATA_ROOT / "transductive" / "train.csv"
    test_csv = DATA_ROOT / "transductive" / "test.csv"
    d_rev = {str(d): i for i, d in enumerate(drug_ids_global)}
    g_rev = {str(g): i for i, g in enumerate(gene_ids_global)}
    rows, cols = [], []
    for path in (train_csv, test_csv):
        if not path.is_file(): continue
        df = pd.read_csv(path, header=None)
        for _, row in df.iterrows():
            di, gi = d_rev.get(str(row[0])), g_rev.get(str(row[1]).strip())
            if di is not None and gi is not None:
                rows.append(di); cols.append(gi)
    if not rows: return None
    return csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(num_drugs, num_genes))

def ensure_train_hard_negatives_similar_top5(train_drugs, train_genes, train_labels, num_drugs, num_genes, drug_ids_global, gene_ids_global, log_file=None):
    cache_path = TRAIN_HARD_NEG_CACHE
    if cache_path.is_file(): return
    z = np.load(ALL_GENE_EMBED_CACHE, allow_pickle=True)
    E = np.asarray(z["embeddings"], dtype=np.float32)
    E_norm = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)

    td, tg = train_drugs.astype(np.int64), train_genes.astype(np.int64)
    mask_csr = _transductive_pair_csr(num_drugs, num_genes, drug_ids_global, gene_ids_global, log_file)
    if mask_csr is None:
        mask_csr = csr_matrix((np.ones(len(td), dtype=np.float32), (td, tg)), shape=(num_drugs, num_genes))

    S = (E_norm @ E_norm.T).astype(np.float32)
    N, k = len(td), HARD_NEG_TOP_K
    buf, neg_inf = np.empty(num_genes, dtype=np.float32), np.float32(-1e30)

    row_lens, flat_idx_parts, flat_sim_parts = [], [], []
    for i in tqdm(range(N), desc="困难负样本 top5"):
        d, g = int(td[i]), int(tg[i])
        np.copyto(buf, S[g])
        buf[mask_csr[d].indices] = neg_inf
        buf[g] = neg_inf
        finite = np.isfinite(buf)
        if not finite.any():
            row_lens.append(0)
            continue
        sub_idx = np.flatnonzero(finite)
        sub_scores = buf[sub_idx]
        sub_k = min(k, int(sub_scores.shape[0]))
        top_local = np.argpartition(-sub_scores, sub_k - 1)[:sub_k]
        top_local = top_local[np.argsort(-sub_scores[top_local])]

        row_lens.append(int(top_local.shape[0]))
        flat_idx_parts.append(sub_idx[top_local].astype(np.int32))
        flat_sim_parts.append(sub_scores[top_local].astype(np.float32))

    hard_neg_offsets = np.zeros(N + 1, dtype=np.int64)
    for i in range(N): hard_neg_offsets[i + 1] = hard_neg_offsets[i] + row_lens[i]
    hard_neg_gene_idx = np.concatenate(flat_idx_parts, axis=0) if flat_idx_parts else np.array([], dtype=np.int32)
    hard_neg_cos_sim = np.concatenate(flat_sim_parts, axis=0) if flat_sim_parts else np.array([], dtype=np.float32)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, train_drug_idx=td, train_gene_idx=tg, train_label=train_labels.astype(np.int64),
                        hard_neg_offsets=hard_neg_offsets, hard_neg_gene_idx=hard_neg_gene_idx, hard_neg_cos_sim=hard_neg_cos_sim)

# ==================== 训练和评估函数 ====================
def tokenize_drug_gene_pairs_flat(drug_idx_flat, gene_idx_flat, drug_descriptions, gene_descriptions, drug_ids_global, gene_ids_global, tokenizer, max_length):
    input_ids, attention_mask, token_type_ids = [], [], []
    for di, gi in zip(drug_idx_flat.tolist(), gene_idx_flat.tolist()):
        drug_id = drug_ids_global[di]
        gene_id = gene_ids_global[gi]
        drug_desc = drug_descriptions.get(drug_id, f"Drug {drug_id}")[:292]
        _gd = lookup_gene_description(gene_descriptions, gene_id)
        gene_desc = (_gd if _gd is not None else f"Gene {gene_id}")[:220]
        enc = tokenizer(text=f"Drug: {drug_desc}", text_pair=f"Gene: {gene_desc}",
                        max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
        input_ids.append(enc['input_ids'].squeeze(0))
        attention_mask.append(enc['attention_mask'].squeeze(0))
        token_type_ids.append(enc['token_type_ids'].squeeze(0))
    return {
        'input_ids': torch.stack(input_ids, dim=0),
        'attention_mask': torch.stack(attention_mask, dim=0),
        'token_type_ids': torch.stack(token_type_ids, dim=0),
    }

def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None, *,
                tokenizer=None, drug_descriptions=None, gene_descriptions=None,
                drug_ids_global=None, gene_ids_global=None, max_length=512,
                contrastive_weight=0.0, contrastive_temperature=0.1, contrastive_use_cosine_weight=False):

    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    use_fp16 = scaler is not None

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    neg_chunk = int(TRAIN_CONFIG.get('contrastive_neg_chunk_size', 8))

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        loss_contrastive_val = 0.0
        did_contrastive = False

        # --- 1. 计算正样本网络，提取特征 ---
        with autocast('cuda', enabled=use_fp16):
            logits, drug_p, gene_p = model(input_ids, attention_mask, token_type_ids, return_pair_pools=True)
            loss_ce = F.cross_entropy(logits, labels)

        need_contrastive = (contrastive_weight > 0 and tokenizer is not None and batch.get('hard_neg_genes') is not None)

        if need_contrastive:
            hn = batch['hard_neg_genes']
            hw = batch['hard_neg_weight']
            dix = batch['drug_idx']
            valid = hn >= 0

            if valid.any():
                did_contrastive = True
                B, K = hn.shape

                # 构建用于扁平化抽取的映射
                b_flat = torch.arange(B, device=valid.device, dtype=torch.long).unsqueeze(1).expand_as(hn)[valid]
                di_exp = dix.unsqueeze(1).expand_as(hn)[valid]
                g_flat = hn[valid]
                w_flat = hw[valid].float()
                m_total = int(valid.sum().item())

                # --- 2. 收集所有负样本的特征 (Chunk by Chunk 以防爆显存) ---
                gene_negs_flat_list = []
                for s in range(0, m_total, neg_chunk):
                    e = min(s + neg_chunk, m_total)
                    di_c = di_exp[s:e]
                    g_c = g_flat[s:e]

                    neg_enc = tokenize_drug_gene_pairs_flat(
                        di_c.cpu(), g_c.cpu(), drug_descriptions, gene_descriptions,
                        drug_ids_global, gene_ids_global, tokenizer, max_length
                    )
                    ni = neg_enc['input_ids'].to(device, non_blocking=True)
                    na = neg_enc['attention_mask'].to(device, non_blocking=True)
                    nt = neg_enc['token_type_ids'].to(device, non_blocking=True)

                    with autocast('cuda', enabled=use_fp16):
                        _, _, gene_neg = model(ni, na, nt, return_pair_pools=True)
                    gene_negs_flat_list.append(gene_neg)

                all_gene_negs_flat = torch.cat(gene_negs_flat_list, dim=0)

                # --- 3. 精确计算 InfoNCE 对比损失 ---
                with autocast('cuda', enabled=use_fp16):
                    # L2 归一化用于余弦相似度
                    drug_p_norm = F.normalize(drug_p, dim=-1)
                    gene_p_norm = F.normalize(gene_p, dim=-1)
                    gene_negs_norm = F.normalize(all_gene_negs_flat, dim=-1)

                    # 正样本相似度: [B]
                    pos_sim = (drug_p_norm * gene_p_norm).sum(dim=-1)
                    # 负样本相似度: [m_total]
                    neg_sim = (drug_p_norm[b_flat] * gene_negs_norm).sum(dim=-1)

                    if contrastive_use_cosine_weight:
                        neg_sim = neg_sim * w_flat.to(device).clamp(min=1e-6)

                    # 缩放温度参数 \tau
                    tau = contrastive_temperature
                    pos_sim_scaled = pos_sim / tau
                    neg_sim_scaled = neg_sim / tau

                    # 还原负样本维度到 [B, K]，无效填 -inf
                    neg_sim_matrix = torch.full((B, K), -float('inf'), device=device)
                    neg_sim_matrix[valid] = neg_sim_scaled

                    # 合并正负样本相似度，构成完整的分母集合 [B, 1 + K]
                    all_sims = torch.cat([pos_sim_scaled.unsqueeze(1), neg_sim_matrix], dim=1)

                    # 使用 torch.logsumexp 解决 FP16 溢出问题 (等价于 -log(exp(pos) / sum(exp)))
                    # 公式推导: L = -pos + log(exp(pos) + sum(exp(neg)))
                    loss_infonce_per_sample = torch.logsumexp(all_sims, dim=1) - pos_sim_scaled

                    # 仅在至少有一个有效负样本的 Batch 行上计算平均
                    valid_b_mask = valid.any(dim=1)
                    loss_infonce_mean = loss_infonce_per_sample[valid_b_mask].mean()

                    # 乘上损失权重
                    loss_contrastive = contrastive_weight * loss_infonce_mean

                loss_total_tensor = loss_ce + loss_contrastive
                loss_contrastive_val = float(loss_contrastive.detach().item())
            else:
                loss_total_tensor = loss_ce
        else:
            loss_total_tensor = loss_ce

        loss_total = float(loss_total_tensor.detach().item())

        # --- 4. 统一反向传播 ---
        if use_fp16:
            scaler.scale(loss_total_tensor).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_total_tensor.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['max_grad_norm'])
            optimizer.step()

        scheduler.step()

        total_loss += loss_total
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        postfix = {'loss': loss_total}
        if did_contrastive:
            postfix['infonce'] = loss_contrastive_val
        progress_bar.set_postfix(postfix)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    all_probs = np.array(all_probs)
    top5_preds = np.argsort(all_probs, axis=1)[:, -5:]
    top5_accuracy = np.mean([label in top5_preds[i] for i, label in enumerate(all_labels)])

    return accuracy, top5_accuracy, all_preds, all_labels

# ==================== 主函数 ====================
def main():
    log_and_print("=" * 80, LOG_FILE)
    log_and_print("BioLinkBERT + MLP 分类器训练", LOG_FILE)
    log_and_print("=" * 80, LOG_FILE)
    log_and_print(f"数据集: {DATASET_NAME}", LOG_FILE)
    log_and_print(f"训练配置: {TRAIN_CONFIG}", LOG_FILE)

    if torch.cuda.is_available():
        device = torch.device(f'cuda:{GPU_ID}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    log_and_print("\n[1/12] 加载全局数据...", LOG_FILE)
    drug_ids_global, gene_ids_global = load_global_ids()
    drug_descriptions = load_drug_descriptions()
    gene_descriptions = load_gene_descriptions()

    log_and_print("\n[2/12] 加载训练数据和测试数据...", LOG_FILE)
    train_drugs, train_genes, train_labels, test_drugs, test_genes, test_labels = load_training_and_test_data()

    ensure_precomputed_train_gene_embeddings(train_genes, gene_ids_global, gene_descriptions, device, LOG_FILE)
    ensure_precomputed_all_gene_embeddings(gene_ids_global, gene_descriptions, device, LOG_FILE)

    verify_global_ids_vs_transductive_csv(drug_ids_global, gene_ids_global, LOG_FILE)
    audit_gene_text_sample(gene_ids_global, gene_descriptions, LOG_FILE)

    ensure_train_hard_negatives_similar_top5(
        train_drugs, train_genes, train_labels, len(drug_ids_global), len(gene_ids_global),
        drug_ids_global, gene_ids_global, LOG_FILE
    )

    hnz = np.load(TRAIN_HARD_NEG_CACHE, allow_pickle=True)
    hard_neg_offsets = hnz['hard_neg_offsets']
    hard_neg_gene_idx = hnz['hard_neg_gene_idx']
    hard_neg_cos_sim = hnz['hard_neg_cos_sim']

    log_and_print("\n[7/12] 加载BioLinkBERT模型...", LOG_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)
    bert_model = AutoModel.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)

    model = BioLinkBERTClassifier(bert_model, num_classes=14)
    model = model.to(device)
    if TRAIN_CONFIG.get('gradient_checkpointing'):
        model.bert.gradient_checkpointing_enable()

    max_length = TRAIN_CONFIG['max_length']

    train_dataset = DrugGeneDataset(
        train_drugs, train_genes, train_labels, drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global, tokenizer, max_length,
        hard_neg_offsets=hard_neg_offsets, hard_neg_gene_idx=hard_neg_gene_idx, hard_neg_cos_sim=hard_neg_cos_sim,
    )

    test_dataset = DrugGeneDataset(
        test_drugs, test_genes, test_labels, drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global, tokenizer, max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)

    log_and_print("\n[9/12] 设置优化器...", LOG_FILE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=TRAIN_CONFIG['learning_rate'], weight_decay=TRAIN_CONFIG['weight_decay'])

    total_steps = len(train_loader) * TRAIN_CONFIG['num_epochs']
    warmup_steps = int(total_steps * TRAIN_CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler(device='cuda') if TRAIN_CONFIG['fp16'] and torch.cuda.is_available() else None

    log_and_print("\n[10/12] 开始训练...", LOG_FILE)
    best_test_acc = 0.0
    best_epoch = 0
    best_test_preds = None
    best_test_labels_eval = None
    BERT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_save_path = None
    training_start_time = time.time()
    official_test_acc_history = []

    for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
        epoch_start_time = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, scaler,
            tokenizer=tokenizer, drug_descriptions=drug_descriptions, gene_descriptions=gene_descriptions,
            drug_ids_global=drug_ids_global, gene_ids_global=gene_ids_global, max_length=max_length,
            contrastive_weight=TRAIN_CONFIG['contrastive_weight'],
            contrastive_temperature=TRAIN_CONFIG['contrastive_temperature'],
            contrastive_use_cosine_weight=TRAIN_CONFIG['contrastive_use_cosine_weight']
        )
        epoch_time = time.time() - epoch_start_time

        test_acc, test_top5_acc, test_preds, test_labels_eval = evaluate(model, test_loader, device)
        official_test_acc_history.append(test_acc)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_test_preds = test_preds
            best_test_labels_eval = test_labels_eval
            acc_tag = int(round(float(test_acc) * 1000))
            save_path = BERT_SAVE_DIR / f'best_biolinkbert_{DATASET_NAME}_{timestamp}_acc{acc_tag}.pt'
            if best_save_path is not None and best_save_path != save_path and best_save_path.exists():
                best_save_path.unlink()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'bert_state_dict': model.bert.state_dict(),
                'classifier_state_dict': model.classifier.state_dict(),
                'test_acc': test_acc,
            }, save_path)
            best_save_path = save_path

    # ... 省略评估部分，与原版完全一致 ...

if __name__ == "__main__":
    main()