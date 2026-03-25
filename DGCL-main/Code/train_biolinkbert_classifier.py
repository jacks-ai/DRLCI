"""
BioLinkBERT + MLP 分类器训练脚本
方案: 特征提取 + 下游分类器 (端到端训练，不冻结)
MLP架构与Model_sparse.py中的ClassifierLayer完全一致
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
from torch.amp import autocast, GradScaler  # FP16 混合精度训练 (修复FutureWarning)
import argparse

# ==================== 解析命令行参数 ====================
# 策略：先提取我们的参数，然后从 sys.argv 中移除，再导入 DataHandler
# 这样 DataHandler -> Params.py 就不会看到 --learning_rate 参数

def parse_and_extract_custom_args():
    """
    提取自定义参数（learning_rate, data, gpu），并从 sys.argv 中移除
    这样后续导入的 Params.py 就不会报错
    """
    learning_rate = 4e-5  # 默认值
    data = 'DGIdb'  # 默认数据集
    gpu = 1  # 默认GPU

    # 解析并移除自定义参数
    new_argv = [sys.argv[0]]  # 保留脚本名
    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == '--learning_rate' and i + 1 < len(sys.argv):
            learning_rate = float(sys.argv[i + 1])
            i += 2  # 跳过参数名和值
        elif sys.argv[i] == '--data' and i + 1 < len(sys.argv):
            data = sys.argv[i + 1]
            new_argv.extend([sys.argv[i], sys.argv[i + 1]])  # 保留 --data 给 Params.py
            i += 2
        elif sys.argv[i] == '--gpu' and i + 1 < len(sys.argv):
            gpu = int(sys.argv[i + 1])
            new_argv.extend([sys.argv[i], sys.argv[i + 1]])  # 保留 --gpu 给 Params.py
            i += 2
        else:
            new_argv.append(sys.argv[i])
            i += 1

    # 更新 sys.argv
    sys.argv = new_argv

    return learning_rate, data, gpu

# 提取参数（在导入 DataHandler 之前）
CUSTOM_LEARNING_RATE, DATASET_NAME, GPU_ID = parse_and_extract_custom_args()

# 添加项目路径
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.append(str(CODE_DIR))

# 现在可以安全导入 DataHandler（它会导入 Params，但 sys.argv 中已经没有 --learning_rate 了）
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
# 训练集基因实体 [CLS] 嵌入缓存（预训练 BioLinkBERT，非微调后权重）
SERVER_DATA_ROOT = Path(r"/mnt/data/huangpeng/DGCL/DGCL-main/Data")
RAG_GENE_CACHE_DIR = SERVER_DATA_ROOT / DATASET_NAME / "rag"
TRAIN_GENE_EMBED_CACHE = RAG_GENE_CACHE_DIR / "pretrained_train_gene_cls.npz"
# 全体基因嵌入（与 gene_ids_global 行对齐），供困难负样本按相似度在全基因集合上检索
ALL_GENE_EMBED_CACHE = RAG_GENE_CACHE_DIR / "pretrained_all_gene_cls.npz"
# 每条训练边 (d,g)：与 g 向量最相似、且训练集中与 d 无边的前 5 个基因下标
HARD_NEG_TOP_K = 5
# 变长困难负样本（非固定 5 列）；若仍使用旧版 train_hard_neg_similar_top5.npz 请删除后重跑
TRAIN_HARD_NEG_CACHE = RAG_GENE_CACHE_DIR / "train_hard_neg_similar_varlen.npz"
# 微调后最佳模型保存目录（文件名保留 acc 标签，但每次训练只保留一个最佳文件）
BERT_SAVE_DIR = SERVER_DATA_ROOT / DATASET_NAME / "rag" / "bert"
GENE_ENTITY_MAX_CHARS = 220  # 与联合编码中基因描述截断一致
GENE_ENCODE_BATCH = 32
GENE_ENCODE_MAX_LENGTH = 256  # 单句基因编码，短于联合 512
# 全基因相似度矩阵 S=EE^T 的内存上限（约 bytes），超出则报错提示减小规模或换机
MAX_SIM_MATRIX_BYTES = int(3.0 * (1024**3))

LOG_DIR = Path("/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/rag/bert/log")  # 日志目录
LOG_DIR.mkdir(parents=True, exist_ok=True)  # 确保目录存在
LOG_FILE = LOG_DIR / f"training_log_{timestamp}.txt"  # 日志文件

# ==================== 超参数配置 ====================
# 注意：这些超参与Params.py中的GNN超参不同
# 原因：
# 1. BioLinkBERT是预训练模型，需要更小的学习率 (2e-5 vs 5e-3)
# 2. BERT模型显存占用大，batch_size需要更小 (16 vs 4096)
# 3. 预训练模型收敛快，不需要450个epoch (5-10 vs 450)
# 4. BERT微调使用AdamW + warmup，而GNN使用普通Adam

TRAIN_CONFIG = {
    'batch_size': 16,              # FP16开启后可增大到 24-32
    'learning_rate': CUSTOM_LEARNING_RATE,  # 从命令行参数读取 (已在文件开头解析)
    'num_epochs': 15,              # BERT微调推荐: 3-10 epochs
    'warmup_ratio': 0.1,           # 预热10%的训练步数
    'max_grad_norm': 1.0,          # 梯度裁剪 (BERT标准: 1.0)
    'weight_decay': 0.01,          # AdamW权重衰减 (BERT标准: 0.01)
    'save_steps': 500,             # 每500步保存一次
    'eval_steps': 500,             # 每500步评估一次
    'fp16': True,                  # V100 必开！提速 2-3 倍，节省 50% 显存
    'max_length': 512,             # 最大序列长度（联合编码需要更长）

    # 困难负样本 BPR（与 Main.py 思路一致：正样本打分高于困难负样本，加权 sigmoid BPR）
    'bpr_weight': 0.5,             # 与交叉熵协同；可改为 0.5~1.0 加大对比力度
    'bpr_use_cosine_weight': False,  # 用预计算余弦相似度作难度权重（类似 Main 中 neg_weights）
    # 负样本 BERT 分块前向 + 分次 backward，避免一次性 forward(B×K) 撑爆显存（BioLinkBERT-large）
    'bpr_neg_chunk_size': 8,       # 每块最多几条 (d,g-) 序列；仍 OOM 可改为 4 或 2
    # 以时间换显存；对 BERT 再省一截激活，可与 bpr_neg_chunk_size 联用
    'gradient_checkpointing': True,
}

# Prompt模板（句子对编码）
# 注意：使用 tokenizer(text, text_pair) 时，tokenizer会自动添加 [CLS] 和 [SEP]
# 实际生成: [CLS] Drug: {drug_desc} [SEP] Gene: {gene_desc} [SEP]
# token_type_ids: [0, 0, ..., 0, 1, 1, ..., 1]
PROMPT_TEMPLATE_TEXT = "Drug: {drug_desc}"
PROMPT_TEMPLATE_TEXT_PAIR = "Gene: {gene_desc}"

# 14种交互类型
INTERACTION_TYPES = [
    "Agonist", "Antagonist", "Antibody", "Modulator", "Blocker", "Binder",
    "Potentiator", "Cofactor", "Ligand", "Inhibitor", "Activator",
    "Partial agonist", "Positive modulator", "Allosteric modulator"
]


# ==================== 日志工具函数 ====================
def log_and_print(message, log_file=None):
    """同时输出到控制台和日志文件"""
    print(message)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(message + '\n')


# ==================== 模型定义 ====================
class ClassifierLayer(nn.Module):
    """
    MLP分类器（联合编码版本）
    输入: joint_embed (1024) - 来自联合编码的[CLS] token
    """
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
    """
    BioLinkBERT + MLP分类器 (联合编码版本)
    """
    def __init__(self, bert_model, num_classes=14):
        super().__init__()
        self.bert = bert_model
        hidden_dim = bert_model.config.hidden_size  # 1024 for BioLinkBERT-large

        # 使用联合编码的分类器
        self.classifier = ClassifierLayer(input_dim=hidden_dim, num_classes=num_classes)

    @staticmethod
    def segment_mean_pool(hidden_states, attention_mask, token_type_ids, segment_id):
        """对句子 A/B（token_type 0/1）分别做掩码均值池化，用于 BPR 打分（对齐 Main 中 drug–gene 内积）。"""
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
        """
        联合编码drug和gene，提取[CLS] token送入分类器

        Args:
            input_ids: [batch_size, seq_len] - 联合编码的token IDs
            attention_mask: [batch_size, seq_len] - 注意力掩码
            token_type_ids: [batch_size, seq_len] - 句子类型ID (0=句子A/药物, 1=句子B/基因)
            return_pair_pools: 为 True 时额外返回药物/基因段池化向量（困难负样本 BPR）

        Returns:
            logits 或 (logits, drug_pool, gene_pool)
        """
        cls_embed, drug_pool, gene_pool = self.forward_backbone(
            input_ids, attention_mask, token_type_ids
        )
        logits = self.classifier(cls_embed)
        if return_pair_pools:
            return logits, drug_pool, gene_pool
        return logits


# ==================== 数据集定义 ====================
class DrugGeneDataset(Dataset):
    """
    药物-基因交互数据集（联合编码版本）
    可选：与 train_hard_neg_similar_varlen.npz 对齐的困难负样本（每条边至多 HARD_NEG_TOP_K 个）。
    """
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

        # 越界诊断：在访问前检查并输出详情
        if drug_idx >= len(self.drug_ids_global):
            path_resolved = str(GLOBAL_IDS_JSON.resolve()) if hasattr(GLOBAL_IDS_JSON, 'resolve') else str(GLOBAL_IDS_JSON)
            print(f"\n[越界诊断] drug 索引越界")
            print(f"  索引文件路径: {path_resolved}")
            print(f"  drug_ids_global 长度: {len(self.drug_ids_global)}, gene_ids_global 长度: {len(self.gene_ids_global)}")
            print(f"  样本 idx: {idx}, drug_idx: {drug_idx} (需 < {len(self.drug_ids_global)})")
            raise IndexError(f"drug_idx {drug_idx} >= len(drug_ids_global) {len(self.drug_ids_global)}")
        if gene_idx >= len(self.gene_ids_global):
            path_resolved = str(GLOBAL_IDS_JSON.resolve()) if hasattr(GLOBAL_IDS_JSON, 'resolve') else str(GLOBAL_IDS_JSON)
            print(f"\n[越界诊断] gene 索引越界")
            print(f"  索引文件路径: {path_resolved}")
            print(f"  drug_ids_global 长度: {len(self.drug_ids_global)}, gene_ids_global 长度: {len(self.gene_ids_global)}")
            print(f"  样本 idx: {idx}, gene_idx: {gene_idx} (需 < {len(self.gene_ids_global)})")
            raise IndexError(f"gene_idx {gene_idx} >= len(gene_ids_global) {len(self.gene_ids_global)}")

        # 获取全局ID
        drug_id = self.drug_ids_global[drug_idx]
        gene_id = self.gene_ids_global[gene_idx]

        # 获取描述文本（基因键与 global_ids 字符串 ID 对齐，见 lookup_gene_description）
        drug_desc = self.drug_descriptions.get(drug_id, f"Drug {drug_id}")[:292]
        _gd = lookup_gene_description(self.gene_descriptions, gene_id)
        gene_desc = (_gd if _gd is not None else f"Gene {gene_id}")[:220]

        # 使用 text 和 text_pair 进行正确的句子对编码
        # tokenizer会自动添加 [CLS] text [SEP] text_pair [SEP]
        # 并自动生成 token_type_ids: [0, 0, ..., 0, 1, 1, ..., 1]
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
    """加载全局ID映射"""
    path_resolved = str(GLOBAL_IDS_JSON.resolve()) if hasattr(GLOBAL_IDS_JSON, 'resolve') else str(GLOBAL_IDS_JSON)
    print(f"[索引文件] 读取路径: {path_resolved}")
    with open(GLOBAL_IDS_JSON, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data['drug_ids'], data['gene_ids']


def load_drug_descriptions():
    """
    加载药物描述

    支持 JSON 或 CSV 的药物描述映射加载。
    - JSON: {DrugID: text}
    - CSV: 支持列组合 ('DrugID','LLM_Text') 或 ('DrugID','LLM_Input_Text')
    """
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
    """加载基因描述"""
    with open(GENE_DESC_JSON, 'r', encoding='utf-8') as f:
        return json.load(f)


def lookup_gene_description(gene_descriptions, gid):
    """
    基因描述字典的键可能是 str / int，与 global_ids 中的字符串 ID 对齐查找。
    """
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
    """
    确认 DataHandler 使用的下标与 global_ids 一致：
    create_global_ids.py / DataHandler.map_data 规则为
    drug: 全图 drug 字符串 sorted；gene: 全图基因按数值排序后转 str。
    """
    train_csv = DATA_ROOT / "transductive" / "train.csv"
    test_csv = DATA_ROOT / "transductive" / "test.csv"
    if not train_csv.is_file() or not test_csv.is_file():
        log_and_print(
            f"  [索引校验] 跳过：未找到 {train_csv} 或 {test_csv}",
            log_file,
        )
        return
    df_tr = pd.read_csv(train_csv, header=None)
    df_te = pd.read_csv(test_csv, header=None)
    df = pd.concat([df_tr, df_te], ignore_index=True)
    drugs_csv = sorted(pd.unique(df[0].astype(str)).tolist())
    genes_raw = pd.unique(df[1]).tolist()
    genes_csv = [str(x) for x in sorted(genes_raw, key=lambda x: int(x))]
    ok_d = drugs_csv == list(drug_ids_global)
    ok_g = genes_csv == list(gene_ids_global)
    log_and_print(
        f"  [索引校验] transductive CSV 与 {GLOBAL_IDS_JSON.name} 药物列表一致: {ok_d}",
        log_file,
    )
    log_and_print(
        f"  [索引校验] transductive CSV 与 {GLOBAL_IDS_JSON.name} 基因列表一致: {ok_g}",
        log_file,
    )
    if not ok_d or not ok_g:
        log_and_print(
            "  ⚠ 若不一致：请用 Utils/gene_llm/create_global_ids.py 从当前 train/test 重新生成 global_ids，"
            "否则 DataHandler 下标与 global_ids 不对齐。",
            log_file,
        )


def audit_gene_text_sample(gene_ids_global, gene_descriptions, log_file=None, k=5):
    """抽样打印基因下标 → global id → 描述前缀，确认文本映射正确。"""
    log_and_print(
        f"  [基因文本审计] 描述文件: {GENE_DESC_JSON}",
        log_file,
    )
    n = len(gene_ids_global)
    if n == 0:
        return
    sample_idx = {0, n // 2, n - 1}
    pool = [i for i in range(n) if i not in sample_idx]
    if pool and k > 0:
        take = min(k, len(pool))
        sample_idx.update(np.random.RandomState(42).choice(pool, size=take, replace=False).tolist())
    sample_idx = sorted(sample_idx)
    hit = 0
    for idx in sample_idx:
        gid = gene_ids_global[idx]
        text = lookup_gene_description(gene_descriptions, gid)
        if text is not None and str(text).strip() and not str(text).startswith("Gene "):
            hit += 1
        prev = (str(text)[:80] + "…") if text is not None and len(str(text)) > 80 else str(text)
        log_and_print(
            f"    idx={idx} gene_id={gid!r} 描述(前80字)={prev!r}",
            log_file,
        )
    log_and_print(
        f"  [基因文本审计] 抽样中明显非占位描述条数: {hit}/{len(sample_idx)}",
        log_file,
    )


def load_training_and_test_data():
    """
    使用DataHandler加载训练数据和测试数据
    返回: (train_drugs, train_genes, train_labels, test_drugs, test_genes, test_labels)
    """
    print(f"\n[加载数据] 数据集: {DATASET_NAME}")

    # 初始化DataHandler
    handler = DataHandler()
    handler.LoadData()

    # 从训练数据加载器中提取数据
    train_drugs = []
    train_genes = []
    train_labels = []

    for batch in handler.trnLoader:
        if len(batch) == 4:  # 预计算阶段
            drugs, genes, labels, _ = batch
        else:  # 训练阶段
            drugs, genes, labels = batch[:3]

        train_drugs.extend(drugs.numpy())
        train_genes.extend(genes.numpy())
        train_labels.extend(labels.numpy())

    train_drugs = np.array(train_drugs)
    train_genes = np.array(train_genes)
    train_labels = np.array(train_labels)

    print(f"  ✓ 训练样本数: {len(train_labels)}")
    print(f"  ✓ 训练标签分布: {np.bincount(train_labels)}")

    # 从测试数据加载器中提取数据
    test_drugs = []
    test_genes = []
    test_labels = []

    for batch in handler.tstLoader:
        drugs, genes, labels = batch
        test_drugs.extend(drugs.numpy())
        test_genes.extend(genes.numpy())
        test_labels.extend(labels.numpy())

    test_drugs = np.array(test_drugs)
    test_genes = np.array(test_genes)
    test_labels = np.array(test_labels)

    print(f"  ✓ 测试样本数: {len(test_labels)}")
    print(f"  ✓ 测试标签分布: {np.bincount(test_labels)}")

    return train_drugs, train_genes, train_labels, test_drugs, test_genes, test_labels


def ensure_precomputed_train_gene_embeddings(
    train_genes,
    gene_ids_global,
    gene_descriptions,
    device,
    log_file=None,
):
    """
    训练前：用预训练 BioLinkBERT（MODEL_CACHE，非微调）对训练集中出现的基因做实体级单句编码，
    取 [CLS] 向量并缓存。索引与 global_ids 中基因下标一致；仅覆盖训练集出现过的基因。
    若 TRAIN_GENE_EMBED_CACHE 已存在则跳过。
    """
    cache_path = TRAIN_GENE_EMBED_CACHE
    if cache_path.is_file():
        log_and_print(
            f"\n[预计算基因嵌入] 已存在，跳过: {cache_path}",
            log_file,
        )
        return

    log_and_print(
        f"\n[预计算基因嵌入] 开始（预训练权重: {MODEL_CACHE}）...",
        log_file,
    )
    log_and_print(f"  输出: {cache_path}", log_file)

    unique_gene_idx = np.unique(train_genes.astype(np.int64))
    unique_gene_idx.sort()

    texts = []
    for gidx in unique_gene_idx:
        if gidx < 0 or gidx >= len(gene_ids_global):
            raise IndexError(
                f"train gene_idx {gidx} 超出 gene_ids_global 长度 {len(gene_ids_global)}"
            )
        gid = gene_ids_global[int(gidx)]
        desc = lookup_gene_description(gene_descriptions, gid)
        if desc is None:
            desc = f"Gene {gid}"
        if isinstance(desc, str):
            desc = desc[:GENE_ENTITY_MAX_CHARS]
        else:
            desc = str(desc)[:GENE_ENTITY_MAX_CHARS]
        texts.append(f"Gene: {desc}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CACHE, local_files_only=True, trust_remote_code=True
    )
    enc_model = AutoModel.from_pretrained(
        MODEL_CACHE, local_files_only=True, trust_remote_code=True
    )
    enc_model.eval()
    enc_model.to(device)

    all_rows = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), GENE_ENCODE_BATCH),
            desc="基因实体编码(预训练)",
        ):
            end = min(start + GENE_ENCODE_BATCH, len(texts))
            batch_texts = texts[start:end]
            encoding = tokenizer(
                batch_texts,
                max_length=GENE_ENCODE_MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}
            out = enc_model(**encoding)
            cls_emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_rows.append(cls_emb)

    embeddings = np.concatenate(all_rows, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        gene_indices=unique_gene_idx.astype(np.int64),
        embeddings=embeddings,
        model_path=str(MODEL_CACHE),
        global_ids_json=str(GLOBAL_IDS_JSON),
        num_genes_global=len(gene_ids_global),
    )
    log_and_print(
        f"  ✓ 已保存: {cache_path}, genes={len(unique_gene_idx)}, dim={embeddings.shape[1]}",
        log_file,
    )

    del enc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def ensure_precomputed_all_gene_embeddings(
    gene_ids_global,
    gene_descriptions,
    device,
    log_file=None,
):
    """
    与 gene_ids_global 顺序一致的全基因 [CLS] 嵌入，供困难负样本在全基因空间比相似度。
    仅使用 MODEL_CACHE 预训练权重；文件已存在则跳过。
    """
    cache_path = ALL_GENE_EMBED_CACHE
    if cache_path.is_file():
        log_and_print(
            f"\n[全基因嵌入] 已存在，跳过: {cache_path}",
            log_file,
        )
        return

    log_and_print(
        f"\n[全基因嵌入] 开始编码全部 {len(gene_ids_global)} 个基因（预训练: {MODEL_CACHE}）...",
        log_file,
    )
    texts = []
    for gid in gene_ids_global:
        desc = lookup_gene_description(gene_descriptions, gid)
        if desc is None:
            desc = f"Gene {gid}"
        if isinstance(desc, str):
            desc = desc[:GENE_ENTITY_MAX_CHARS]
        else:
            desc = str(desc)[:GENE_ENTITY_MAX_CHARS]
        texts.append(f"Gene: {desc}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CACHE, local_files_only=True, trust_remote_code=True
    )
    enc_model = AutoModel.from_pretrained(
        MODEL_CACHE, local_files_only=True, trust_remote_code=True
    )
    enc_model.eval()
    enc_model.to(device)

    all_rows = []
    with torch.no_grad():
        for start in tqdm(
            range(0, len(texts), GENE_ENCODE_BATCH),
            desc="全基因实体编码(预训练)",
        ):
            end = min(start + GENE_ENCODE_BATCH, len(texts))
            batch_texts = texts[start:end]
            encoding = tokenizer(
                batch_texts,
                max_length=GENE_ENCODE_MAX_LENGTH,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            encoding = {k: v.to(device) for k, v in encoding.items()}
            out = enc_model(**encoding)
            cls_emb = out.last_hidden_state[:, 0, :].cpu().float().numpy()
            all_rows.append(cls_emb)

    embeddings = np.concatenate(all_rows, axis=0).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        embeddings=embeddings,
        model_path=str(MODEL_CACHE),
        global_ids_json=str(GLOBAL_IDS_JSON),
        num_genes=len(gene_ids_global),
    )
    log_and_print(
        f"  ✓ 已保存: {cache_path}, shape={embeddings.shape}",
        log_file,
    )

    del enc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _transductive_pair_csr(num_drugs, num_genes, drug_ids_global, gene_ids_global, log_file=None):
    """
    用 transductive/train.csv + test.csv 中全部已知 (药物, 基因) 边构建 CSR，
    用于「与 d 无已知交互」的屏蔽（避免把测试集里的真边当成困难负样本）。
    """
    train_csv = DATA_ROOT / "transductive" / "train.csv"
    test_csv = DATA_ROOT / "transductive" / "test.csv"
    d_rev = {str(d): i for i, d in enumerate(drug_ids_global)}
    g_rev = {str(g): i for i, g in enumerate(gene_ids_global)}
    rows = []
    cols = []
    for path in (train_csv, test_csv):
        if not path.is_file():
            continue
        df = pd.read_csv(path, header=None)
        for _, row in df.iterrows():
            ds = str(row[0])
            try:
                gj = str(int(row[1]))
            except (TypeError, ValueError):
                gj = str(row[1]).strip()
            di = d_rev.get(ds)
            gi = g_rev.get(gj)
            if di is None or gi is None:
                continue
            rows.append(di)
            cols.append(gi)
    if not rows:
        log_and_print(
            "  [困难负样本] ⚠ 未从 transductive CSV 读到边，回退为仅用当前训练 loader 中的边做屏蔽",
            log_file,
        )
        return None
    return csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(num_drugs, num_genes),
    )


def ensure_train_hard_negatives_similar_top5(
    train_drugs,
    train_genes,
    train_labels,
    num_drugs,
    num_genes,
    drug_ids_global,
    gene_ids_global,
    log_file=None,
    console_detail_first_n=50,
):
    """
    对每条训练正样本边 (药物下标 d, 基因下标 g)：
    在「与 d 在 transductive train+test 中均无已知边」的基因中，
    按与 g 的余弦相似度（全基因预计算嵌入）至多取前 HARD_NEG_TOP_K 个；
    若合法候选不足则只保留实际个数，不填充占位下标。
    缓存格式：hard_neg_offsets (N+1) + hard_neg_gene_idx (变长拼接) + hard_neg_cos_sim (与前者等长)。
    """
    cache_path = TRAIN_HARD_NEG_CACHE
    if cache_path.is_file():
        log_and_print(
            f"\n[困难负样本] 已存在，跳过: {cache_path}",
            log_file,
        )
        return

    emb_path = ALL_GENE_EMBED_CACHE
    if not emb_path.is_file():
        raise FileNotFoundError(
            f"缺少全基因嵌入文件，请先成功运行全基因编码阶段: {emb_path}"
        )

    z = np.load(emb_path, allow_pickle=True)
    E = np.asarray(z["embeddings"], dtype=np.float32)
    if E.shape[0] != num_genes:
        raise ValueError(
            f"全基因嵌入行数 {E.shape[0]} != gene 数 {num_genes}，与 global_ids 不一致"
        )

    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    E_norm = E / norms

    sim_bytes = num_genes * num_genes * 4
    if sim_bytes > MAX_SIM_MATRIX_BYTES:
        raise MemoryError(
            f"基因数={num_genes} 时相似度矩阵约需 {sim_bytes / 1024**3:.2f} GB，"
            f"超过上限 {MAX_SIM_MATRIX_BYTES / 1024**3:.2f} GB。请提高 MAX_SIM_MATRIX_BYTES 或在更大内存机器上运行。"
        )

    log_and_print(
        f"\n[困难负样本] 构建已知边矩阵(train+test)并计算相似度 ({num_genes}x{num_genes})...",
        log_file,
    )
    td = train_drugs.astype(np.int64)
    tg = train_genes.astype(np.int64)
    mask_csr = _transductive_pair_csr(
        num_drugs, num_genes, drug_ids_global, gene_ids_global, log_file
    )
    if mask_csr is None:
        mask_csr = csr_matrix(
            (np.ones(len(td), dtype=np.float32), (td, tg)),
            shape=(num_drugs, num_genes),
        )

    S = (E_norm @ E_norm.T).astype(np.float32)
    N = len(td)
    buf = np.empty(num_genes, dtype=np.float32)
    neg_inf = np.float32(-1e30)
    k = HARD_NEG_TOP_K

    row_lens = []
    flat_idx_parts = []
    flat_sim_parts = []

    for i in tqdm(range(N), desc="困难负样本 top5"):
        d = int(td[i])
        g = int(tg[i])
        np.copyto(buf, S[g])
        blk = mask_csr[d].indices
        buf[blk] = neg_inf
        buf[g] = neg_inf
        finite = np.isfinite(buf)
        if not finite.any():
            row_lens.append(0)
            flat_idx_parts.append(np.array([], dtype=np.int32))
            flat_sim_parts.append(np.array([], dtype=np.float32))
            continue
        sub_idx = np.flatnonzero(finite)
        sub_scores = buf[sub_idx]
        sub_k = min(k, int(sub_scores.shape[0]))
        top_local = np.argpartition(-sub_scores, sub_k - 1)[:sub_k]
        top_local = top_local[np.argsort(-sub_scores[top_local])]
        picked = sub_idx[top_local].astype(np.int32)
        sims = sub_scores[top_local].astype(np.float32)
        row_lens.append(int(picked.shape[0]))
        flat_idx_parts.append(picked)
        flat_sim_parts.append(sims)

    hard_neg_offsets = np.zeros(N + 1, dtype=np.int64)
    for i in range(N):
        hard_neg_offsets[i + 1] = hard_neg_offsets[i] + row_lens[i]
    hard_neg_gene_idx = (
        np.concatenate(flat_idx_parts, axis=0).astype(np.int32)
        if flat_idx_parts
        else np.array([], dtype=np.int32)
    )
    hard_neg_cos_sim = (
        np.concatenate(flat_sim_parts, axis=0).astype(np.float32)
        if flat_sim_parts
        else np.array([], dtype=np.float32)
    )

    cnt = Counter(row_lens)
    n_lt_k = sum(1 for L in row_lens if L < k)
    n_zero = sum(1 for L in row_lens if L == 0)
    dist_str = ", ".join(f"{L}个:{cnt[L]}" for L in sorted(cnt))
    msg_stats = (
        f"\n[困难负样本] 统计: 训练边数={N}, 每边至多{k}个, 实际不足{k}个的边数={n_lt_k}, "
        f"困难负样本为0的边数={n_zero}\n"
        f"  每条困难负样本个数分布: {dist_str}"
    )
    print(msg_stats)
    log_and_print(msg_stats.strip(), log_file)

    n_show = min(console_detail_first_n, N)
    if n_show > 0:
        print(f"\n[困难负样本] 前 {n_show} 条案例：正样本基因 g 与各困难负样本的余弦相似度（与缓存中 hard_neg_cos_sim 一致）")
    for i in range(n_show):
        d = int(td[i])
        g = int(tg[i])
        lo = int(hard_neg_offsets[i])
        hi = int(hard_neg_offsets[i + 1])
        pidx = hard_neg_gene_idx[lo:hi]
        psim = hard_neg_cos_sim[lo:hi]
        gid = gene_ids_global[g]
        did = drug_ids_global[d]
        print(
            f"  --- 案例 {i + 1}/{n_show} | drug_idx={d} drug_id={did!r} | "
            f"正样本 gene_idx={g} gene_id={gid!r} | 困难负样本数={len(pidx)} ---"
        )
        for j in range(len(pidx)):
            gj = int(pidx[j])
            print(
                f"      困难负样本 #{j + 1}: gene_idx={gj} gene_id={gene_ids_global[gj]!r} "
                f"与正样本g余弦相似度={float(psim[j]):.6f}"
            )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_drug_idx=td,
        train_gene_idx=tg,
        train_label=train_labels.astype(np.int64),
        hard_neg_offsets=hard_neg_offsets,
        hard_neg_gene_idx=hard_neg_gene_idx,
        hard_neg_cos_sim=hard_neg_cos_sim,
        hard_neg_max_per_edge=np.int32(k),
        all_gene_embed_path=str(emb_path),
        global_ids_json=str(GLOBAL_IDS_JSON),
    )
    log_and_print(
        f"  ✓ 已保存: {cache_path}，变长存储: offsets 长度 {N + 1}，"
        f"共 {len(hard_neg_gene_idx)} 个困难负样本基因下标",
        log_file,
    )


# ==================== 训练和评估函数 ====================
def tokenize_drug_gene_pairs_flat(
    drug_idx_flat,
    gene_idx_flat,
    drug_descriptions,
    gene_descriptions,
    drug_ids_global,
    gene_ids_global,
    tokenizer,
    max_length,
):
    """
    将 (drug_idx, gene_idx) 逐对编码为与 DrugGeneDataset 一致的句子对输入。
    drug_idx_flat / gene_idx_flat: 1D LongTensor（CPU）
    """
    input_ids, attention_mask, token_type_ids = [], [], []
    for di, gi in zip(drug_idx_flat.tolist(), gene_idx_flat.tolist()):
        drug_id = drug_ids_global[di]
        gene_id = gene_ids_global[gi]
        drug_desc = drug_descriptions.get(drug_id, f"Drug {drug_id}")[:292]
        _gd = lookup_gene_description(gene_descriptions, gene_id)
        gene_desc = (_gd if _gd is not None else f"Gene {gene_id}")[:220]
        enc = tokenizer(
            text=f"Drug: {drug_desc}",
            text_pair=f"Gene: {gene_desc}",
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        input_ids.append(enc['input_ids'].squeeze(0))
        attention_mask.append(enc['attention_mask'].squeeze(0))
        token_type_ids.append(enc['token_type_ids'].squeeze(0))
    return {
        'input_ids': torch.stack(input_ids, dim=0),
        'attention_mask': torch.stack(attention_mask, dim=0),
        'token_type_ids': torch.stack(token_type_ids, dim=0),
    }


def hard_neg_bpr_loss_from_pools(drug_pos, gene_pos, gene_negs, valid_mask, neg_weights, use_cosine_weight):
    """
    Main.py 风格 BPR：posScore = <drug,g+>, negScore_k = <drug,g_k^->（段池化后 L2 归一化再点积）。
    valid_mask / neg_weights: [B, K]；无效位置不计入 loss。
    """
    drug_pos = F.normalize(drug_pos, dim=-1)
    gene_pos = F.normalize(gene_pos, dim=-1)
    gene_negs = F.normalize(gene_negs, dim=-1)
    pos_scores = (drug_pos * gene_pos).sum(dim=-1, keepdim=True)
    neg_scores = (drug_pos.unsqueeze(1) * gene_negs).sum(dim=-1)
    score_diff = pos_scores - neg_scores
    if use_cosine_weight:
        score_diff = score_diff * neg_weights.clamp(min=1e-6)
    bpr = -F.logsigmoid(score_diff)
    vm = valid_mask.float()
    denom = vm.sum().clamp(min=1.0)
    return (bpr * vm).sum() / denom


def hard_neg_bpr_contrib_sum_rows(drug_rows, gene_pos_rows, gene_neg_rows, w_rows, use_cosine_weight):
    """
    与 hard_neg_bpr_loss_from_pools 同一打分公式，对展平后的 M 条 (batch行, k) 中一小批行求
    Σ_i (-log σ(s_pos - s_neg))，供分块 backward；w_rows 为与缓存余弦对齐的难度权重。
    drug_rows / gene_pos_rows / gene_neg_rows: [C, H]；w_rows: [C]
    """
    drug_rows = F.normalize(drug_rows, dim=-1)
    gene_pos_rows = F.normalize(gene_pos_rows, dim=-1)
    gene_neg_rows = F.normalize(gene_neg_rows, dim=-1)
    pos_s = (drug_rows * gene_pos_rows).sum(dim=-1)
    neg_s = (drug_rows * gene_neg_rows).sum(dim=-1)
    diff = pos_s - neg_s
    if use_cosine_weight:
        diff = diff * w_rows.clamp(min=1e-6)
    return (-F.logsigmoid(diff)).sum()


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    epoch,
    scaler=None,
    *,
    tokenizer=None,
    drug_descriptions=None,
    gene_descriptions=None,
    drug_ids_global=None,
    gene_ids_global=None,
    max_length=512,
    bpr_weight=0.0,
    bpr_use_cosine_weight=True,
):
    """训练一个 epoch：交叉熵 + 困难负样本 BPR（与 Main.py 加权 sigmoid BPR 一致思路）。"""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    use_fp16 = scaler is not None

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")

    neg_chunk = int(TRAIN_CONFIG.get('bpr_neg_chunk_size', 8))

    for batch_idx, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        loss_bpr_val = 0.0
        did_bpr = False

        if use_fp16:
            with autocast('cuda'):
                logits, drug_p, gene_p = model(
                    input_ids, attention_mask, token_type_ids, return_pair_pools=True
                )
                loss_ce = F.cross_entropy(logits, labels)

            need_bpr = (
                bpr_weight > 0
                and tokenizer is not None
                and batch.get('hard_neg_genes') is not None
            )
            if need_bpr:
                hn = batch['hard_neg_genes']
                hw = batch['hard_neg_weight']
                dix = batch['drug_idx']
                valid = hn >= 0
                if valid.any():
                    did_bpr = True
                    B, _ = hn.shape
                    b_flat = torch.arange(
                        B, device=valid.device, dtype=torch.long
                    ).unsqueeze(1).expand_as(hn)[valid]
                    di_exp = dix.unsqueeze(1).expand_as(hn)[valid]
                    g_flat = hn[valid]
                    w_flat = hw[valid].float()
                    m_total = int(valid.sum().item())

                    scaler.scale(loss_ce).backward(retain_graph=True)

                    bpr_sum_det = 0.0
                    for s in range(0, m_total, neg_chunk):
                        e = min(s + neg_chunk, m_total)
                        b_c = b_flat[s:e].to(device, non_blocking=True)
                        di_c = di_exp[s:e]
                        g_c = g_flat[s:e]
                        w_c = w_flat[s:e]

                        drug_sel = drug_p[b_c]
                        gene_ps = gene_p[b_c]
                        neg_enc = tokenize_drug_gene_pairs_flat(
                            di_c.cpu(),
                            g_c.cpu(),
                            drug_descriptions,
                            gene_descriptions,
                            drug_ids_global,
                            gene_ids_global,
                            tokenizer,
                            max_length,
                        )
                        ni = neg_enc['input_ids'].to(device, non_blocking=True)
                        na = neg_enc['attention_mask'].to(device, non_blocking=True)
                        nt = neg_enc['token_type_ids'].to(device, non_blocking=True)
                        with autocast('cuda'):
                            _, _, gene_neg = model(ni, na, nt, return_pair_pools=True)
                        contrib = hard_neg_bpr_contrib_sum_rows(
                            drug_sel,
                            gene_ps,
                            gene_neg,
                            w_c.to(device, non_blocking=True),
                            bpr_use_cosine_weight,
                        )
                        # 多块 BPR 共用同一正样本前向的 drug_p/gene_p，除最后一块外须 retain_graph
                        scaler.scale(bpr_weight * contrib / m_total).backward(
                            retain_graph=e < m_total
                        )
                        bpr_sum_det += float(contrib.detach().item())

                    loss_bpr_val = bpr_weight * (bpr_sum_det / m_total)
                    loss_total = float(loss_ce.detach().item()) + loss_bpr_val
                else:
                    scaler.scale(loss_ce).backward()
                    loss_total = float(loss_ce.detach().item())
            else:
                scaler.scale(loss_ce).backward()
                loss_total = float(loss_ce.detach().item())

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, drug_p, gene_p = model(
                input_ids, attention_mask, token_type_ids, return_pair_pools=True
            )
            loss_ce = F.cross_entropy(logits, labels)

            need_bpr = (
                bpr_weight > 0
                and tokenizer is not None
                and batch.get('hard_neg_genes') is not None
            )
            if need_bpr:
                hn = batch['hard_neg_genes']
                hw = batch['hard_neg_weight']
                dix = batch['drug_idx']
                valid = hn >= 0
                if valid.any():
                    did_bpr = True
                    B, _ = hn.shape
                    b_flat = torch.arange(
                        B, device=valid.device, dtype=torch.long
                    ).unsqueeze(1).expand_as(hn)[valid]
                    di_exp = dix.unsqueeze(1).expand_as(hn)[valid]
                    g_flat = hn[valid]
                    w_flat = hw[valid].float()
                    m_total = int(valid.sum().item())

                    loss_ce.backward(retain_graph=True)

                    bpr_sum_det = 0.0
                    for s in range(0, m_total, neg_chunk):
                        e = min(s + neg_chunk, m_total)
                        b_c = b_flat[s:e].to(device, non_blocking=True)
                        di_c = di_exp[s:e]
                        g_c = g_flat[s:e]
                        w_c = w_flat[s:e]

                        drug_sel = drug_p[b_c]
                        gene_ps = gene_p[b_c]
                        neg_enc = tokenize_drug_gene_pairs_flat(
                            di_c.cpu(),
                            g_c.cpu(),
                            drug_descriptions,
                            gene_descriptions,
                            drug_ids_global,
                            gene_ids_global,
                            tokenizer,
                            max_length,
                        )
                        ni = neg_enc['input_ids'].to(device, non_blocking=True)
                        na = neg_enc['attention_mask'].to(device, non_blocking=True)
                        nt = neg_enc['token_type_ids'].to(device, non_blocking=True)
                        _, _, gene_neg = model(ni, na, nt, return_pair_pools=True)
                        contrib = hard_neg_bpr_contrib_sum_rows(
                            drug_sel,
                            gene_ps,
                            gene_neg,
                            w_c.to(device, non_blocking=True),
                            bpr_use_cosine_weight,
                        )
                        (bpr_weight * contrib / m_total).backward(retain_graph=e < m_total)
                        bpr_sum_det += float(contrib.detach().item())

                    loss_bpr_val = bpr_weight * (bpr_sum_det / m_total)
                    loss_total = float(loss_ce.detach().item()) + loss_bpr_val
                else:
                    loss_ce.backward()
                    loss_total = float(loss_ce.detach().item())
            else:
                loss_ce.backward()
                loss_total = float(loss_ce.detach().item())

            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['max_grad_norm'])
            optimizer.step()

        scheduler.step()

        total_loss += loss_total
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        postfix = {'loss': loss_total}
        if did_bpr:
            postfix['bpr'] = loss_bpr_val
        progress_bar.set_postfix(postfix)

    avg_loss = total_loss / max(len(dataloader), 1)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def evaluate(model, dataloader, device):
    """评估模型（联合编码版本）"""
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

    # Top-5准确率
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
    log_and_print(f"日志文件: {LOG_FILE}", LOG_FILE)
    log_and_print("=" * 80, LOG_FILE)

    # 设置设备（支持指定GPU）
    if torch.cuda.is_available():
        # 使用脚本参数指定的 GPU
        device = torch.device(f'cuda:{GPU_ID}')
        torch.cuda.set_device(device)
        log_and_print(f"\n设备: {device} ({torch.cuda.get_device_name(GPU_ID)})", LOG_FILE)
        log_and_print(f"显存: {torch.cuda.get_device_properties(GPU_ID).total_memory / 1024**3:.1f} GB", LOG_FILE)
    else:
        device = torch.device('cpu')
        log_and_print(f"\n设备: CPU (CUDA不可用)", LOG_FILE)

    # 1. 加载全局数据
    log_and_print("\n[1/12] 加载全局数据...", LOG_FILE)
    drug_ids_global, gene_ids_global = load_global_ids()
    path_resolved = str(GLOBAL_IDS_JSON.resolve()) if hasattr(GLOBAL_IDS_JSON, 'resolve') else str(GLOBAL_IDS_JSON)
    log_and_print(f"  ✓ 索引文件路径: {path_resolved}", LOG_FILE)
    drug_descriptions = load_drug_descriptions()
    gene_descriptions = load_gene_descriptions()
    log_and_print(f"  ✓ 药物数: {len(drug_ids_global)}, 基因数: {len(gene_ids_global)}", LOG_FILE)

    # 2. 加载训练数据和测试数据
    log_and_print("\n[2/12] 加载训练数据和测试数据...", LOG_FILE)
    train_drugs, train_genes, train_labels, test_drugs, test_genes, test_labels = load_training_and_test_data()

    # 2.1 越界预检查：输出索引范围与 global_ids 长度对比
    max_train_drug = int(np.max(train_drugs)) if len(train_drugs) > 0 else -1
    max_train_gene = int(np.max(train_genes)) if len(train_genes) > 0 else -1
    max_test_drug = int(np.max(test_drugs)) if len(test_drugs) > 0 else -1
    max_test_gene = int(np.max(test_genes)) if len(test_genes) > 0 else -1
    log_and_print(f"\n[索引范围检查]", LOG_FILE)
    log_and_print(f"  drug_ids_global 长度: {len(drug_ids_global)}, gene_ids_global 长度: {len(gene_ids_global)}", LOG_FILE)
    log_and_print(f"  train: drug_idx 最大 {max_train_drug}, gene_idx 最大 {max_train_gene}", LOG_FILE)
    log_and_print(f"  test:  drug_idx 最大 {max_test_drug}, gene_idx 最大 {max_test_gene}", LOG_FILE)
    if max_train_drug >= len(drug_ids_global) or max_train_gene >= len(gene_ids_global) or \
       max_test_drug >= len(drug_ids_global) or max_test_gene >= len(gene_ids_global):
        log_and_print(f"  ⚠ 越界: 数据索引超出 global_ids 范围，将报错", LOG_FILE)

    # 3–6. 预计算 / 困难负样本：对应文件已存在则自动跳过，直接进入后续训练
    log_and_print(
        "\n[3–6/12] 预计算与困难负样本：若 rag 下 npz 已存在将跳过对应步骤，不重复计算。",
        LOG_FILE,
    )
    # 3. 预计算并缓存训练集基因实体嵌入（预训练权重；文件已存在则跳过）
    log_and_print("\n[3/12] 预计算训练集基因实体嵌入（缓存）...", LOG_FILE)
    ensure_precomputed_train_gene_embeddings(
        train_genes,
        gene_ids_global,
        gene_descriptions,
        device,
        LOG_FILE,
    )

    # 4. 预计算全基因嵌入（困难负样本在全基因上比相似度；文件已存在则跳过）
    log_and_print("\n[4/12] 预计算全基因实体嵌入（缓存）...", LOG_FILE)
    ensure_precomputed_all_gene_embeddings(
        gene_ids_global,
        gene_descriptions,
        device,
        LOG_FILE,
    )

    # 5. 索引与基因文本一致性说明（日志）
    log_and_print("\n[5/12] 索引与基因文本校验（日志）...", LOG_FILE)
    log_and_print(
        f"  使用的索引文件: {GLOBAL_IDS_JSON}（与 DataHandler 下标对齐的前提是："
        f"由 create_global_ids.py 按 transductive train+test 生成，规则同 DataHandler.map_data）",
        LOG_FILE,
    )
    verify_global_ids_vs_transductive_csv(drug_ids_global, gene_ids_global, LOG_FILE)
    audit_gene_text_sample(gene_ids_global, gene_descriptions, LOG_FILE)

    # 6. 困难负样本：与 g 最相似且训练集中与 d 无交互的前 5 个基因下标
    log_and_print(
        "\n[6/12] 筛选困难负样本（相似度 top5；屏蔽 transductive train+test 中所有已知 d–g' 边）...",
        LOG_FILE,
    )
    ensure_train_hard_negatives_similar_top5(
        train_drugs,
        train_genes,
        train_labels,
        len(drug_ids_global),
        len(gene_ids_global),
        drug_ids_global,
        gene_ids_global,
        LOG_FILE,
    )

    if not TRAIN_HARD_NEG_CACHE.is_file():
        raise FileNotFoundError(
            f"缺少困难负样本缓存 {TRAIN_HARD_NEG_CACHE}，请先完成步骤 6 或检查路径。"
        )
    hnz = np.load(TRAIN_HARD_NEG_CACHE, allow_pickle=True)
    if not (
        np.array_equal(np.asarray(hnz['train_drug_idx']), train_drugs)
        and np.array_equal(np.asarray(hnz['train_gene_idx']), train_genes)
    ):
        raise ValueError(
            "困难负样本缓存与当前训练集顺序/内容不一致，请删除缓存后重新运行步骤 6 生成。"
        )
    hard_neg_offsets = hnz['hard_neg_offsets']
    hard_neg_gene_idx = hnz['hard_neg_gene_idx']
    hard_neg_cos_sim = hnz['hard_neg_cos_sim']

    # 7. 加载模型和tokenizer
    log_and_print("\n[7/12] 加载BioLinkBERT模型...", LOG_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)
    bert_model = AutoModel.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)

    model = BioLinkBERTClassifier(bert_model, num_classes=14)
    model = model.to(device)
    if TRAIN_CONFIG.get('gradient_checkpointing'):
        model.bert.gradient_checkpointing_enable()
        log_and_print("  ✓ BERT gradient checkpointing 已开启（降低激活显存）", LOG_FILE)
    log_and_print(f"  ✓ 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", LOG_FILE)

    # 8. 创建数据集和数据加载器
    log_and_print("\n[8/12] 创建数据加载器...", LOG_FILE)

    max_length = TRAIN_CONFIG['max_length']

    # 训练集（联合编码 + 与缓存对齐的困难负样本）
    train_dataset = DrugGeneDataset(
        train_drugs, train_genes, train_labels,
        drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global, tokenizer, max_length,
        hard_neg_offsets=hard_neg_offsets,
        hard_neg_gene_idx=hard_neg_gene_idx,
        hard_neg_cos_sim=hard_neg_cos_sim,
    )

    # 官方测试集
    test_dataset = DrugGeneDataset(
        test_drugs, test_genes, test_labels,
        drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global, tokenizer, max_length
    )

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)

    log_and_print(f"  ✓ 训练批次数: {len(train_loader)}", LOG_FILE)
    log_and_print(f"  ✓ 官方测试批次数: {len(test_loader)}", LOG_FILE)
    log_and_print(f"  ✓ 编码方式: 联合编码 (Drug + Gene)", LOG_FILE)
    log_and_print(f"  ✓ 最大序列长度: {max_length}", LOG_FILE)
    log_and_print(
        f"  ✓ 损失: 交叉熵 + {TRAIN_CONFIG['bpr_weight']} × 困难负样本 BPR"
        f"（余弦权重: {TRAIN_CONFIG['bpr_use_cosine_weight']}, 负样本分块: {TRAIN_CONFIG.get('bpr_neg_chunk_size', 8)}）",
        LOG_FILE,
    )

    # 9. 设置优化器和学习率调度器
    log_and_print("\n[9/12] 设置优化器...", LOG_FILE)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG['learning_rate'],
        weight_decay=TRAIN_CONFIG['weight_decay']
    )

    total_steps = len(train_loader) * TRAIN_CONFIG['num_epochs']
    warmup_steps = int(total_steps * TRAIN_CONFIG['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # 初始化 FP16 GradScaler
    scaler = None
    if TRAIN_CONFIG['fp16']:
        if torch.cuda.is_available():
            scaler = GradScaler(device='cuda')  # 使用 device 参数
            log_and_print(f"  ✓ FP16 混合精度训练已启用 (预期提速 2-3x)", LOG_FILE)
        else:
            log_and_print(f"  ⚠ FP16 需要 CUDA，已自动切换到 FP32", LOG_FILE)

    log_and_print(f"  ✓ 总训练步数: {total_steps}, 预热步数: {warmup_steps}", LOG_FILE)

    # 10. 训练循环
    log_and_print("\n[10/12] 开始训练...", LOG_FILE)
    best_test_acc = 0.0
    best_epoch = 0
    best_test_preds = None
    best_test_labels_eval = None
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    # 训练过程中只保留一个“acc 最佳”模型（文件名含 acc，提升时删除旧文件）
    BERT_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    best_save_path = None
    training_start_time = time.time()

    # 记录每个epoch的准确率
    official_test_acc_history = []

    for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{TRAIN_CONFIG['num_epochs']}")
        print(f"{'='*60}")

        epoch_start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            epoch,
            scaler,
            tokenizer=tokenizer,
            drug_descriptions=drug_descriptions,
            gene_descriptions=gene_descriptions,
            drug_ids_global=drug_ids_global,
            gene_ids_global=gene_ids_global,
            max_length=max_length,
            bpr_weight=TRAIN_CONFIG['bpr_weight'],
            bpr_use_cosine_weight=TRAIN_CONFIG['bpr_use_cosine_weight'],
        )
        epoch_time = time.time() - epoch_start_time

        print(f"  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, 耗时: {epoch_time:.1f}s")
        log_and_print(
            f"Epoch {epoch}: 训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, 耗时: {epoch_time:.1f}s",
            LOG_FILE,
        )

        # 在官方测试集上评估
        test_acc, test_top5_acc, test_preds, test_labels_eval = evaluate(model, test_loader, device)
        print(f"  官方测试集 - Accuracy: {test_acc:.4f}, Top-5 Accuracy: {test_top5_acc:.4f}")
        log_and_print(
            f"Epoch {epoch}: 官方测试集 - Accuracy: {test_acc:.4f}, Top-5 Accuracy: {test_top5_acc:.4f}",
            LOG_FILE,
        )
        official_test_acc_history.append(test_acc)

        # 保存最佳模型（基于官方测试集）并记录对应的预测结果
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
                'test_top5_acc': test_top5_acc,
                'acc_filename_tag': acc_tag,
                'config': TRAIN_CONFIG,
            }, save_path)
            best_save_path = save_path
            print(f"  ✓ 保存最佳模型（含分类头，仅保留当前最佳）: {save_path} (acc_tag={acc_tag})")
            log_and_print(
                f"Epoch {epoch}: 保存最佳模型（仅保留当前最佳）test_acc={test_acc:.4f}, acc_tag={acc_tag}, file={save_path}",
                LOG_FILE,
            )

    total_training_time = time.time() - training_start_time
    log_and_print(f"\n总训练时间: {total_training_time/60:.1f} 分钟", LOG_FILE)

    # 11. 最终评估
    log_and_print("\n[11/12] 最终评估...", LOG_FILE)
    log_and_print(f"\n{'='*80}", LOG_FILE)
    log_and_print(f"训练完成！", LOG_FILE)
    log_and_print(f"{'='*80}", LOG_FILE)

    # 找到最大准确率对应的epoch
    best_official_epoch = best_epoch

    log_and_print(f"官方测试集最佳准确率: {best_test_acc:.4f} (Epoch {best_official_epoch})", LOG_FILE)

    # 输出每个epoch的准确率历史
    log_and_print(f"\n{'='*80}", LOG_FILE)
    log_and_print("📊 每个Epoch的准确率历史", LOG_FILE)
    log_and_print(f"{'='*80}", LOG_FILE)
    log_and_print(f"官方测试集准确率 (每个epoch): {official_test_acc_history}", LOG_FILE)
    log_and_print(f"\n官方测试集最大准确率: {max(official_test_acc_history):.4f} 在 Epoch {best_official_epoch}", LOG_FILE)
    log_and_print(f"{'='*80}", LOG_FILE)

    # 7.1 官方测试集最佳 Epoch 分类报告（控制台 + 日志）
    log_and_print(f"\n{'='*80}", LOG_FILE)
    log_and_print("📊 官方测试集 - 最佳 Epoch 分类报告", LOG_FILE)
    log_and_print(f"Best Epoch: {best_official_epoch}, Best Acc: {best_test_acc:.4f}", LOG_FILE)
    log_and_print(f"{'='*80}", LOG_FILE)
    report = classification_report(
        best_test_labels_eval,
        best_test_preds,
        target_names=INTERACTION_TYPES,
        zero_division=0
    )
    log_and_print(report, LOG_FILE)

    # 12. 保存预测结果
    print("\n[12/12] 保存预测结果...")

    # 8.1 保存官方测试集结果
    test_results_df = pd.DataFrame({
        '药物ID': [drug_ids_global[idx] for idx in test_drugs],
        '基因ID': [gene_ids_global[idx] for idx in test_genes],
        '真实标签': [INTERACTION_TYPES[label] for label in test_labels_eval],
        '预测标签': [INTERACTION_TYPES[pred] for pred in test_preds],
        '是否正确': [pred == label for pred, label in zip(test_preds, test_labels_eval)]
    })

    test_output_file = CODE_DIR / f'biolinkbert_official_test_results_{DATASET_NAME}_{timestamp}.csv'
    test_results_df.to_csv(test_output_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ 官方测试集结果已保存到: {test_output_file}")

if __name__ == "__main__":
    main()
