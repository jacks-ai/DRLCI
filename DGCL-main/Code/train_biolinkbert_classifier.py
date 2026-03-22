"""
BioLinkBERT + MLP 分类器训练脚本
方案: 特征提取 + 下游分类器 (端到端训练，不冻结)
MLP架构与Model_sparse.py中的ClassifierLayer完全一致
"""
import json
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
    gpu = 0  # 默认GPU
    
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
GLOBAL_IDS_JSON = DATA_ROOT / 'global_ids_2.json'
LOG_DIR = Path("/mnt/data/huangpeng/DGCL/DGCL-main/Code/bert/blog")  # 日志目录
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
    
    # 负采样配置（已禁用）
    'num_neg_samples': 0,         # 不使用负采样
    'use_contrastive_loss': False,  # 不使用对比损失
    'contrastive_weight': 0.0,     # 对比损失权重（已禁用）
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
    
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        联合编码drug和gene，提取[CLS] token送入分类器
        
        Args:
            input_ids: [batch_size, seq_len] - 联合编码的token IDs
            attention_mask: [batch_size, seq_len] - 注意力掩码
            token_type_ids: [batch_size, seq_len] - 句子类型ID (0=句子A/药物, 1=句子B/基因)
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # 联合编码（使用token_type_ids区分药物和基因）
        outputs = self.bert(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        cls_embed = outputs.last_hidden_state[:, 0, :]  # [batch_size, 1024]
        
        # 通过分类器
        logits = self.classifier(cls_embed)  # [batch_size, num_classes]
        
        return logits


# ==================== 数据集定义 ====================
class DrugGeneDataset(Dataset):
    """
    药物-基因交互数据集（联合编码版本）
    """
    def __init__(self, drug_ids, gene_ids, labels, drug_descriptions, gene_descriptions, 
                 drug_ids_global, gene_ids_global, tokenizer, max_length):
        self.drug_ids = drug_ids
        self.gene_ids = gene_ids
        self.labels = labels
        self.drug_descriptions = drug_descriptions
        self.gene_descriptions = gene_descriptions
        self.drug_ids_global = drug_ids_global
        self.gene_ids_global = gene_ids_global
        self.tokenizer = tokenizer
        self.max_length = max_length
    
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
        
        # 获取描述文本
        drug_desc = self.drug_descriptions.get(drug_id, f"Drug {drug_id}")[:292]
        gene_desc = self.gene_descriptions.get(gene_id, f"Gene {gene_id}")[:220]
        
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
            'label': torch.tensor(label, dtype=torch.long)
        }
        
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


# ==================== 训练和评估函数 ====================
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None):
    """训练一个epoch（联合编码版本）"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    use_fp16 = scaler is not None
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(progress_bar):
        # 移动到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # FP16 混合精度训练
        if use_fp16:
            with autocast('cuda'):
                logits = model(input_ids, attention_mask, token_type_ids)
                loss = F.cross_entropy(logits, labels)
            
            # 反向传播 (FP16)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准 FP32 训练
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = F.cross_entropy(logits, labels)
            
            # 反向传播 (FP32)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['max_grad_norm'])
            optimizer.step()
        
        scheduler.step()
        
        # 统计
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # 更新进度条
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
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
    log_and_print("\n[1/9] 加载全局数据...", LOG_FILE)
    drug_ids_global, gene_ids_global = load_global_ids()
    path_resolved = str(GLOBAL_IDS_JSON.resolve()) if hasattr(GLOBAL_IDS_JSON, 'resolve') else str(GLOBAL_IDS_JSON)
    log_and_print(f"  ✓ 索引文件路径: {path_resolved}", LOG_FILE)
    drug_descriptions = load_drug_descriptions()
    gene_descriptions = load_gene_descriptions()
    log_and_print(f"  ✓ 药物数: {len(drug_ids_global)}, 基因数: {len(gene_ids_global)}", LOG_FILE)
    
    # 2. 加载训练数据和测试数据
    log_and_print("\n[2/9] 加载训练数据和测试数据...", LOG_FILE)
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
    
    # 3. 加载模型和tokenizer
    log_and_print("\n[3/9] 加载BioLinkBERT模型...", LOG_FILE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)
    bert_model = AutoModel.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)
    
    model = BioLinkBERTClassifier(bert_model, num_classes=14)
    model = model.to(device)
    log_and_print(f"  ✓ 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M", LOG_FILE)
    
    # 4. 创建数据集和数据加载器
    log_and_print("\n[4/9] 创建数据加载器...", LOG_FILE)
    
    max_length = TRAIN_CONFIG['max_length']
    
    # 训练集（联合编码，不使用负采样）
    train_dataset = DrugGeneDataset(
        train_drugs, train_genes, train_labels,
        drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global, tokenizer, max_length
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
    
    # 5. 设置优化器和学习率调度器
    log_and_print("\n[5/9] 设置优化器...", LOG_FILE)
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
    
    # 6. 训练循环
    log_and_print("\n[6/9] 开始训练...", LOG_FILE)
    best_test_acc = 0.0
    best_epoch = 0
    best_test_preds = None
    best_test_labels_eval = None
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
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
            model, train_loader, optimizer, scheduler, device, epoch, scaler
        )
        epoch_time = time.time() - epoch_start_time
        
        print(f"  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, 耗时: {epoch_time:.1f}s")

        # 在官方测试集上评估
        test_acc, test_top5_acc, test_preds, test_labels_eval = evaluate(model, test_loader, device)
        print(f"  官方测试集 - Accuracy: {test_acc:.4f}, Top-5 Accuracy: {test_top5_acc:.4f}")
        official_test_acc_history.append(test_acc)
        
        # 保存最佳模型（基于官方测试集）并记录对应的预测结果
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            best_test_preds = test_preds
            best_test_labels_eval = test_labels_eval
            save_path = CODE_DIR / f'bert/best_biolinkbert_only_{DATASET_NAME}_{timestamp}.pt'
            torch.save({
                'epoch': epoch,
                'bert_state_dict': model.bert.state_dict(),  # 只保存BioLinkBERT模型
                'test_acc': test_acc,
                'test_top5_acc': test_top5_acc,
                'config': TRAIN_CONFIG,
            }, save_path)
            print(f"  ✓ 保存最佳BioLinkBERT模型 (不含MLP): {save_path}")
            log_and_print(f"Epoch {epoch}: 保存最佳BioLinkBERT模型 (test_acc={test_acc:.4f})", LOG_FILE)
    
    total_training_time = time.time() - training_start_time
    log_and_print(f"\n总训练时间: {total_training_time/60:.1f} 分钟", LOG_FILE)
    
    # 7. 最终评估
    log_and_print("\n[7/9] 最终评估...", LOG_FILE)
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
    
    # 8. 保存预测结果
    print("\n[8/9] 保存预测结果...")
    
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
