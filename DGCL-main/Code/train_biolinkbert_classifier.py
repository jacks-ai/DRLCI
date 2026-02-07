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

# 添加项目路径
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.append(str(CODE_DIR))

from Params import args
from DataHandler import DataHandler

# ==================== 路径配置 ====================
timestamp = datetime.now().strftime("%m%d_%H%M%S")
save_path = CODE_DIR / f'bert/best_biolinkbert_classifier_{timestamp}.pt'
DATA_ROOT = PROJECT_ROOT / 'Data' / args.data
MODEL_CACHE = Path(r"/mnt/data/huangpeng/DGCL/mymodel/BioLinkBERT")
ERROR_CASES_CSV = "/mnt/data/huangpeng/DGCL/DGCL-main/log/0204_233151_DGIdb_error_cases.csv"  # 已注释，改用DataHandler加载测试集
DRUG_DESC_CSV = DATA_ROOT / 'drug_text' / 'mixed_drug_descriptions.csv'
GENE_DESC_JSON = DATA_ROOT / 'gene_text' / 'gene_embeddings_txt.json'
GLOBAL_IDS_JSON = DATA_ROOT / 'global_ids.json'
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
    'learning_rate': 2e-5,         # BERT微调推荐: 2e-5 ~ 5e-5
    'num_epochs': 10,              # BERT微调推荐: 3-10 epochs
    'warmup_ratio': 0.1,           # 预热10%的训练步数
    'max_grad_norm': 1.0,          # 梯度裁剪 (BERT标准: 1.0)
    'weight_decay': 0.01,          # AdamW权重衰减 (BERT标准: 0.01)
    'save_steps': 500,             # 每500步保存一次
    'eval_steps': 500,             # 每500步评估一次
    'fp16': True,                  # V100 必开！提速 2-3 倍，节省 50% 显存
    'max_length': 256,             # 最大序列长度（联合编码需要更长）
    
    # 负采样配置（已禁用）
    'num_neg_samples': 0,         # 不使用负采样
    'use_contrastive_loss': False,  # 不使用对比损失
    'contrastive_weight': 0.0,     # 对比损失权重（已禁用）
}

# Prompt模板（联合编码）
PROMPT_TEMPLATE = "[CLS] Drug: {drug_desc} [SEP] Gene: {gene_desc} [SEP]"

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
    
    def forward(self, input_ids, attention_mask):
        """
        联合编码drug和gene，提取[CLS] token送入分类器
        
        Args:
            input_ids: [batch_size, seq_len] - 联合编码的token IDs
            attention_mask: [batch_size, seq_len] - 注意力掩码
        
        Returns:
            logits: [batch_size, num_classes]
        """
        # 联合编码
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
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
        drug_idx = self.drug_ids[idx]
        gene_idx = self.gene_ids[idx]
        label = self.labels[idx]
        
        # 获取全局ID
        drug_id = self.drug_ids_global[drug_idx]
        gene_id = self.gene_ids_global[gene_idx]
        
        # 获取描述文本
        drug_desc = self.drug_descriptions.get(drug_id, f"Drug {drug_id}")[:200]
        gene_desc = self.gene_descriptions.get(gene_id, f"Gene {gene_id}")[:200]
        
        # 构建联合prompt
        joint_prompt = f"[CLS] Drug: {drug_desc} [SEP] Gene: {gene_desc} [SEP]"
        
        # Tokenize
        encoding = self.tokenizer(
            joint_prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }
        
        return result


# ==================== 数据加载函数 ====================
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


def load_training_and_test_data():
    """
    使用DataHandler加载训练数据和测试数据
    返回: (train_drugs, train_genes, train_labels, test_drugs, test_genes, test_labels)
    """
    print(f"\n[加载数据] 数据集: {args.data}")
    
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


def load_test_data_from_error_cases(drug_ids_global, gene_ids_global):
    """
    加载测试数据 (错误案例)
    """
    print(f"\n[加载错误案例测试集] 来源: {ERROR_CASES_CSV}")
    
    df = pd.read_csv(ERROR_CASES_CSV, encoding='utf-8')
    df_iter1 = df[df['Iteration'] == 1]
    
    error_drugs = df_iter1['药物ID'].values.astype(int)
    error_genes = df_iter1['基因ID'].values.astype(int)
    error_labels = df_iter1['真实标签'].values.astype(int)
    
    print(f"  ✓ 错误案例数: {len(error_labels)}")
    
    return error_drugs, error_genes, error_labels


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
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # FP16 混合精度训练
        if use_fp16:
            with autocast('cuda'):
                logits = model(input_ids, attention_mask)
                loss = F.cross_entropy(logits, labels)
            
            # 反向传播 (FP16)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准 FP32 训练
            logits = model(input_ids, attention_mask)
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
            labels = batch['label'].to(device)
            
            logits = model(input_ids, attention_mask)
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
    log_and_print(f"数据集: {args.data}", LOG_FILE)
    log_and_print(f"训练配置: {TRAIN_CONFIG}", LOG_FILE)
    log_and_print(f"日志文件: {LOG_FILE}", LOG_FILE)
    log_and_print("=" * 80, LOG_FILE)
    
    # 设置设备（支持指定GPU）
    if torch.cuda.is_available():
        # 从 Params.py 读取 GPU 配置
        gpu_id = args.gpu if hasattr(args, 'gpu') else 0
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)
        log_and_print(f"\n设备: {device} ({torch.cuda.get_device_name(gpu_id)})", LOG_FILE)
        log_and_print(f"显存: {torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3:.1f} GB", LOG_FILE)
    else:
        device = torch.device('cpu')
        log_and_print(f"\n设备: CPU (CUDA不可用)", LOG_FILE)
    
    # 1. 加载全局数据
    log_and_print("\n[1/9] 加载全局数据...", LOG_FILE)
    drug_ids_global, gene_ids_global = load_global_ids()
    drug_descriptions = load_drug_descriptions()
    gene_descriptions = load_gene_descriptions()
    log_and_print(f"  ✓ 药物数: {len(drug_ids_global)}, 基因数: {len(gene_ids_global)}", LOG_FILE)
    
    # 2. 加载训练数据和测试数据
    log_and_print("\n[2/9] 加载训练数据和测试数据...", LOG_FILE)
    train_drugs, train_genes, train_labels, test_drugs, test_genes, test_labels = load_training_and_test_data()
    
    # 2.5 加载错误案例测试集
    log_and_print("\n[2.5/9] 加载错误案例测试集...", LOG_FILE)
    error_drugs, error_genes, error_labels = load_test_data_from_error_cases(drug_ids_global, gene_ids_global)
    
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
    
    # 错误案例测试集
    error_dataset = DrugGeneDataset(
        error_drugs, error_genes, error_labels,
        drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global, tokenizer, max_length
    )
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    error_loader = DataLoader(error_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    
    log_and_print(f"  ✓ 训练批次数: {len(train_loader)}", LOG_FILE)
    log_and_print(f"  ✓ 官方测试批次数: {len(test_loader)}", LOG_FILE)
    log_and_print(f"  ✓ 错误案例批次数: {len(error_loader)}", LOG_FILE)
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
    best_error_acc = 0.0
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    training_start_time = time.time()
    
    # 记录每个epoch的准确率
    official_test_acc_history = []
    error_cases_acc_history = []
    
    for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
        log_and_print(f"\n{'='*60}", LOG_FILE)
        log_and_print(f"Epoch {epoch}/{TRAIN_CONFIG['num_epochs']}", LOG_FILE)
        log_and_print(f"{'='*60}", LOG_FILE)
        
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, scaler
        )
        epoch_time = time.time() - epoch_start_time
        
        log_and_print(f"  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, 耗时: {epoch_time:.1f}s", LOG_FILE)
        
        # 在官方测试集上评估
        test_acc, test_top5_acc, test_preds, test_labels_eval = evaluate(model, test_loader, device)
        log_and_print(f"  官方测试集 - Accuracy: {test_acc:.4f}, Top-5 Accuracy: {test_top5_acc:.4f}", LOG_FILE)
        official_test_acc_history.append(test_acc)
        
        # 在错误案例上评估
        error_acc, error_top5_acc, error_preds, error_labels_eval = evaluate(model, error_loader, device)
        log_and_print(f"  错误案例集 - Accuracy: {error_acc:.4f}, Top-5 Accuracy: {error_top5_acc:.4f}", LOG_FILE)
        error_cases_acc_history.append(error_acc)
        
        # 保存最佳模型（基于官方测试集）
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_error_acc = error_acc  # 记录此时错误案例的准确率
            save_path = CODE_DIR / f'bert/best_biolinkbert_classifier_{timestamp}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_top5_acc': test_top5_acc,
                'error_acc': error_acc,
                'error_top5_acc': error_top5_acc,
                'config': TRAIN_CONFIG,
            }, save_path)
            log_and_print(f"  ✓ 保存最佳模型: {save_path}", LOG_FILE)
    
    total_training_time = time.time() - training_start_time
    log_and_print(f"\n总训练时间: {total_training_time/60:.1f} 分钟", LOG_FILE)
    
    # 7. 最终评估
    log_and_print("\n[7/9] 最终评估...", LOG_FILE)
    log_and_print(f"\n{'='*80}", LOG_FILE)
    log_and_print(f"训练完成！", LOG_FILE)
    log_and_print(f"{'='*80}", LOG_FILE)
    
    # 找到最大准确率对应的epoch
    best_official_epoch = np.argmax(official_test_acc_history) + 1  # +1 因为epoch从1开始
    best_error_epoch = np.argmax(error_cases_acc_history) + 1
    
    log_and_print(f"最佳官方测试准确率: {best_test_acc:.4f} (Epoch {best_official_epoch})", LOG_FILE)
    log_and_print(f"对应的错误案例准确率: {best_error_acc:.4f}", LOG_FILE)
    log_and_print(f"\n错误案例集最佳准确率: {max(error_cases_acc_history):.4f} (Epoch {best_error_epoch})", LOG_FILE)
    
    # 输出每个epoch的准确率历史
    log_and_print(f"\n{'='*80}", LOG_FILE)
    log_and_print("📊 每个Epoch的准确率历史", LOG_FILE)
    log_and_print(f"{'='*80}", LOG_FILE)
    log_and_print(f"官方测试集准确率 (每个epoch): {official_test_acc_history}", LOG_FILE)
    log_and_print(f"错误案例集准确率 (每个epoch): {error_cases_acc_history}", LOG_FILE)
    log_and_print(f"\n官方测试集最大准确率: {max(official_test_acc_history):.4f} 在 Epoch {best_official_epoch}", LOG_FILE)
    log_and_print(f"错误案例集最大准确率: {max(error_cases_acc_history):.4f} 在 Epoch {best_error_epoch}", LOG_FILE)
    log_and_print(f"{'='*80}", LOG_FILE)
    
    # 7.1 官方测试集分类报告
    print(f"\n{'='*80}")
    print("📊 官方测试集 - 分类报告")
    print(f"{'='*80}")
    print(classification_report(test_labels_eval, test_preds, target_names=INTERACTION_TYPES, zero_division=0))
    
    # 7.2 错误案例测试集分类报告
    print(f"\n{'='*80}")
    print("📊 错误案例测试集 - 分类报告")
    print(f"{'='*80}")
    print(classification_report(error_labels_eval, error_preds, target_names=INTERACTION_TYPES, zero_division=0))
    
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
    
    test_output_file = CODE_DIR / f'biolinkbert_official_test_results_{timestamp}.csv'
    test_results_df.to_csv(test_output_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ 官方测试集结果已保存到: {test_output_file}")
    
    # 8.2 保存错误案例测试集结果
    error_results_df = pd.DataFrame({
        '药物ID': [drug_ids_global[idx] for idx in error_drugs],
        '基因ID': [gene_ids_global[idx] for idx in error_genes],
        '真实标签': [INTERACTION_TYPES[label] for label in error_labels_eval],
        '预测标签': [INTERACTION_TYPES[pred] for pred in error_preds],
        '是否正确': [pred == label for pred, label in zip(error_preds, error_labels_eval)]
    })
    
    error_output_file = CODE_DIR / f'biolinkbert_error_cases_results_{timestamp}.csv'
    error_results_df.to_csv(error_output_file, index=False, encoding='utf-8-sig')
    print(f"  ✓ 错误案例结果已保存到: {error_output_file}")

if __name__ == "__main__":
    main()
