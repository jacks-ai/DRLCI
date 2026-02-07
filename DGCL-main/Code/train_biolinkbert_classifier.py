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
from torch.cuda.amp import autocast, GradScaler  # FP16 混合精度训练

# 添加项目路径
CODE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CODE_DIR.parent
sys.path.append(str(CODE_DIR))

from Params import args
from DataHandler import DataHandler

# ==================== 路径配置 ====================
DATA_ROOT = PROJECT_ROOT / 'Data' / args.data
MODEL_CACHE = Path(r"/mnt/data/huangpeng/DGCL/mymodel/BioLinkBERT")
# ERROR_CASES_CSV = "/mnt/data/huangpeng/DGCL/DGCL-main/log/0204_233151_DGIdb_error_cases.csv"  # 已注释，改用DataHandler加载测试集
DRUG_DESC_CSV = DATA_ROOT / 'drug_text' / 'mixed_drug_descriptions.csv'
GENE_DESC_JSON = DATA_ROOT / 'gene_text' / 'gene_embeddings_txt.json'
GLOBAL_IDS_JSON = DATA_ROOT / 'global_ids.json'

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
}

# Prompt模板
PROMPT_TEMPLATE = "[CLS] Drug: {drug_desc} [SEP] Gene: {gene_desc} [SEP] The interaction between this drug and gene is [MASK] ."

# 14种交互类型
INTERACTION_TYPES = [
    "Agonist", "Antagonist", "Antibody", "Modulator", "Blocker", "Binder",
    "Potentiator", "Cofactor", "Ligand", "Inhibitor", "Activator",
    "Partial agonist", "Positive modulator", "Allosteric modulator"
]


# ==================== 模型定义 ====================
class ClassifierLayer(nn.Module):
    """
    与Model_sparse.py完全一致的MLP分类器
    输入: drug_embed (1024) + gene_embed (1024) = 2048
    """
    def __init__(self, input_dim=2048, num_classes=14):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(input_dim, 128)
        self.lin2 = nn.Linear(128, num_classes)
    
    def forward(self, dEmbeds, gEmbeds):
        embeds = torch.cat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        ret = self.lin2(embeds)
        return ret


class BioLinkBERTClassifier(nn.Module):
    """
    BioLinkBERT + MLP分类器 (端到端训练)
    """
    def __init__(self, bert_model, num_classes=14):
        super().__init__()
        self.bert = bert_model
        hidden_dim = bert_model.config.hidden_size  # 1024 for BioLinkBERT-large
        
        # 使用与Model_sparse.py一致的分类器
        self.classifier = ClassifierLayer(input_dim=hidden_dim * 2, num_classes=num_classes)
    
    def forward(self, drug_input_ids, drug_attention_mask, gene_input_ids, gene_attention_mask):
        """
        分别编码drug和gene，提取[CLS] token，然后拼接送入分类器
        """
        # 编码drug
        drug_outputs = self.bert(input_ids=drug_input_ids, attention_mask=drug_attention_mask)
        drug_cls = drug_outputs.last_hidden_state[:, 0, :]  # [batch_size, 1024]
        
        # 编码gene
        gene_outputs = self.bert(input_ids=gene_input_ids, attention_mask=gene_attention_mask)
        gene_cls = gene_outputs.last_hidden_state[:, 0, :]  # [batch_size, 1024]
        
        # 通过分类器
        logits = self.classifier(drug_cls, gene_cls)  # [batch_size, num_classes]
        
        return logits


# ==================== 数据集定义 ====================
class DrugGeneDataset(Dataset):
    """
    药物-基因交互数据集
    """
    def __init__(self, drug_ids, gene_ids, labels, drug_descriptions, gene_descriptions, 
                 drug_ids_global, gene_ids_global, tokenizer, max_length=256):
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
        
        # 构建prompt (分别为drug和gene)
        drug_prompt = f"[CLS] Drug: {drug_desc} [SEP]"
        gene_prompt = f"[CLS] Gene: {gene_desc} [SEP]"
        
        # Tokenize
        drug_encoding = self.tokenizer(
            drug_prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        gene_encoding = self.tokenizer(
            gene_prompt,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'drug_input_ids': drug_encoding['input_ids'].squeeze(0),
            'drug_attention_mask': drug_encoding['attention_mask'].squeeze(0),
            'gene_input_ids': gene_encoding['input_ids'].squeeze(0),
            'gene_attention_mask': gene_encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }


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


# def load_test_data_from_error_cases(drug_ids_global, gene_ids_global):
#     """
#     加载测试数据 (错误案例) - 已注释，改用DataHandler加载
#     """
#     print(f"\n[加载测试数据] 来源: {ERROR_CASES_CSV}")
#     
#     df = pd.read_csv(ERROR_CASES_CSV, encoding='utf-8')
#     df_iter1 = df[df['Iteration'] == 1]
#     
#     test_drugs = df_iter1['药物ID'].values.astype(int)
#     test_genes = df_iter1['基因ID'].values.astype(int)
#     test_labels = df_iter1['真实标签'].values.astype(int)
#     
#     print(f"  ✓ 测试样本数: {len(test_labels)}")
#     
#     return test_drugs, test_genes, test_labels


# ==================== 训练和评估函数 ====================
def train_epoch(model, dataloader, optimizer, scheduler, device, epoch, scaler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    use_fp16 = scaler is not None
    
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch in progress_bar:
        # 移动到设备
        drug_input_ids = batch['drug_input_ids'].to(device)
        drug_attention_mask = batch['drug_attention_mask'].to(device)
        gene_input_ids = batch['gene_input_ids'].to(device)
        gene_attention_mask = batch['gene_attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # FP16 混合精度训练
        if use_fp16:
            with autocast():
                logits = model(drug_input_ids, drug_attention_mask, gene_input_ids, gene_attention_mask)
                loss = F.cross_entropy(logits, labels)
            
            # 反向传播 (FP16)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
        else:
            # 标准 FP32 训练
            logits = model(drug_input_ids, drug_attention_mask, gene_input_ids, gene_attention_mask)
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
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            drug_input_ids = batch['drug_input_ids'].to(device)
            drug_attention_mask = batch['drug_attention_mask'].to(device)
            gene_input_ids = batch['gene_input_ids'].to(device)
            gene_attention_mask = batch['gene_attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(drug_input_ids, drug_attention_mask, gene_input_ids, gene_attention_mask)
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
    print("=" * 80)
    print("BioLinkBERT + MLP 分类器训练")
    print("=" * 80)
    print(f"数据集: {args.data}")
    print(f"训练配置: {TRAIN_CONFIG}")
    print("=" * 80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n设备: {device}")
    
    # 1. 加载全局数据
    print("\n[1/8] 加载全局数据...")
    drug_ids_global, gene_ids_global = load_global_ids()
    drug_descriptions = load_drug_descriptions()
    gene_descriptions = load_gene_descriptions()
    print(f"  ✓ 药物数: {len(drug_ids_global)}, 基因数: {len(gene_ids_global)}")
    
    # 2. 加载训练数据和测试数据
    print("\n[2/8] 加载训练数据和测试数据...")
    train_drugs, train_genes, train_labels, test_drugs, test_genes, test_labels = load_training_and_test_data()
    
    # 3. 加载模型和tokenizer
    print("\n[3/8] 加载BioLinkBERT模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)
    bert_model = AutoModel.from_pretrained(MODEL_CACHE, local_files_only=True, trust_remote_code=True)
    
    model = BioLinkBERTClassifier(bert_model, num_classes=14)
    model = model.to(device)
    print(f"  ✓ 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 4. 创建数据集和数据加载器
    print("\n[4/8] 创建数据加载器...")
    train_dataset = DrugGeneDataset(
        train_drugs, train_genes, train_labels,
        drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global, tokenizer
    )
    test_dataset = DrugGeneDataset(
        test_drugs, test_genes, test_labels,
        drug_descriptions, gene_descriptions,
        drug_ids_global, gene_ids_global, tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG['batch_size'], shuffle=False)
    
    print(f"  ✓ 训练批次数: {len(train_loader)}")
    print(f"  ✓ 测试批次数: {len(test_loader)}")
    
    # 5. 设置优化器和学习率调度器
    print("\n[5/8] 设置优化器...")
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
            scaler = GradScaler()
            print(f"  ✓ FP16 混合精度训练已启用 (预期提速 2-3x)")
        else:
            print(f"  ⚠ FP16 需要 CUDA，已自动切换到 FP32")
    
    print(f"  ✓ 总训练步数: {total_steps}, 预热步数: {warmup_steps}")
    
    # 6. 训练循环
    print("\n[6/8] 开始训练...")
    best_test_acc = 0.0
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    training_start_time = time.time()
    
    for epoch in range(1, TRAIN_CONFIG['num_epochs'] + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{TRAIN_CONFIG['num_epochs']}")
        print(f"{'='*60}")
        
        epoch_start_time = time.time()
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, scaler)
        epoch_time = time.time() - epoch_start_time
        print(f"  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}, 耗时: {epoch_time:.1f}s")
        
        # 评估
        test_acc, test_top5_acc, test_preds, test_labels_eval = evaluate(model, test_loader, device)
        print(f"  测试 - Accuracy: {test_acc:.4f}, Top-5 Accuracy: {test_top5_acc:.4f}")
        
        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            save_path = CODE_DIR / f'bert/best_biolinkbert_classifier_{timestamp}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_top5_acc': test_top5_acc,
                'config': TRAIN_CONFIG,
            }, save_path)
            print(f"  ✓ 保存最佳模型: {save_path}")
    
    total_training_time = time.time() - training_start_time
    print(f"\n总训练时间: {total_training_time/60:.1f} 分钟")
    
    # 7. 最终评估和保存结果
    print("\n[7/8] 最终评估...")
    print(f"\n{'='*80}")
    print(f"训练完成！")
    print(f"{'='*80}")
    print(f"最佳测试准确率: {best_test_acc:.4f}")
    
    # 生成分类报告
    print("\n分类报告:")
    print(classification_report(test_labels_eval, test_preds, target_names=INTERACTION_TYPES, zero_division=0))
    
    # 保存预测结果
    results_df = pd.DataFrame({
        '药物ID': [drug_ids_global[idx] for idx in test_drugs],
        '基因ID': [gene_ids_global[idx] for idx in test_genes],
        '真实标签': [INTERACTION_TYPES[label] for label in test_labels_eval],
        '预测标签': [INTERACTION_TYPES[pred] for pred in test_preds],
        '是否正确': [pred == label for pred, label in zip(test_preds, test_labels_eval)]
    })
    
    output_file = CODE_DIR / f'biolinkbert_classifier_results_{timestamp}.csv'
    results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存到: {output_file}")


if __name__ == "__main__":
    main()
