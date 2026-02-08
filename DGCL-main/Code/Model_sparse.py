import torch as t
from torch import nn
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, calcRegLoss
import numpy as np
import torch_sparse
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


# --- 新增：特征级门控拼接层 (Feature-wise Gated Concatenation)  ---
class GatedConcatFusion(nn.Module):
    def __init__(self, latdim):
        super(GatedConcatFusion, self).__init__()
        self.output_dim = latdim * 2

        # 【新增修复】加入 LayerNorm，强制稳定输入分布
        self.ln = nn.LayerNorm(self.output_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.output_dim // 2, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, struct_feat, text_feat):
        combined = t.cat([struct_feat, text_feat], dim=1)

        # 先过 LayerNorm 再进 Gate 计算权重
        # 这样即使 combined 有点大，算权重的网络也不会崩
        weights = self.gate(self.ln(combined))

        fused_embeds = combined * weights
        return fused_embeds


# --- 新增：GCN + BioLinkBERT 融合门控 (方案 A - 保留完整 BERT 维度) ---
class BERTGCNFusion(nn.Module):
    """
    融合 GCN 特征 (256维，drug+gene拼接) 和 BioLinkBERT 特征 (1024维)
    方案 A：保留 BERT 完整维度，不压缩，与 GatedConcatFusion 设计一致
    """
    def __init__(self, gcn_dim=256, bert_dim=1024):
        super().__init__()
        
        # 不压缩，直接拼接
        total_dim = gcn_dim + bert_dim  # 256 + 1024 = 1280
        
        # LayerNorm 稳定训练
        self.ln = nn.LayerNorm(total_dim)
        
        # 门控机制（与 GatedConcatFusion 完全一致的设计）
        self.gate = nn.Sequential(
            nn.Linear(total_dim, total_dim // 2),  # 1280 → 640
            nn.ReLU(),
            nn.Linear(total_dim // 2, total_dim),  # 640 → 1280
            nn.Sigmoid()
        )
    
    def forward(self, gcn_feat, bert_feat):
        """
        Args:
            gcn_feat: [batch_size, 256] - GCN 提取的特征（drug+gene拼接）
            bert_feat: [batch_size, 1024] - BERT [CLS] 特征
        
        Returns:
            fused: [batch_size, 1280] - 融合后的特征（保留完整信息）
        """
        # 直接拼接，不压缩
        combined = t.cat([gcn_feat, bert_feat], dim=1)  # [batch, 1280]
        
        # 门控融合（与 GatedConcatFusion 相同的流程）
        weights = self.gate(self.ln(combined))
        fused = combined * weights
        
        return fused  # [batch, 1280]


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 步骤 1: 初始化结构嵌入
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))  # 药物结构嵌入
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))  # 基因结构嵌入

        #  检查是否启用 BERT 融合（方案 B）
        self.use_bert_fusion = (args.use_bert_fusion == 1)
        
        # 检查是否使用额外的文本特征（与 BERT 融合互斥）
        if self.use_bert_fusion and args.use_llm_embeddings:
            print("⚠️  警告：use_bert_fusion 和 use_llm_embeddings 不能同时启用！")
            print("  自动禁用 use_llm_embeddings，使用 BERT 融合方案")
            args.use_llm_embeddings = False
        
        self.use_text_features = (
                args.use_llm_embeddings
                and args.pretrained_drug_embed_path is not None
                and args.pretrained_gene_embed_path is not None
        )

        if self.use_text_features:
            # 步骤 2: 加载预训练 LLM 嵌入
            print("加载预训练的LLM文本嵌入作为辅助特征...")
            drug_embeds_text = t.from_numpy(np.load(args.pretrained_drug_embed_path)).float()
            gene_embeds_text = t.from_numpy(np.load(args.pretrained_gene_embed_path)).float()

            assert drug_embeds_text.shape[0] == args.drug, "Drug count mismatch"
            assert gene_embeds_text.shape[0] == args.gene, "Gene count mismatch"

            drug_text_dim = drug_embeds_text.shape[1]
            gene_text_dim = gene_embeds_text.shape[1]

            # 投影层：将文本维度对齐到 args.latdim
            self.drug_text_proj = nn.Linear(drug_text_dim, args.latdim)
            self.gene_text_proj = nn.Linear(gene_text_dim, args.latdim)

            # 初始化权重
            nn.init.xavier_uniform_(self.drug_text_proj.weight)
            nn.init.xavier_uniform_(self.gene_text_proj.weight)

            self.register_buffer('drug_embeds_text', drug_embeds_text)
            self.register_buffer('gene_embeds_text', gene_embeds_text)

            # 初始化门控拼接融合模块
            self.fusion_layer = GatedConcatFusion(args.latdim)
            print("Gated Concat 融合模块已初始化.")
        else:
            print("随机初始化，Initializing drug and gene embeddings randomly.")

        # Initialize GCN
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        # Initialize classifier
        self.classifierLayer = ClassifierLayer()

        self.edgeDropper = SpAdjDropEdge()
        
        # 方案 A：初始化 BERT 融合相关组件
        if self.use_bert_fusion:
            print("🔥 启用方案 A：GCN + BioLinkBERT 融合（保留完整 BERT 维度）")
            # BERT 融合门控（256 GCN + 1024 BERT = 1280 维输出）
            self.bert_gcn_fusion = BERTGCNFusion(gcn_dim=256, bert_dim=1024)
            # BERT 特征缓存将在外部加载并传入
            self.bert_features_cache = None
            print("  ✓ BERT-GCN 融合门控已初始化（输出 1280 维，等待加载特征缓存）")

    def forward(self, adj, keepRate):
        # 步骤 3: GCN 传播 (结构视图)
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]

        for gcn in self.gcnLayers:
            struct_embeds = gcn(self.edgeDropper(adj, keepRate), embedsLst[-1])
            embedsLst.append(struct_embeds)

        final_struct_embeds = sum(embedsLst)

        all_text_feat = None

        # 步骤 4: 融合文本特征
        if self.use_text_features:
            # 投影文本特征
            drug_text_feat = self.drug_text_proj(self.drug_embeds_text.to(final_struct_embeds.device))
            gene_text_feat = self.gene_text_proj(self.gene_embeds_text.to(final_struct_embeds.device))

            # 归一化 (防止数值范围差异过大)
            drug_text_feat = F.normalize(drug_text_feat, p=2, dim=1)
            gene_text_feat = F.normalize(gene_text_feat, p=2, dim=1)

            all_text_feat = t.concat([drug_text_feat, gene_text_feat], axis=0)

            # 使用 Gated Concat 进行融合
            # 输出维度: [nodes, latdim * 2]
            final_embeds = self.fusion_layer(final_struct_embeds, all_text_feat)
        else:
            final_embeds = final_struct_embeds

        return final_embeds,final_struct_embeds,all_text_feat

    def forward_gcn(self, adj):
        # 用于推理或无 Dropout 的前向传播
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        final_struct_embeds = sum(embedsLst)

        if self.use_text_features:
            drug_text_feat = self.drug_text_proj(self.drug_embeds_text.to(final_struct_embeds.device))
            gene_text_feat = self.gene_text_proj(self.gene_embeds_text.to(final_struct_embeds.device))

            drug_text_feat = F.normalize(drug_text_feat, p=2, dim=1)
            gene_text_feat = F.normalize(gene_text_feat, p=2, dim=1)
            all_text_feat = t.concat([drug_text_feat, gene_text_feat], axis=0)

            final_embeds = self.fusion_layer(final_struct_embeds, all_text_feat)
        else:
            final_embeds = final_struct_embeds

        return final_embeds[:args.drug], final_embeds[args.drug:]

    def calcLosses(self, drugs, genes, labels, adj, keepRate):
        embeds,struct_view,text_view = self.forward(adj, keepRate)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)
        sslLoss=0
        # 没有用
        # if self.use_text_features:
        #     alpha = 0.0000001
        #     sslLoss = contrastLoss(struct_view[drugs], text_view[drugs],args.temp) + \
        #               contrastLoss(struct_view[genes], text_view[genes],args.temp)
        #     sslLoss = sslLoss*alpha
        return ceLoss, sslLoss
    
    def calcLosses_with_bert_cached(self, drugs, genes, labels, adj, keepRate):
        """
        方案 A 优化版：使用预计算的 BERT 特征缓存，保留完整 BERT 维度
        避免每次都重新计算 BERT，提升训练速度 450x
        
        Args:
            drugs: [batch_size] - 药物索引
            genes: [batch_size] - 基因索引
            labels: [batch_size] - 标签
            adj: 邻接矩阵
            keepRate: dropout 保留率
        """
        # 1. 获取 GCN 特征（使用 forward_gcn 获取纯结构特征，避免文本融合）
        drug_gcn_embeds, gene_gcn_embeds = self.forward_gcn(adj)
        
        # 提取当前 batch 的 GCN 特征
        drug_gcn_feat = drug_gcn_embeds[drugs]  # [batch_size, 128]
        gene_gcn_feat = gene_gcn_embeds[genes]  # [batch_size, 128]
        gcn_feat = t.cat([drug_gcn_feat, gene_gcn_feat], dim=1)  # [batch_size, 256]
        
        # 2. 从缓存中获取 BERT 特征（无需计算，直接查表）
        batch_size = len(drugs)
        bert_feats = []
        
        for i in range(batch_size):
            drug_idx = drugs[i].item()
            gene_idx = genes[i].item()
            key = (drug_idx, gene_idx)
            
            # 从缓存中查找
            if key in self.bert_features_cache:
                bert_feat = self.bert_features_cache[key]
                bert_feats.append(bert_feat)
            else:
                # 如果缓存中没有，使用零向量（不应该发生）
                print(f"⚠️ Warning: BERT feature not found for ({drug_idx}, {gene_idx})")
                bert_feats.append(np.zeros(1024, dtype=np.float32))
        
        # 转换为 tensor
        bert_feat = t.from_numpy(np.stack(bert_feats)).to(drugs.device)  # [batch_size, 1024]
        
        # 3. 融合 GCN 和 BERT 特征（保留完整维度）
        fused_feat = self.bert_gcn_fusion(gcn_feat, bert_feat)  # [batch_size, 1280]
        
        # 4. 分类（方案 A：直接使用融合后的 1280 维特征）
        # 不再拆分，因为融合特征已经包含了 drug+gene+BERT 的完整信息
        # 分类器会自动处理 1280 维输入
        pre = self.classifierLayer(fused_feat, fused_feat)  # 两个参数都传入完整特征
        ceLoss = ce(pre, labels)
        
        return ceLoss, t.tensor(0.0).to(ceLoss.device)

    def predict(self, adj, drugs, genes):
        embeds,_,_ = self.forward(adj, 1.0)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        pre = self.classifierLayer(dEmbeds, gEmbeds)
        return pre
    
    def predict_with_bert_cached(self, adj, drugs, genes):
        """
        方案 A 优化版：使用预计算的 BERT 特征缓存进行预测，保留完整 BERT 维度
        """
        # 1. 获取 GCN 特征
        drug_gcn_embeds, gene_gcn_embeds = self.forward_gcn(adj)
        
        drug_gcn_feat = drug_gcn_embeds[drugs]
        gene_gcn_feat = gene_gcn_embeds[genes]
        gcn_feat = t.cat([drug_gcn_feat, gene_gcn_feat], dim=1)
        
        # 2. 从缓存中获取 BERT 特征
        batch_size = len(drugs)
        bert_feats = []
        
        for i in range(batch_size):
            drug_idx = drugs[i].item()
            gene_idx = genes[i].item()
            key = (drug_idx, gene_idx)
            
            if key in self.bert_features_cache:
                bert_feat = self.bert_features_cache[key]
                bert_feats.append(bert_feat)
            else:
                print(f"⚠️ Warning: BERT feature not found for ({drug_idx}, {gene_idx})")
                bert_feats.append(np.zeros(1024, dtype=np.float32))
        
        bert_feat = t.from_numpy(np.stack(bert_feats)).to(drugs.device)
        
        # 3. 融合（保留完整维度）
        fused_feat = self.bert_gcn_fusion(gcn_feat, bert_feat)  # [batch, 1280]
        
        # 4. 分类（直接使用融合后的完整特征）
        pre = self.classifierLayer(fused_feat, fused_feat)
        return pre

    def getEmbeds(self):
        self.unfreeze(self.gcnLayers)
        return t.concat([self.dEmbeds, self.gEmbeds], axis=0)

    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True

    def getGCN(self):
        return self.gcnLayers


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds, flag=True):
        if (flag):
            return t.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse_coo_tensor(newIdxs, newVals, adj.shape)


class ClassifierLayer(nn.Module):
    def __init__(self):
        super(ClassifierLayer, self).__init__()

        # 步骤 5: 动态调整分类器输入维度
        # 基础维度
        input_dim = args.latdim

        # 判断输入维度
        if args.use_bert_fusion == 1:
            # 方案 A：BERT-GCN 融合（保留完整 BERT 维度）
            # 融合后的特征维度是 1280 (256 GCN + 1024 BERT)
            # 由于 forward 中传入的是 (fused_feat, fused_feat)，实际拼接后是 1280 * 2 = 2560
            classifier_input_dim = 1280 * 2
            print(f"  ✓ 分类器输入维度: {classifier_input_dim} (BERT-GCN 融合，保留完整 BERT)")
        elif args.use_llm_embeddings:
            # 原有方案：GatedConcatFusion
            # 如果使用文本特征，经过 GatedConcatFusion 后，节点嵌入维度是 latdim * 2
            # 分类器输入是 (Drug || Gene)，所以总维度是 (latdim * 2) * 2 = latdim * 4
            classifier_input_dim = input_dim * 4
        else:
            # 仅结构: latdim + latdim = latdim * 2
            classifier_input_dim = input_dim * 2

        self.lin1 = nn.Linear(classifier_input_dim, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.concat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        ret = self.lin2(embeds)
        return ret