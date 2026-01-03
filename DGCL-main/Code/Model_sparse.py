import torch as t
from torch import nn
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, calcRegLoss
import numpy as np
from copy import deepcopy
import torch_sparse
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


# --- 特征级门控拼接层 (保持不变) ---
class GatedConcatFusion(nn.Module):
    def __init__(self, latdim):
        super(GatedConcatFusion, self).__init__()
        self.output_dim = latdim * 2

        # 加入 LayerNorm 增加数值稳定性，防止 NAN
        self.ln = nn.LayerNorm(self.output_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.output_dim // 2, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, struct_feat, text_feat):
        combined = t.cat([struct_feat, text_feat], dim=1)

        # 先过 LN 再进 Gate
        weights = self.gate(self.ln(combined))

        fused_embeds = combined * weights
        return fused_embeds


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 步骤 1: 初始化结构嵌入
        print("随机初始化，Initializing drug and gene embeddings randomly.")
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))

        self.use_text_features = (
                args.use_llm_embeddings
                and args.pretrained_drug_embed_path is not None
                and args.pretrained_gene_embed_path is not None
        )

        if self.use_text_features:
            print("加载预训练的LLM文本嵌入作为辅助特征...")
            drug_embeds_text = t.from_numpy(np.load(args.pretrained_drug_embed_path)).float()
            gene_embeds_text = t.from_numpy(np.load(args.pretrained_gene_embed_path)).float()

            assert drug_embeds_text.shape[0] == args.drug, "Drug count mismatch"
            assert gene_embeds_text.shape[0] == args.gene, "Gene count mismatch"

            drug_text_dim = drug_embeds_text.shape[1]
            gene_text_dim = gene_embeds_text.shape[1]

            # 使用 MLP 替代 Linear，没有作用，不要考虑
            # 投影层：将文本维度对齐到 args.latdim
            self.drug_text_proj = nn.Linear(drug_text_dim, args.latdim)
            self.gene_text_proj = nn.Linear(gene_text_dim, args.latdim)

            # 初始化权重
            nn.init.xavier_uniform_(self.drug_text_proj.weight)
            nn.init.xavier_uniform_(self.gene_text_proj.weight)

            self.register_buffer('drug_embeds_text', drug_embeds_text)
            self.register_buffer('gene_embeds_text', gene_embeds_text)

            self.fusion_layer = GatedConcatFusion(args.latdim)
            print("Gated Concat 融合模块及 MLP 投影层已初始化.")

        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
        self.classifierLayer = ClassifierLayer()
        self.edgeDropper = SpAdjDropEdge()

    def forward(self, adj, keepRate):
        # --- 视图 A: 结构视图 ---
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]

        for gcn in self.gcnLayers:
            struct_embeds = gcn(self.edgeDropper(adj, keepRate), embedsLst[-1])
            embedsLst.append(struct_embeds)

        final_struct_embeds = sum(embedsLst)

        # [关键修复] 对结构特征进行归一化，防止与文本特征量级差距过大导致 NAN
        final_struct_embeds = F.normalize(final_struct_embeds, p=2, dim=1)

        final_embeds = final_struct_embeds
        final_text_embeds = None  # 初始化变量

        # --- 视图 B: 文本视图 ---
        if self.use_text_features:
            drug_text_feat = self.drug_text_proj(self.drug_embeds_text.to(final_struct_embeds.device))
            gene_text_feat = self.gene_text_proj(self.gene_embeds_text.to(final_struct_embeds.device))

            # 归一化文本特征
            drug_text_feat = F.normalize(drug_text_feat, p=2, dim=1)
            gene_text_feat = F.normalize(gene_text_feat, p=2, dim=1)

            final_text_embeds = t.concat([drug_text_feat, gene_text_feat], axis=0)

            # --- 融合 ---
            final_embeds = self.fusion_layer(final_struct_embeds, final_text_embeds)

        # [修改返回值] 同时返回：融合后嵌入，纯结构嵌入，纯文本嵌入
        return final_embeds, final_struct_embeds, final_text_embeds

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
        # [解包] 获取三个返回值
        fused_embeds, struct_view, text_view = self.forward(adj, keepRate)

        # 1. 主任务：分类 Loss (使用融合后的特征)
        dEmbeds = fused_embeds[:args.drug]
        gEmbeds = fused_embeds[args.drug:]

        pre = self.classifierLayer(dEmbeds[drugs], gEmbeds[genes])
        ceLoss = ce(pre, labels)

        # 2. 辅助任务：SSL 对比学习 Loss (强制对齐)
        sslLoss = t.tensor(0.0).to(fused_embeds.device)

        if self.use_text_features and text_view is not None:
            # 这里的 struct_view 和 text_view 已经是全量节点且归一化过的

            # 找出当前 batch 涉及到的所有唯一节点，减少计算量
            # 或者你可以对全图计算 (更准但显存占用大)，这里演示只计算 batch 内的
            unique_nodes = t.unique(t.cat([drugs, genes]))

            # 计算对比损失 (InfoNCE)
            # View A -> View B
            loss1 = contrastLoss(struct_view, text_view, unique_nodes, temp=0.1)
            # View B -> View A (双向)
            loss2 = contrastLoss(text_view, struct_view, unique_nodes, temp=0.1)

            sslLoss = 0.5 * (loss1 + loss2)

        # 联合优化
        alpha = 0.05  # 控制 SSL 权重的超参数
        return ceLoss + alpha * sslLoss, sslLoss

    def predict(self, adj, drugs, genes):
        # [修正] 适配 forward 的返回值
        fused_embeds, _, _ = self.forward(adj, 1.0)

        dEmbeds = fused_embeds[:args.drug]
        gEmbeds = fused_embeds[args.drug:]

        pre = self.classifierLayer(dEmbeds[drugs], gEmbeds[genes])
        return pre

    def getEmbeds(self):
        # 简单返回结构嵌入用于后续分析，或者你可以返回融合后的
        return t.concat([self.dEmbeds, self.gEmbeds], axis=0)

    # ... 其他辅助函数 (unfreeze, getGCN 等) 保持不变 ...


# --- GCNLayer, SpAdjDropEdge 保持不变 ---
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
        if keepRate == 1.0: return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate
        newIdxs = idxs[:, mask]
        return t.sparse_coo_tensor(newIdxs, newVals, adj.shape)


# --- ClassifierLayer (保持维度修正) ---
class ClassifierLayer(nn.Module):
    def __init__(self):
        super(ClassifierLayer, self).__init__()
        input_dim = args.latdim

        # 因为 GatedConcatFusion 是拼接，维度是 latdim * 2
        # 分类器输入 (Drug || Gene)，总维度 latdim * 4
        if args.use_llm_embeddings:
            classifier_input_dim = input_dim * 4
        else:
            classifier_input_dim = input_dim * 2

        self.lin1 = nn.Linear(classifier_input_dim, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.concat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        ret = self.lin2(embeds)
        return ret