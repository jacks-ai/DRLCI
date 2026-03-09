import torch as t
from torch import nn
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, calcRegLoss
import numpy as np
import torch_sparse
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


# --- 新增：特征级门控拼接层 (Feature-wise Gated Concatenation) ---
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


class JointGatedFusion(nn.Module):
    """
    联合编码专用门控层：对 pair 级的 joint 文本向量做特征级门控，
    门控权重由 (struct_d, struct_g, joint_embed) 共同决定。 加了这个门控拟合慢一些了
    """
    def __init__(self, latdim):
        super(JointGatedFusion, self).__init__()
        self.input_dim = latdim * 3
        self.output_dim = latdim
        self.ln = nn.LayerNorm(self.input_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, self.input_dim // 2),
            nn.ReLU(),
            nn.Linear(self.input_dim // 2, self.output_dim),
            nn.Sigmoid(),
        )

    def forward(self, struct_d, struct_g, joint_embed):
        combined = t.cat([struct_d, struct_g, joint_embed], dim=1)
        weights = self.gate(self.ln(combined))
        return joint_embed * weights


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 步骤 1: 初始化结构嵌入
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))  # 药物结构嵌入
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))  # 基因结构嵌入

        # 联合编码 或 实体级文本：二选一
        self.use_joint_encoding = getattr(args, 'use_joint_encoding', False)
        self.use_text_features = False
        if args.use_llm_embeddings:
            if self.use_joint_encoding and getattr(args, 'joint_embed_train_path', None) and getattr(args, 'joint_embed_test_path', None):
                self.use_text_features = True
                self._init_joint_embeddings(args)
            elif not self.use_joint_encoding and args.pretrained_drug_embed_path and args.pretrained_gene_embed_path:
                self.use_text_features = True
                self._init_entity_embeddings(args)
        if not self.use_text_features:
            print("随机初始化，Initializing drug and gene embeddings randomly.")

        # (d_idx, g_idx) -> train/test 行号，由 Main 从 DataHandler 注入
        self.train_pair_to_row = None
        self.test_pair_to_row = None

        # GCN / 分类器 / DropEdge（所有分支共用）
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
        self.classifierLayer = ClassifierLayer()
        self.edgeDropper = SpAdjDropEdge()

    def _init_joint_embeddings(self, args):
        """联合编码：加载 joint_embeddings_train/test.npy，投影 1024 -> latdim"""
        print("加载联合编码文本嵌入 (joint_embeddings_*.npy)...")
        joint_train = t.from_numpy(np.load(args.joint_embed_train_path)).float()
        joint_test = t.from_numpy(np.load(args.joint_embed_test_path)).float()
        joint_dim = joint_train.shape[1]  # 1024
        self.joint_text_proj = nn.Linear(joint_dim, args.latdim)
        nn.init.xavier_uniform_(self.joint_text_proj.weight)
        self.register_buffer('joint_embeds_train', joint_train)
        self.register_buffer('joint_embeds_test', joint_test)
        # 针对联合编码的样本级门控层
        self.joint_fusion_layer = JointGatedFusion(args.latdim)
        print(f"  训练联合嵌入路径: {args.joint_embed_train_path}, 测试: {args.joint_embed_test_path}")
        print(f"  训练联合嵌入: {joint_train.shape}, 测试: {joint_test.shape}, 投影 -> {args.latdim}")

    def _init_entity_embeddings(self, args):
        """实体级：药物/基因分别嵌入 + 门控融合"""
        print("加载预训练的LLM文本嵌入作为辅助特征（实体级）...")
        drug_embeds_text = t.from_numpy(np.load(args.pretrained_drug_embed_path)).float()
        gene_embeds_text = t.from_numpy(np.load(args.pretrained_gene_embed_path)).float()
        assert drug_embeds_text.shape[0] == args.drug, "Drug count mismatch"
        assert gene_embeds_text.shape[0] == args.gene, "Gene count mismatch"
        self.drug_text_proj = nn.Linear(drug_embeds_text.shape[1], args.latdim)
        self.gene_text_proj = nn.Linear(gene_embeds_text.shape[1], args.latdim)
        nn.init.xavier_uniform_(self.drug_text_proj.weight)
        nn.init.xavier_uniform_(self.gene_text_proj.weight)
        self.register_buffer('drug_embeds_text', drug_embeds_text)
        self.register_buffer('gene_embeds_text', gene_embeds_text)
        self.fusion_layer = GatedConcatFusion(args.latdim)
        print("Gated Concat 融合模块已初始化.")

    #  这里只有图结构，做文本联合嵌入不好做
    def forward(self, adj, keepRate):
        # 步骤 3: GCN 传播 (结构视图)
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]

        for gcn in self.gcnLayers:
            struct_embeds = gcn(self.edgeDropper(adj, keepRate), embedsLst[-1])
            embedsLst.append(struct_embeds)

        final_struct_embeds = sum(embedsLst)

        all_text_feat = None
        # 步骤 4: 文本特征（联合编码在样本级做，不在此做节点融合）
        if self.use_text_features and not self.use_joint_encoding:
            drug_text_feat = self.drug_text_proj(self.drug_embeds_text.to(final_struct_embeds.device))
            gene_text_feat = self.gene_text_proj(self.gene_embeds_text.to(final_struct_embeds.device))
            drug_text_feat = F.normalize(drug_text_feat, p=2, dim=1)
            gene_text_feat = F.normalize(gene_text_feat, p=2, dim=1)
            all_text_feat = t.concat([drug_text_feat, gene_text_feat], axis=0)
            final_embeds = self.fusion_layer(final_struct_embeds, all_text_feat)
        else:
            final_embeds = final_struct_embeds

        return final_embeds, final_struct_embeds, all_text_feat

    def forward_gcn(self, adj):
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        final_struct_embeds = sum(embedsLst)

        if self.use_text_features and not self.use_joint_encoding:
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
        embeds, struct_view, text_view = self.forward(adj, keepRate)
        if self.use_joint_encoding:
            struct_d = struct_view[:args.drug][drugs]
            struct_g = struct_view[args.drug:][genes]
            row_indices = [self.train_pair_to_row[(d.item(), g.item())] for d, g in zip(drugs.cpu(), genes.cpu())]
            row_indices = t.tensor(row_indices, dtype=t.long, device=embeds.device)
            batch_joint = self.joint_embeds_train.to(embeds.device)[row_indices]
            batch_joint = F.normalize(self.joint_text_proj(batch_joint), p=2, dim=1)
            # 样本级门控：根据结构特征调整联合文本向量
            batch_joint = self.joint_fusion_layer(struct_d, struct_g, batch_joint)
            pre = self.classifierLayer(struct_d, struct_g, batch_joint)
        else:
            dEmbeds = embeds[:args.drug][drugs]
            gEmbeds = embeds[args.drug:][genes]
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

    def predict(self, adj, drugs, genes):
        embeds, struct_view, _ = self.forward(adj, 1.0) # 返回融合嵌入，结构嵌入，文本嵌入
        if self.use_joint_encoding:
            struct_d = struct_view[:args.drug][drugs]
            struct_g = struct_view[args.drug:][genes]
            row_indices = [self.test_pair_to_row[(d.item(), g.item())] for d, g in zip(drugs.cpu(), genes.cpu())]
            row_indices = t.tensor(row_indices, dtype=t.long, device=embeds.device)
            batch_joint = self.joint_embeds_test.to(embeds.device)[row_indices]
            batch_joint = F.normalize(self.joint_text_proj(batch_joint), p=2, dim=1)
            batch_joint = self.joint_fusion_layer(struct_d, struct_g, batch_joint)  # 先降维再融合
            pre = self.classifierLayer(struct_d, struct_g, batch_joint)
        else:
            dEmbeds = embeds[:args.drug][drugs]
            gEmbeds = embeds[args.drug:][genes]
            pre = self.classifierLayer(dEmbeds, gEmbeds)
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
        input_dim = args.latdim
        # 联合编码: (struct_d, struct_g, joint_proj) -> latdim*3；实体级: (d, g) 各 latdim*2 -> latdim*4；仅结构: latdim*2
        if getattr(args, 'use_joint_encoding', False):
            classifier_input_dim = input_dim * 3
        elif args.use_llm_embeddings:
            classifier_input_dim = input_dim * 4
        else:
            classifier_input_dim = input_dim * 2

        self.lin1 = nn.Linear(classifier_input_dim, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds, joint_embed=None):
        if joint_embed is not None:
            embeds = t.cat((dEmbeds, gEmbeds, joint_embed), dim=1)
        else:
            embeds = t.cat((dEmbeds, gEmbeds), dim=1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        return self.lin2(embeds)