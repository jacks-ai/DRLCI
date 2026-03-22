import torch as t
from torch import nn
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, calcRegLoss
import numpy as np
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class SELayer(nn.Module):
    """Squeeze-Excitation 式通道门控（与 temp.py 一致），对节点特征逐维重标定。"""

    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        reduced = max(1, channel // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduced, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced, channel, bias=False),
            nn.Sigmoid(),
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        b, c = x.size()
        y = self.fc(x).view(b, c)
        return x * y.expand_as(x)


# --- 新增：特征级门控拼接层 (用于实体级文本融合) ---
class GatedConcatFusion(nn.Module):
    def __init__(self, latdim):
        super(GatedConcatFusion, self).__init__()
        self.output_dim = latdim * 2

        self.ln = nn.LayerNorm(self.output_dim)

        self.gate = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Linear(self.output_dim // 2, self.output_dim),
            nn.Sigmoid()
        )

    def forward(self, struct_feat, text_feat):
        combined = t.cat([struct_feat, text_feat], dim=1)
        weights = self.gate(self.ln(combined))
        fused_embeds = combined * weights
        return fused_embeds

class JointGatedFusion(nn.Module):
    """   JointGatedFusion 会被用上  --use_joint_encoding false才会关掉
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
            if self.use_joint_encoding and getattr(args, 'joint_embed_train_path', None) and getattr(args,
                                                                                                     'joint_embed_test_path',
                                                                                                     None):
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

        # 图传播：由对称归一化 spmm（固定度权重）改为与 temp.Causal_GraphConvolution 一致的注意力聚合（可学习不等权）
        attn_dropout = getattr(args, 'gat_dropout', 0.0)
        self.gcnLayers = nn.Sequential(
            *[GraphAttentionConvolution(args.latdim, args.latdim, dropout=attn_dropout) for _ in range(args.gnn_layer)]
        )
        se_reduction = getattr(args, 'se_reduction', 16)
        self.seLayer = SELayer(args.latdim, reduction=se_reduction)
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
        print(f"  训练联合嵌入路径: {args.joint_embed_train_path}, 测试: {args.joint_embed_test_path}，，，，，，，，，，，，，，")
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

    def forward(self, adj, keepRate):
        # 步骤 3: GCN 传播 (结构视图)
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]

        for gcn in self.gcnLayers:
            gcn_emb = gcn(self.edgeDropper(adj, keepRate), embedsLst[-1])
            se_out = self.seLayer(gcn_emb)
            adjusted = t.add(gcn_emb, se_out)
            adjusted1 = t.add(gcn_emb, embedsLst[-1])
            struct_embeds = t.add(adjusted, adjusted1)
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
            gcn_emb = gcn(adj, embedsLst[-1])
            se_out = self.seLayer(gcn_emb)
            adjusted = t.add(gcn_emb, se_out)
            adjusted1 = t.add(gcn_emb, embedsLst[-1])
            embeds = t.add(adjusted, adjusted1)
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
            pre = self.classifierLayer(struct_d, struct_g, batch_joint)
        else:
            dEmbeds = embeds[:args.drug][drugs]
            gEmbeds = embeds[args.drug:][genes]
            pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)
        sslLoss = 0
        return ceLoss, sslLoss

    def predict(self, adj, drugs, genes):
        embeds, struct_view, _ = self.forward(adj, 1.0)
        if self.use_joint_encoding:
            struct_d = struct_view[:args.drug][drugs]
            struct_g = struct_view[args.drug:][genes]
            row_indices = [self.test_pair_to_row[(d.item(), g.item())] for d, g in zip(drugs.cpu(), genes.cpu())]
            row_indices = t.tensor(row_indices, dtype=t.long, device=embeds.device)
            batch_joint = self.joint_embeds_test.to(embeds.device)[row_indices]
            batch_joint = F.normalize(self.joint_text_proj(batch_joint), p=2, dim=1)
            pre = self.classifierLayer(struct_d, struct_g, batch_joint)
        else:
            dEmbeds = embeds[:args.drug][drugs]
            gEmbeds = embeds[args.drug:][genes]
            pre = self.classifierLayer(dEmbeds, gEmbeds)
        return pre

    def getEmbeds(self):
        self.unfreeze(self.gcnLayers)
        self.unfreeze(self.seLayer)
        return t.concat([self.dEmbeds, self.gEmbeds], axis=0)

    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True

    def getGCN(self):
        return self.gcnLayers


class GraphAttentionConvolution(nn.Module):
    """
    与 temp.py 中 Causal_GraphConvolution 相同的注意力机制：a、[Wh_i||Wh_j] 式打分 + 邻接掩码 + softmax，
    用学习到的权重对邻居做不等权聚合。不包含因果干预掩码，也不在注意力之后再乘一遍 adj（单次聚合）。
    """

    def __init__(self, in_features, out_features, dropout=0.0, score_act=F.relu):
        super(GraphAttentionConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.a = nn.Parameter(t.empty(size=(2 * out_features, 1)))
        self.dropout = dropout
        self.score_act = score_act
        self.weight = nn.Parameter(t.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        t.nn.init.xavier_uniform_(self.weight)
        t.nn.init.xavier_uniform_(self.a)

    def forward(self, adj, embeds, flag=True):
        embeds = F.dropout(embeds, self.dropout, training=self.training)
        Wh = t.mm(embeds, self.weight)
        Wh1 = t.matmul(Wh, self.a[: self.out_features, :])
        Wh2 = t.matmul(Wh, self.a[self.out_features :, :])
        e = Wh1 + Wh2.T
        e = self.score_act(e)

        zero_vec = -5e4 * t.ones_like(e)
        adj_dense = adj.to_dense() if adj.is_sparse else adj
        attention = t.where(adj_dense > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = t.matmul(attention, Wh)
        return h_prime


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

        # 联合编码: (struct_d, struct_g, joint_proj) -> latdim*3；实体级: (d, g) -> latdim*4；仅结构: latdim*2
        if getattr(args, 'use_joint_encoding', False):
            classifier_input_dim = input_dim * 3
            # 【代码可维护性优化】：在这里实例化联合编码专用门控
            print("使用JointGatedFusion 门控进行融合")
            self.joint_fusion = JointGatedFusion(input_dim)
        elif args.use_llm_embeddings:
            classifier_input_dim = input_dim * 4
            self.joint_fusion = None
        else:
            classifier_input_dim = input_dim * 2
            self.joint_fusion = None

        self.lin1 = nn.Linear(classifier_input_dim, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds, joint_embed=None):
        if joint_embed is not None:
            # 【平滑替换】：如果启用了门控融合，就走门控；否则退回原始的暴力拼接
            if self.joint_fusion is not None:
                joint_embed = self.joint_fusion(dEmbeds, gEmbeds, joint_embed)
                embeds = t.cat((dEmbeds, gEmbeds, joint_embed), dim=1)
            else:
                embeds = t.cat((dEmbeds, gEmbeds, joint_embed), dim=1)
        else:
            embeds = t.cat((dEmbeds, gEmbeds), dim=1)

        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        return self.lin2(embeds)