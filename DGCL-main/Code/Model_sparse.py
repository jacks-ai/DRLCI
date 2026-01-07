import torch as t
from torch import nn
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, calcRegLoss
import numpy as np
import torch_sparse
import torch.nn.functional as F
from torch.nn.parameter import Parameter

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class Causal_GraphConvolution(nn.Module):
    """
    因果图卷积层实现
    基于注意力机制的因果感知图卷积网络层，用于捕获节点间的因果关系

    参数:
    - in_features: 输入特征维度
    - out_features: 输出特征维度
    - dropout: dropout率
    - act: 激活函数
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu, ):
        super(Causal_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 注意力参数，大小为2倍输出维度，分别用于源节点和目标节点的注意力计算
        self.a = Parameter(t.empty(size=(2 * out_features, 1)))
        self.dropout = dropout
        self.act = act
        # 特征变换矩阵
        self.weight = nn.Parameter(t.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重参数"""
        t.nn.init.xavier_uniform_(self.weight)

    def causality_message(self):
        """
        预留的因果消息传递接口
        可用于实现更复杂的因果推理机制
        """
        pass

    def _prepare_causality_attention_input(self, Wh):
        """
        准备因果注意力的输入

        参数:
        - Wh: 经过线性变换后的节点特征

        返回:
        - 注意力分数矩阵

        实现步骤:
        1. 将注意力参数分为两部分，分别用于源节点和目标节点
        2. 计算两部分注意力分数
        3. 组合得到最终的注意力分数
        """
        # 计算源节点的注意力分数
        Wh1 = t.matmul(Wh, self.a[:self.out_features, :])
        # 计算目标节点的注意力分数
        Wh2 = t.matmul(Wh, self.a[self.out_features:, :])
        # 组合注意力分数（广播加法）
        e = Wh1 + Wh2.T
        return self.act(e)

    def forward(self, input, adj):
        """
        前向传播过程

        参数:
        - input: 输入特征
        - adj: 邻接矩阵

        返回:
        - 更新后的节点特征

        实现步骤:
        1. 应用dropout
        2. 对每个输入样本进行因果感知的消息传递
        3. 聚合所有消息并应用激活函数
        """
        # 应用dropout正则化
        input = F.dropout(input, self.dropout, self.training)

        # 因果感知的消息计算
        Whs = []
        # 对每个输入样本进行处理
        for inp in t.unbind(input, dim=0):
            # 线性特征变换
            Wh = t.mm(inp, self.weight)
            # 计算因果注意力分数
            e = self._prepare_causality_attention_input(Wh)

            # 创建注意力掩码，将不相连节点的注意力设为很小的负值
            zero_vec = -5e4 * t.ones_like(e)
            adj_dense = adj.to_dense()  # 将稀疏邻接矩阵转换为稠密矩阵
            # 应用邻接矩阵掩码，只保留相连节点间的注意力
            attention = t.where(adj_dense > 0, e, zero_vec)

            # 注意力归一化
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)
            # 基于注意力权重聚合邻居信息 这里就是计算出来的P
            h_prime = t.matmul(attention, Wh)
            Whs.append(h_prime)

        # 堆叠所有处理后的样本
        support = t.stack(Whs, dim=0)

        # 最终的图卷积操作：对每个样本应用邻接矩阵进行消息传递
        # 对输入张量 support 的每个切片进行稀疏矩阵乘法（SpMM），再将结果堆叠起来
        # unbind将张量 support 沿 dim=0（第0维）拆分为多个子张量（切片）
        # t.spmm(adj, sup)计算稀疏矩阵 adj 与稠密矩阵 sup 的乘积
        output = t.stack(
            [t.spmm(adj, sup) for sup in t.unbind(support, dim=0)],
            dim=0)

        # 应用激活函数
        output = self.act(output)
        return output

    def __repr__(self):
        """返回层的字符串表示"""
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.reset_parameters()

    def reset_parameters(self):
        # 初始化内部序列层的参数
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0)

# Adaptive Synergistic Fusion Module (ASFM)  自适应协同融合模块
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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 步骤 1: 初始化结构嵌入
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))  # 药物结构嵌入
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))  # 基因结构嵌入

        # 检查是否使用额外的文本特征
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

        # 初始化SE注意力层，用于特征重标定
        # 这里修改为权重shape随超参数embedding size变化
        # self.seLayer = SELayer(args.latdim)

        # 初始化因果图卷积层，用于捕获节点间的因果关系
        # self.causalGcnLayer = Causal_GraphConvolution(args.latdim, args.latdim)

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

    def predict(self, adj, drugs, genes):
        embeds,_,_ = self.forward(adj, 1.0)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

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

        # 步骤 5: 动态调整分类器输入维度
        # 基础维度
        input_dim = args.latdim

        # [关键修改]
        # 如果使用文本特征，经过 GatedConcatFusion 后，节点嵌入维度是 latdim * 2
        # 分类器输入是 (Drug || Gene)，所以总维度是 (latdim * 2) * 2 = latdim * 4
        if args.use_llm_embeddings:
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