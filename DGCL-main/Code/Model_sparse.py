import torch as t
from torch import nn
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, calcRegLoss
import numpy as np
import torch_sparse
import torch.nn.functional as F

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


# --- 新增：SE注意力层 (Squeeze-and-Excitation Layer) ---
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
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

    def forward(self, x):
        b, c = x.size()
        y = self.fc(x).view(b, c)
        return x * y.expand_as(x)


# --- 新增：因果图卷积层 (Causal Graph Convolution) ---
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

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(Causal_GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # 注意力参数，大小为2倍输出维度，分别用于源节点和目标节点的注意力计算
        self.a = nn.Parameter(t.empty(size=(2 * out_features, 1)))
        self.dropout = dropout
        self.act = act
        # 特征变换矩阵
        self.weight = nn.Parameter(t.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """初始化权重参数"""
        t.nn.init.xavier_uniform_(self.weight)

    def _prepare_causality_attention_input(self, Wh):
        """
        准备因果注意力的输入

        参数:
        - Wh: 经过线性变换后的节点特征

        返回:
        - 注意力分数矩阵
        """
        # 计算源节点的注意力分数
        Wh1 = t.matmul(Wh, self.a[:self.out_features, :])
        # 计算目标节点的注意力分数
        Wh2 = t.matmul(Wh, self.a[self.out_features:, :])
        # 组合注意力分数（广播加法）
        e = Wh1 + Wh2.T
        return self.act(e)

    def forward(self, input, adj, intervention_mask=None):
        """
        参数:
        - input: 输入特征
        - adj: 邻接矩阵
        - intervention_mask: [num_nodes, num_nodes] 因果干预掩码
                            值为0的位置表示需要切断的连接
                            值为1的位置表示保留的连接
        """
        input = F.dropout(input, self.dropout, self.training)

        Whs = []
        for inp in t.unbind(input, dim=0):
            Wh = t.mm(inp, self.weight)
            e = self._prepare_causality_attention_input(Wh)

            # 1. 应用邻接矩阵掩码（保留图结构）
            zero_vec = -5e4 * t.ones_like(e)
            adj_dense = adj.to_dense()
            attention = t.where(adj_dense > 0, e, zero_vec)

            # 2. 【关键】应用因果干预掩码（Do-operator）
            if intervention_mask is not None:
                # 将需要切断的连接（mask=0）设为 -inf
                # 这样 softmax 后这些位置的权重会变为 0
                attention = t.where(
                    intervention_mask > 0,  # mask=1 的位置保留
                    attention,  # 保留原始注意力分数
                    t.full_like(attention, float('-inf'))  # mask=0 的位置设为 -inf
                )

            # 3. 注意力归一化（-inf 会被 softmax 转为 0）
            attention = F.softmax(attention, dim=1)
            attention = F.dropout(attention, self.dropout, training=self.training)

            # 4. 基于干预后的注意力聚合信息
            h_prime = t.matmul(attention, Wh)
            Whs.append(h_prime)

        support = t.stack(Whs, dim=0)
        output = t.stack([t.spmm(adj, sup) for sup in t.unbind(support, dim=0)], dim=0)
        output = self.act(output)
        return output


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
            print("=" * 60)
            print("🔄 加载预训练的LLM文本嵌入作为辅助特征...")
            print("=" * 60)
            print(f"📁 药物嵌入文件路径: {args.pretrained_drug_embed_path}")
            print(f"📁 基因嵌入文件路径: {args.pretrained_gene_embed_path}")
            
            drug_embeds_text = t.from_numpy(np.load(args.pretrained_drug_embed_path)).float()
            gene_embeds_text = t.from_numpy(np.load(args.pretrained_gene_embed_path)).float()

            print(f"✅ 药物嵌入加载成功: 形状 {drug_embeds_text.shape} (期望: [{args.drug}, *])")
            print(f"✅ 基因嵌入加载成功: 形状 {gene_embeds_text.shape} (期望: [{args.gene}, *])")
            print("=" * 60)
            
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
        self.seLayer = SELayer(args.latdim)

        # 初始化因果图卷积层，用于捕获节点间的因果关系
        self.causalGcnLayer = Causal_GraphConvolution(args.latdim, args.latdim)

    def forward(self, adj, keepRate, intervention_mask=None):
        """
        前向传播（支持因果干预）
        
        参数:
        - adj: 邻接矩阵
        - keepRate: dropout保留率
        - intervention_mask: 因果干预掩码（可选）
        """
        # 步骤 3: GCN 传播 (结构视图)
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]
        gcnEmbedsLst = [struct_embeds]
        causalEmbedsLst = [struct_embeds]
        embeds_3d = t.unsqueeze(struct_embeds, 0)  # 在0维位置增加一个维度，用于因果GCN

        # 因果干预在这里实现
        for i in range(args.gnn_layer):
            # 1. 普通图卷积传递
            gcnEmbeds = self.gcnLayers[i](self.edgeDropper(adj, keepRate), embedsLst[-1])
            se_weights = self.seLayer(gcnEmbeds)
            gcnEmbedsLst.append(gcnEmbeds)

            # 2. 特征增强：结合SE注意力和残差连接   Global information modeling
            adjusted_embeds = t.add(gcnEmbeds, se_weights)
            # 这里是残差连接，将当前层输出与上一层输入相加
            adjusted_embeds1 = t.add(gcnEmbeds, embedsLst[-1])
            ae = t.add(adjusted_embeds, adjusted_embeds1)

            # 3. 因果感知的图卷积传递（应用干预掩码）
            gcnEmbeds_c = self.causalGcnLayer(embeds_3d, adj, intervention_mask)  # 传入干预掩码
            gcnEmbeds_c = gcnEmbeds_c.squeeze(0)  # 压缩维度
            causalEmbedsLst.append(gcnEmbeds_c)

            # 4. 因果特征增强：使用SE注意力进行特征重标定
            gcnEmbeds_c2 = self.seLayer(gcnEmbeds_c)
            gcnEmbeds_c = t.mul(gcnEmbeds_c2, gcnEmbeds_c)

            # 5. 特征融合：组合普通GCN和因果GCN的特征
            xsc = t.add(ae, gcnEmbeds_c)
            embedsLst.append(xsc)

        # 6. 聚合所有层的特征
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

        return final_embeds, final_struct_embeds, all_text_feat, gcnEmbedsLst, causalEmbedsLst

    def forward_gcn(self, adj, intervention_mask=None):
        """
        用于推理或无 Dropout 的前向传播（支持因果干预）
        
        参数:
        - adj: 邻接矩阵
        - intervention_mask: 因果干预掩码（可选，测试时通常不使用）
        """
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]
        gcnEmbedsLst = [struct_embeds]
        causalEmbedsLst = [struct_embeds]
        embeds_3d = t.unsqueeze(struct_embeds, 0)  # 在0维位置增加一个维度

        # 因果干预在这里实现
        for i in range(args.gnn_layer):
            # 1. 普通图卷积传递
            gcnEmbeds = self.gcnLayers[i](adj, embedsLst[-1])
            se_weights = self.seLayer(gcnEmbeds)
            gcnEmbedsLst.append(gcnEmbeds)

            # 2. 特征增强：结合SE注意力和残差连接
            adjusted_embeds = t.add(gcnEmbeds, se_weights)
            adjusted_embeds1 = t.add(gcnEmbeds, embedsLst[-1])
            ae = t.add(adjusted_embeds, adjusted_embeds1)

            # 3. 因果感知的图卷积传递（应用干预掩码）
            gcnEmbeds_c = self.causalGcnLayer(embeds_3d, adj, intervention_mask)
            gcnEmbeds_c = gcnEmbeds_c.squeeze(0)
            causalEmbedsLst.append(gcnEmbeds_c)

            # 4. 因果特征增强：使用SE注意力进行特征重标定
            gcnEmbeds_c2 = self.seLayer(gcnEmbeds_c)
            gcnEmbeds_c = t.mul(gcnEmbeds_c2, gcnEmbeds_c)

            # 5. 特征融合：组合普通GCN和因果GCN的特征
            xsc = t.add(ae, gcnEmbeds_c)
            embedsLst.append(xsc)

        # 6. 聚合所有层的特征
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

    def calcLosses(self, drugs, genes, labels, adj, keepRate, intervention_mask=None):
        """
        计算损失（支持因果干预）
        
        参数:
        - drugs: 药物索引
        - genes: 基因索引
        - labels: 标签
        - adj: 邻接矩阵
        - keepRate: dropout保留率
        - intervention_mask: 因果干预掩码（可选）
        """
        # 获取模型输出（传入干预掩码）
        embeds, struct_view, text_view, gcnEmbedsLst, causalEmbedsLst = self.forward(adj, keepRate, intervention_mask)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # 选择相关的药物和基因嵌入
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # 计算交叉熵损失
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)

        # 计算自监督对比学习损失
        # 使用普通GCN和因果GCN的输出进行对比学习
        sslLoss = 0
        for i in range(1, args.gnn_layer + 1, 1):
            # 获取普通GCN的特征（作为锚点）
            embeds1 = gcnEmbedsLst[i].detach()
            # 获取因果GCN的特征（作为正样本）
            embeds2 = causalEmbedsLst[i]

            # 分别计算药物和基因的对比损失
            sslLoss += contrastLoss(embeds1[:args.drug], embeds2[:args.drug], t.unique(drugs),
                                    args.temp) + contrastLoss(
                embeds1[args.drug:], embeds2[args.drug:], t.unique(genes), args.temp)

        return ceLoss, sslLoss

    def predict(self, adj, drugs, genes, intervention_mask=None):
        """
        预测（支持因果干预）
        
        参数:
        - adj: 邻接矩阵
        - drugs: 药物索引
        - genes: 基因索引
        - intervention_mask: 因果干预掩码（可选，测试时通常不使用）
        """
        embeds, _, _, _, _ = self.forward(adj, 1.0, intervention_mask)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Perform classification
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