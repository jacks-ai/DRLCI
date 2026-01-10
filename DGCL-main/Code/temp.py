import torch as t
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm
import torch_sparse

init = nn.init.xavier_uniform_  # 使用的是 Xavier均匀分布初始化
uniformInit = nn.init.uniform


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

    def forward(self, x):
        b, c = x.size()
        # 检查 x 的形状，如果是 (batch_size, channel, height, width)，则先做平均池化
        #        if len(x.shape) == 4:
        #            x = F.adaptive_avg_pool2d(x, 1).view(b, c)
        # 否则，假定 x 形状是 (batch_size, channel)
        # 将输入 x 经过一个全连接层 fc 变换为形状为 (b, c) 的输出 y
        y = self.fc(x).view(b, c)
        return x * y.expand_as(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # 初始化药物和基因的嵌入向量
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))

        # 初始化普通图卷积层
        self.gcnLayer = GCNLayer()
        self.gcnLayer1 = GCNLayer1()

        # 初始化分类层
        self.classifierLayer = ClassifierLayer()

        # 初始化边dropout层，用于图结构的正则化
        self.edgeDropper = SpAdjDropEdge()

        # 初始化SE注意力层，用于特征重标定
        # 这里修改为权重shape随超参数embedding size变化
        self.seLayer = SELayer(args.latdim)

        # 初始化因果图卷积层，用于捕获节点间的因果关系
        self.causalGcnLayer = Causal_GraphConvolution(args.latdim, args.latdim)

    # 对比学习和最后的结果预测需要用到此结果
    def forward(self, adj, keepRate):
        if adj.is_sparse:
            adj_dropped = adj.to_dense()  # Convert sparse matrix to dense
        embeds = t.cat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [embeds]
        gcnEmbedsLst = [embeds]
        causalEmbedsLst = [embeds]
        embeds_3d = t.unsqueeze(embeds, 0)  # 在0维位置增加一个维度
        input = embeds_3d

        # 因果干预在这里实现
        for i in range(args.gnn_layer):
            # 1. 普通图卷积传递
            gcnEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), embedsLst[-1])
            se_weights = self.seLayer(gcnEmbeds)
            gcnEmbedsLst.append(gcnEmbeds)

            # 2. 特征增强：结合SE注意力和残差连接   Global information modeling
            adjusted_embeds = t.add(gcnEmbeds, se_weights)
            # 这里是残差连接，将当前层输出与上一层输入相加
            adjusted_embeds1 = t.add(gcnEmbeds, embedsLst[-1])
            ae = t.add(adjusted_embeds, adjusted_embeds1)

            # 3. 因果感知的图卷积传递
            gcnEmbeds_c = self.causalGcnLayer(embeds_3d, adj)  # 使用因果图卷积层处理
            gcnEmbeds_c = gcnEmbeds_c.squeeze(0)  # 压缩维度
            causalEmbedsLst.append(gcnEmbeds_c)

            # 4. 因果特征增强：使用SE注意力进行特征重标定
            gcnEmbeds_c2 = self.seLayer(gcnEmbeds_c)
            gcnEmbeds_c = t.mul(gcnEmbeds_c2, gcnEmbeds_c)

            # 5. 特征融合：组合普通GCN和因果GCN的特征
            xsc = t.add(ae, gcnEmbeds_c)
            embedsLst.append(xsc)

        # 6. 聚合所有层的特征
        embeds = sum(embedsLst)
        return embeds, gcnEmbedsLst, causalEmbedsLst

    def calcLosses(self, drugs, genes, labels, adj, keepRate):
        # 获取模型输出
        embeds, gcnEmbedsLst, causalEmbedsLst = self.forward(adj, keepRate)
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

    def predict(self, adj, drugs, genes):
        embeds, _, _ = self.forward(adj, 1.0)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Perform classification
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        return pre


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return l2_norm(t.spmm(adj, embeds))


#
##DrugBank:Best epoch :  98  , AUC :  0.6754987149675682
class GCNLayer1(nn.Module):
    def forward(self, adj, embeds, flag=True):
        if (flag):
            return t.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


# 这个模块并没有使用传统的因果推理工具（如 do-calculus、因果图等），而是采用了一种基于注意力机制的隐式因果推理方法：
# "因果"的概念在这里更多是通过网络结构和注意力机制来隐式表达，而不是通过显式的因果图或因果推理工具

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


# Define the SpAdjDropEdge layer for graph edge dropout
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
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings):
        # 归一化特征向量，以计算余弦相似度
        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = t.matmul(embeddings, embeddings.T) / self.temperature

        # 生成每个样本自己作为正样本的标签
        labels = t.arange(embeddings.size(0), device=embeddings.device)

        # 计算对比学习的损失
        return F.cross_entropy(similarity_matrix, labels)


class InfoNCELoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, embeddings, positive_indices, negative_indices):
        #        """
        #        embeddings: 全部嵌入向量，形状为 [batch_size, embedding_dim]
        #        positive_indices: 正样本索引列表
        #        negative_indices: 负样本索引列表，形状为 [batch_size, num_negatives]
        #        """
        anchor_embeddings = embeddings
        positive_embeddings = embeddings[positive_indices]

        # 获得负样本嵌入向量
        negative_embeddings = embeddings[negative_indices]

        # 计算anchor与正样本的相似度
        positive_similarity = t.sum(anchor_embeddings * positive_embeddings, dim=1)

        # 计算anchor与所有负样本的相似度
        negative_similarity = t.matmul(anchor_embeddings, negative_embeddings.transpose(0, 1))

        # 应用温度参数
        positive_similarity = positive_similarity / self.temperature
        negative_similarity = negative_similarity / self.temperature

        # 计算logits
        logits = t.cat([positive_similarity.unsqueeze(1), negative_similarity], dim=1)
        labels = t.zeros(logits.size(0), dtype=t.long, device=logits.device)

        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        return loss


# Define the ClassifierLayer for classification
class ClassifierLayer(nn.Module):
    def __init__(self):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(args.latdim * 2, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.cat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        ret = self.lin2(embeds)
        return ret
