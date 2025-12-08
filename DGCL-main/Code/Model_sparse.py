import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, calcRegLoss, innerProduct
import numpy as np
from copy import deepcopy
import torch_sparse
from torch.nn.parameter import Parameter
import time
import os
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
        # 输出 Whs 列表的最终长度和相关信息
        # print(f"[Causal_GraphConvolution] Whs final length: {len(Whs)}")
        # print(f"[Causal_GraphConvolution] Input shape: {input.shape}, batch size (dim=0): {input.shape[0]}")
        # if len(Whs) > 0:
        #     print(f"[Causal_GraphConvolution] First h_prime shape: {Whs[0].shape}")
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

        # Initialize drug and gene embeddings
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))

        # Initialize GCN (Graph Convolutional Network) layer
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        # Initialize classifier layer
        self.classifierLayer = ClassifierLayer()

        self.edgeDropper = SpAdjDropEdge()

        # 初始化SE注意力层，用于特征重标定
        # 这里修改为权重shape随超参数embedding size变化
        self.seLayer = SELayer(args.latdim)

        # 初始化因果图卷积层，用于捕获节点间的因果关系
        self.causalGcnLayer = Causal_GraphConvolution(args.latdim, args.latdim)

    def forward(self, adj, keepRate):
        # Concatenate drug and gene embeddings
        embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [embeds]
        gcnEmbedsLst = [embeds]
        # hyperEmbedsLst = [embeds]
        for gcn in self.gcnLayers:
            embeds = gcn(self.edgeDropper(adj, keepRate), embedsLst[-1])

            gcnEmbedsLst.append(embeds)
            embedsLst.append(embeds)
        # Sum all embeddings
        embeds = sum(embedsLst)
        return embeds, gcnEmbedsLst
    #用于模型生成药物与基因嵌入
    def forward_gcn(self, adj):
        iniEmbeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)

        return mainEmbeds[:args.drug], mainEmbeds[args.drug:]
    def compute_weighted_negative_scores(self, drugEmbeds, hard_negEmbeds, hard_prob):
        """
        计算加权的负样本分数（简化版本）
        """
        # 计算原始的负样本分数
        negScores = innerProduct(drugEmbeds.unsqueeze(1), hard_negEmbeds)  # [batch_size, num_hard_neg]

        # 对负样本分数应用指数函数增强差异
        negScores = t.exp(negScores)  # [batch_size, num_hard_neg]

        # 将概率分布移到GPU并分离计算图
        hard_prob = hard_prob.detach()
        if not hard_prob.is_cuda:
            hard_prob = hard_prob.cuda()  # [batch_size, num_hard_neg]

        # 直接加权 - 每个负样本分数乘以其概率权重
        weighted_negScores = negScores * hard_prob  # [batch_size, num_hard_neg]
        
        # 求和，并保持原有维度数量不变
        weighted_negScores = weighted_negScores.sum(1, keepdim=True)
        weighted_negScores = weighted_negScores * args.num_hard_neg  # [batch_size, 1]

        return weighted_negScores

    def compute_weighted_negative_scores_mixed(self, drugEmbeds, hard_negEmbeds, weights):
        """
        计算混合加权的负样本分数（支持一跳和二跳邻居不同权重）
        
        参数:
        drugEmbeds: [batch_size, 128] - 药物嵌入
        hard_negEmbeds: [batch_size, num_hard_neg, 128] - 困难负样本嵌入
        weights: [batch_size, num_hard_neg] - 每个负样本的权重
        """
        # 计算原始的负样本分数
        negScores = innerProduct(drugEmbeds.unsqueeze(1), hard_negEmbeds)  # [batch_size, num_hard_neg]

        # 对负样本分数应用指数函数增强差异
        negScores = t.exp(negScores)  # [batch_size, num_hard_neg]

        # 将权重移到GPU并分离计算图
        weights = weights.detach()
        if not weights.is_cuda:
            weights = weights.cuda()

        # 应用混合权重 - 每个负样本分数乘以其对应的权重
        weighted_negScores = negScores * weights  # [batch_size, num_hard_neg]
        
        # 求和，保持维度
        weighted_negScores = weighted_negScores.sum(1, keepdim=True)  # [batch_size, 1]

        return weighted_negScores

    def batch_bias_hard_mixed(self, drugEmbeds, posEmbeds, hard_negEmbeds, weights):
        """
        混合权重的困难负样本损失（支持一跳和二跳邻居不同权重）
        """
        # 对正样本嵌入进行调整
        adjusted_posEmbeds = posEmbeds - 0.01
        
        posScores = innerProduct(drugEmbeds.unsqueeze(1), adjusted_posEmbeds.unsqueeze(1))  # [batch_size, 1]
        hard_negative_scores = self.compute_weighted_negative_scores_mixed(drugEmbeds, hard_negEmbeds, weights) # [batch_size, 1]
        
        x = (posScores / (posScores + hard_negative_scores)).mean()
        loss = -t.log(x.clamp(min=1e-8))
        
        return loss

    #困难负样本损失
    def batch_bias_hard(self, drugEmbeds, posEmbeds, hard_negEmbeds, hard_prob):
        """
        困难负样本损失（简化版本）
        """
        # 对正样本嵌入进行调整，减少0.01来驱动负样本学习
        adjusted_posEmbeds = posEmbeds - 0.01
        
        posScores = innerProduct(drugEmbeds.unsqueeze(1), adjusted_posEmbeds.unsqueeze(1))  # [batch_size, 1]
        hard_negative_scores = self.compute_weighted_negative_scores(drugEmbeds, hard_negEmbeds, hard_prob) # [batch_size, 1]
        
        x = (posScores / (posScores + hard_negative_scores)).mean()
        loss = -t.log(x.clamp(min=1e-8))  # 只保留基本的数值稳定性
        
        return loss

    #计算交叉熵损失
    def calcLosses(self, drugs, genes, labels, adj, keepRate):
        embeds, gcnEmbedsLst = self.forward(adj, keepRate)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]  # ([4096, 128])
        gEmbeds = gEmbeds[genes]


        # Calculate Cross-Entropy loss   torch.Size([4096, 14]) 或 torch.Size([4096, 2])
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)

        return ceLoss
    def predict(self, adj, drugs, genes):
        embeds, _ = self.forward(adj, 1.0)
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

#Define the GCN (Graph Convolutional Network) layer
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
    def forward(self, adj, embeds, flag=True):
        if (flag):
            return t.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


# Define the GCN (Graph Convolutional Network) layer
# class GCNLayer(nn.Module):
#     def __init__(self):
#         super(GCNLayer, self).__init__()
#
#     def forward(self, adj, embeds):
#         return l2_norm(t.spmm(adj, embeds))


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
        return t.sparse_coo_tensor(newIdxs, newVals, adj.shape)


# Define the ClassifierLayer for classification
class ClassifierLayer(nn.Module):
    def __init__(self):
        super(ClassifierLayer, self).__init__()
        self.lin1 = nn.Linear(args.latdim * 2, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.concat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        ret = self.lin2(embeds)
        return ret
