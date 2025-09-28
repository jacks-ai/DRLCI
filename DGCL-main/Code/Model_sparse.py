import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, calcRegLoss, innerProduct
import numpy as np
from copy import deepcopy
import torch_sparse
import time
import os
init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


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
