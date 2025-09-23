import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm, calcRegLoss, innerProduct
import numpy as np
from copy import deepcopy
import torch_sparse
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
        计算加权的负样本分数
        参数:
        drugEmbeds: [batch_size, embed_dim] - 药物嵌入
        hard_negEmbeds: [batch_size, num_hard_neg, embed_dim] - 困难负样本嵌入
        hard_prob: [batch_size, num_hard_neg] - 困难负样本概率分布

        返回:
        weighted_negScores: [batch_size, num_hard_neg] - 加权的负样本分数
        aggregated_negScore: [batch_size, 1] - 聚合的负样本分数
        """
        # 计算原始的负样本分数
        negScores = innerProduct(drugEmbeds.unsqueeze(1), hard_negEmbeds)  # [batch_size, num_hard_neg]
        print(f"Raw negScores stats: min={negScores.min():.6f}, max={negScores.max():.6f}, mean={negScores.mean():.6f}")

        # 对负样本分数应用指数函数增强差异
        negScores = t.exp(negScores)  # [batch_size, num_hard_neg]
        print(f"After exp negScores stats: min={negScores.min():.6f}, max={negScores.max():.6f}, mean={negScores.mean():.6f}")
        
        # 检查exp后是否有inf
        if t.isinf(negScores).any():
            print("WARNING: negScores contains Inf after exp!")
            print(f"Number of Inf values: {t.isinf(negScores).sum()}")

        # 将概率分布移到GPU并分离计算图
        hard_prob = hard_prob.detach().cuda()  # [batch_size, num_hard_neg]
        print(f"hard_prob stats: min={hard_prob.min():.6f}, max={hard_prob.max():.6f}, mean={hard_prob.mean():.6f}")

        # 方法1: 直接加权 - 每个负样本分数乘以其概率权重 更丰富的梯度信息，更好的对比学习效果 torch.Size([499, 10])
        weighted_negScores = negScores * hard_prob  # [batch_size, num_hard_neg]
        print(f"After weighting stats: min={weighted_negScores.min():.6f}, max={weighted_negScores.max():.6f}")
        
        # 求和，并保持原有维度数量不变
        weighted_negScores = weighted_negScores.sum(1, keepdim=True)
        print(f"After sum stats: min={weighted_negScores.min():.6f}, max={weighted_negScores.max():.6f}")
        
        weighted_negScores=weighted_negScores * args.num_hard_neg  # [batch_size, 1]
        print(f"Final weighted_negScores stats: min={weighted_negScores.min():.6f}, max={weighted_negScores.max():.6f}")
        
        # 检查最终结果
        if t.isnan(weighted_negScores).any():
            print("WARNING: Final weighted_negScores contains NaN!")
        if t.isinf(weighted_negScores).any():
            print("WARNING: Final weighted_negScores contains Inf!")

        # 加权平均导致稀释，需要乘以num_hard_neg

        # 方法2: 概率加权聚合 - 将所有负样本分数按概率加权求和
        # aggregated_negScore = torch.sum(negScores * hard_prob_detached, dim=1, keepdim=True)  # [batch_size, 1]
        print(f"Original negScores shape: {negScores.shape}")
        print(f"Weighted negScores shape: {weighted_negScores.shape}")
        return weighted_negScores

    #困难负样本损失
    def batch_bias_hard(self, drugEmbeds, posEmbeds,hard_negEmbeds, hard_prob):
        posScores = innerProduct(drugEmbeds.unsqueeze(1), posEmbeds.unsqueeze(1))  # [batch_size, 1]
        hard_negative_scores = self.compute_weighted_negative_scores(drugEmbeds, hard_negEmbeds, hard_prob) # [batch_size, 1]
        
        # 调试信息：检查每个步骤的值
        print(f"posScores stats: min={posScores.min():.6f}, max={posScores.max():.6f}, mean={posScores.mean():.6f}")
        print(f"hard_negative_scores stats: min={hard_negative_scores.min():.6f}, max={hard_negative_scores.max():.6f}, mean={hard_negative_scores.mean():.6f}")
        
        # 检查是否有NaN或无穷大
        if t.isnan(posScores).any():
            print("WARNING: posScores contains NaN!")
        if t.isnan(hard_negative_scores).any():
            print("WARNING: hard_negative_scores contains NaN!")
        if t.isinf(posScores).any():
            print("WARNING: posScores contains Inf!")
        if t.isinf(hard_negative_scores).any():
            print("WARNING: hard_negative_scores contains Inf!")
        
        # 计算分母，检查是否为0
        denominator = posScores + hard_negative_scores
        print(f"denominator stats: min={denominator.min():.6f}, max={denominator.max():.6f}")
        
        # 计算比值
        ratio = posScores / denominator
        print(f"ratio stats: min={ratio.min():.6f}, max={ratio.max():.6f}")
        
        # 检查比值是否有问题
        if t.isnan(ratio).any():
            print("WARNING: ratio contains NaN!")
        if (ratio <= 0).any():
            print("WARNING: ratio contains non-positive values!")
            print(f"Number of non-positive ratios: {(ratio <= 0).sum()}")
        
        # 添加数值稳定性处理
        ratio = t.clamp(ratio, min=1e-8, max=1.0)  # 防止log(0)和log(>1)
        
        loss = -1 * t.log(ratio).mean()
        
        print(f"Final loss: {loss.item():.6f}")
        return loss

    #计算交叉熵损失
    def calcLosses(self, drugs, genes, labels, adj, keepRate):
        embeds, gcnEmbedsLst = self.forward(adj, keepRate)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Calculate Cross-Entropy loss
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
