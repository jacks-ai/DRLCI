import torch as t
from torch import nn
import torch.nn.functional as F
from Params import args
from Utils.Utils import contrastLoss, ce, l2_norm

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Initialize drug and gene embeddings
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim)))
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim)))

        # Initialize GCN (Graph Convolutional Network) layer
        self.gcnLayer = GCNLayer()

        # Initialize HGNN (Hypergraph Neural Network) layer
        self.hgnnLayer = HGNNLayer()

        # Initialize classifier layer
        self.classifierLayer = ClassifierLayer()

        # Initialize trainable parametric matrices for drug-hyperedge matrix and gene-hyperedge matrix
        if args.dense:
            # nn.Parameter 将张量变成模型的参数，意味着在训练过程中，dHyper 和 gHyper 的值会被优化器更新
            self.dHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))
            self.gHyper = nn.Parameter(init(t.empty(args.latdim, args.hyperNum)))

        # Initialize SpAdjDropEdge layer for dropout on the graph edges
        self.edgeDropper = SpAdjDropEdge()

    # forward函数最后会用于embeds，与classifierLayer组合得到最后的预测结果
    def forward(self, adj, keepRate):
        # Concatenate drug and gene embeddings
        embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [embeds]
        gcnEmbedsLst = [embeds]
        hyperEmbedsLst = [embeds]

        # approximate the drug-hyperedge matrix and gene-hyperedge matrix
        ddHyper = self.dEmbeds * args.mult
        ggHyper = self.gEmbeds * args.mult

        # 如果开启密集运算模式
        if args.dense:
            # 在超边空间的表示=嵌入矩阵*投影矩阵  @符号是矩阵乘法
            ddHyper = self.dEmbeds @ self.dHyper
            ggHyper = self.gEmbeds @ self.gHyper

        for i in range(args.gnn_layer):
            # Perform GCN layer operation
            # 没有显式地指定要计算哪一层 但通过 embedsLst[-1] 这个引用，隐式地保证了每一层的计算顺序和上一层的输出被正确地传递给下一层
            # 邻接矩阵与节点嵌入作为输入
            gcnEmbeds = self.gcnLayer(self.edgeDropper(adj, keepRate), embedsLst[-1])

            # Perform HGNN layer operation  超图去掉
            # hyperDEmbeds = self.hgnnLayer(ddHyper, embedsLst[-1][:args.drug])
            # hyperGEmbeds = self.hgnnLayer(ggHyper, embedsLst[-1][args.drug:])
            # hyperEmbeds = t.concat([hyperDEmbeds, hyperGEmbeds], axis=0)

            # Append embeddings to lists
            gcnEmbedsLst.append(gcnEmbeds)
            # hyperEmbedsLst.append(hyperEmbeds)
            # embedsLst.append(gcnEmbeds + hyperEmbeds)
            embedsLst.append(gcnEmbeds)

        # Sum all embeddings
        embeds = sum(embedsLst)
        # return embeds, gcnEmbedsLst, hyperEmbedsLst
        return embeds

    # self.model.calcLosses(drugs, genes, labels, self.handler.torchBiAdj, args.keepRate)
    def calcLosses(self, drugs, genes, labels, adj, keepRate):
        embeds = self.forward(adj, keepRate)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Calculate Cross-Entropy loss
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)

        # Calculate Self-Supervised Learning (SSL) loss 对比去掉
        sslLoss = 0
        # for i in range(1, args.gnn_layer + 1, 1):
        #     # 局部嵌入不需要进行反向传播  每一层计算一次损失值
        #     embeds1 = gcnEmbedsLst[i].detach()
        #     embeds2 = hyperEmbedsLst[i]
        #     sslLoss += contrastLoss(embeds1[:args.drug], embeds2[:args.drug], t.unique(drugs),
        #                             args.temp) + contrastLoss(
        #         embeds1[args.drug:], embeds2[args.drug:], t.unique(genes), args.temp)
        return ceLoss

    # 预测出药物与基因关系
    def predict(self, adj, drugs, genes):
        embeds = self.forward(adj, 1.0)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Perform classification
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        return pre


# Define the GCN (Graph Convolutional Network) layer
# 简化版本的GCN 来捕获本地依赖关系
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        # spmm 稀疏矩阵乘法（Sparse Matrix Multiplication）
        return l2_norm(t.spmm(adj, embeds))


# Define the HGNN (Hypergraph Neural Network) layer
class HGNNLayer(nn.Module):
    def __init__(self):
        super(HGNNLayer, self).__init__()

    def forward(self, adj, embeds):
        lat = adj.T @ embeds
        ret = adj @ lat
        return l2_norm(ret)

# Define the SpAdjDropEdge layer for graph edge dropout
# 使用mask来消去一部分数据
class SpAdjDropEdge(nn.Module):
    def __init__(self):
        super(SpAdjDropEdge, self).__init__()

    def forward(self, adj, keepRate):
        if keepRate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        # floor()向下取整 结果为 0 或 1 然后将 0/1 转换为布尔掩码
        # torch.rand() 生成的是 [0, 1) 范围内的均匀分布随机数
        mask = ((t.rand(edgeNum) + keepRate).floor()).type(t.bool)
        newVals = vals[mask] / keepRate # 丢弃了部分元素后，剩余元素值需要进行缩放，以保持期望值不变
        newIdxs = idxs[:, mask]  # mask 中为 True 的位置对应的 idxs 索引将被保留
        return t.sparse.FloatTensor(newIdxs, newVals, adj.shape)


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
