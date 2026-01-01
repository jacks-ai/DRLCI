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


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # Initialize drug and gene embeddings
        use_pretrained = (
            args.use_llm_embeddings
            and args.pretrained_drug_embed_path is not None
            and args.pretrained_gene_embed_path is not None
        )

        if use_pretrained:
            print("LLM初始化，Loading and projecting pretrained embeddings for drugs and genes...")
            drug_embeds_train_original = t.from_numpy(np.load(args.pretrained_drug_embed_path)).float()
            gene_embeds_train_original = t.from_numpy(np.load(args.pretrained_gene_embed_path)).float()

            # 验证节点数量是否匹配
            print(drug_embeds_train_original.shape[0])
            print(args.drug)
            assert drug_embeds_train_original.shape[0] == args.drug, "Drug count mismatch"
            assert gene_embeds_train_original.shape[0] == args.gene, "Gene count mismatch"

            # 动态获取预训练嵌入的维度
            drug_original_dim = drug_embeds_train_original.shape[1]
            gene_original_dim = gene_embeds_train_original.shape[1]

            # 定义线性投射层，将原始维度映射到模型潜在维度
            self.drug_proj = nn.Linear(drug_original_dim, args.latdim)
            self.gene_proj = nn.Linear(gene_original_dim, args.latdim)

            # 将原始嵌入通过投射层
            drug_embeds_projected = self.drug_proj(drug_embeds_train_original)
            gene_embeds_projected = self.gene_proj(gene_embeds_train_original)

            # 初始化投射层的权重
            nn.init.xavier_uniform_(self.drug_proj.weight)
            nn.init.xavier_uniform_(self.gene_proj.weight)

            # L2 Normalize the projected embeddings to control their magnitude  L2归一化
            normalized_dEmbeds = F.normalize(drug_embeds_projected, p=2, dim=1)
            normalized_gEmbeds = F.normalize(gene_embeds_projected, p=2, dim=1)

            self.dEmbeds = nn.Parameter(normalized_dEmbeds)
            self.gEmbeds = nn.Parameter(normalized_gEmbeds)
            print("Pretrained embeddings loaded, projected, and normalized successfully.")
        else:
            print("随机初始化，Initializing drug and gene embeddings randomly.")
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

        # Sequentially apply GCN layers, using the output of the previous layer as input for the next
        for gcn in self.gcnLayers:
            embeds = gcn(self.edgeDropper(adj, keepRate), embeds)

        # Use the output of the final GCN layer as the definitive embedding
        # The summation of all layers is removed as it causes numerical instability
        return embeds, None
    def forward_gcn(self, adj):
        embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)

        for gcn in self.gcnLayers:
            embeds = gcn(adj, embeds)

        mainEmbeds = embeds # 移除了不稳定的残差连接求和操作，改为使用最后一层 GCN 的输出,这里使用sum的话普通负采样都用不了

        return mainEmbeds[:args.drug], mainEmbeds[args.drug:]

    def calcLosses(self, drugs, genes, labels, adj, keepRate):
        embeds, gcnEmbedsLst = self.forward(adj, keepRate)
        dEmbeds, gEmbeds = embeds[:args.drug], embeds[args.drug:]

        # Select drug and gene embeddings based on input indices
        dEmbeds = dEmbeds[drugs]
        gEmbeds = gEmbeds[genes]

        # Calculate Cross-Entropy loss
        pre = self.classifierLayer(dEmbeds, gEmbeds)
        ceLoss = ce(pre, labels)

        # Calculate Self-Supervised Learning (SSL) loss
        sslLoss = 0


        return ceLoss, sslLoss
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
        self.layer_norm = nn.LayerNorm(args.latdim)

    def forward(self, adj, embeds, flag=True):
        if (flag):
            embeds = t.spmm(adj, embeds)
        else:
            embeds = torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)
        # Apply layer normalization for training stability  层归一化
        embeds = self.layer_norm(embeds)
        return embeds


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
        # Clamp the final scores to a safe range before they enter the loss function
        return t.clamp(ret, min=-10.0, max=10.0)
