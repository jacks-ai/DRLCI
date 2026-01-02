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

        # 步骤 1: 初始化结构嵌入
        # 无论是否使用LLM，都创建可学习的、随机初始化的嵌入，用于学习图结构信息
        print("随机初始化，Initializing drug and gene embeddings randomly.")
        self.dEmbeds = nn.Parameter(init(t.empty(args.drug, args.latdim))) # 药物结构嵌入
        self.gEmbeds = nn.Parameter(init(t.empty(args.gene, args.latdim))) # 基因结构嵌入

        # 检查是否使用额外的文本特征
        self.use_text_features = (
            args.use_llm_embeddings
            and args.pretrained_drug_embed_path is not None
            and args.pretrained_gene_embed_path is not None
        )

        if self.use_text_features:
            # 步骤 2: 如果使用LLM，则加载预训练嵌入作为静态特征
            print("加载预训练的LLM文本嵌入作为辅助特征...")
            drug_embeds_text = t.from_numpy(np.load(args.pretrained_drug_embed_path)).float()
            gene_embeds_text = t.from_numpy(np.load(args.pretrained_gene_embed_path)).float()

            # 验证节点数量是否匹配
            assert drug_embeds_text.shape[0] == args.drug, "Drug count mismatch in text embeddings"
            assert gene_embeds_text.shape[0] == args.gene, "Gene count mismatch in text embeddings"

            # 获取文本嵌入的原始维度
            drug_text_dim = drug_embeds_text.shape[1]
            gene_text_dim = gene_embeds_text.shape[1]

            # 定义线性投射层，将文本特征降维到与结构嵌入相同的维度 (args.latdim)
            self.drug_text_proj = nn.Linear(drug_text_dim, args.latdim)
            self.gene_text_proj = nn.Linear(gene_text_dim, args.latdim)

            # 初始化投射层权重
            nn.init.xavier_uniform_(self.drug_text_proj.weight)
            nn.init.xavier_uniform_(self.gene_text_proj.weight)

            # 将文本嵌入注册为模型的缓冲区(buffer)
            # 缓冲区是模型状态的一部分，但不会被视为模型参数，因此在训练中不会更新
            self.register_buffer('drug_embeds_text', drug_embeds_text)
            self.register_buffer('gene_embeds_text', gene_embeds_text)
            print("预训练文本嵌入加载成功.")

        # Initialize GCN (Graph Convolutional Network) layer
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])

        # Initialize classifier layer
        self.classifierLayer = ClassifierLayer()

        self.edgeDropper = SpAdjDropEdge()

    def forward(self, adj, keepRate):
        # 步骤 3: GCN传播，学习结构信息
        # 将可学习的结构嵌入拼接起来
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]
        
        # 通过多层GCN进行图卷积，聚合邻居信息
        for gcn in self.gcnLayers:
            struct_embeds = gcn(self.edgeDropper(adj, keepRate), embedsLst[-1])
            embedsLst.append(struct_embeds)
        
        # 将所有GCN层的输出求和，得到最终的结构嵌入
        final_struct_embeds = sum(embedsLst)

        # 步骤 4: 融合文本特征 (如果启用)
        if self.use_text_features:
            # 将文本特征通过投射层进行降维
            # .to(final_struct_embeds.device) 确保文本特征和结构嵌入在同一个设备上 (CPU or GPU)
            drug_text_feat = self.drug_text_proj(self.drug_embeds_text.to(final_struct_embeds.device))
            gene_text_feat = self.gene_text_proj(self.gene_embeds_text.to(final_struct_embeds.device))

            # L2归一化以防止数值不稳定
            drug_text_feat = F.normalize(drug_text_feat, p=2, dim=1)
            gene_text_feat = F.normalize(gene_text_feat, p=2, dim=1)

            all_text_feat = t.concat([drug_text_feat, gene_text_feat], axis=0)

            # 核心步骤：将结构嵌入和文本特征在特征维度上进行拼接
            final_embeds = t.concat([final_struct_embeds, all_text_feat], axis=1)
        else:
            # 如果不使用文本特征，则最终嵌入就是结构嵌入
            final_embeds = final_struct_embeds
            
        return final_embeds

    def forward_gcn(self, adj):
        # 该函数用于获取无dropout的“干净”嵌入，逻辑应与主forward函数保持一致
        # 1. GCN传播，学习结构信息
        struct_embeds = t.concat([self.dEmbeds, self.gEmbeds], axis=0)
        embedsLst = [struct_embeds]
        for gcn in self.gcnLayers:
            # 注意：这里直接使用adj，不经过edgeDropper
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        final_struct_embeds = sum(embedsLst)

        # 2. 融合文本特征 (如果启用)
        if self.use_text_features:
            drug_text_feat = self.drug_text_proj(self.drug_embeds_text.to(final_struct_embeds.device))
            gene_text_feat = self.gene_text_proj(self.gene_embeds_text.to(final_struct_embeds.device))

            # L2归一化以防止数值不稳定
            drug_text_feat = F.normalize(drug_text_feat, p=2, dim=1)
            gene_text_feat = F.normalize(gene_text_feat, p=2, dim=1)

            all_text_feat = t.concat([drug_text_feat, gene_text_feat], axis=0)
            # 拼接得到融合嵌入
            final_embeds = t.concat([final_struct_embeds, all_text_feat], axis=1)
        else:
            final_embeds = final_struct_embeds

        # 3. 返回与主forward函数格式一致的融合后嵌入
        return final_embeds[:args.drug], final_embeds[args.drug:]

    def calcLosses(self, drugs, genes, labels, adj, keepRate):
        # forward的输出已经改变，现在只返回一个融合后的嵌入张量
        embeds = self.forward(adj, keepRate)
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
        # forward的输出已经改变
        embeds = self.forward(adj, 1.0)
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


# Define the GCN (Graph Convolutional Network) layer 这里不要有LayerNorm层归一化，会导致NAN
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds, flag=True):
        if (flag):
            return t.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


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
        
        # 步骤 5: 动态调整分类器输入维度
        # 基础维度是结构嵌入的维度
        input_dim = args.latdim
        # 如果使用了文本特征，则每个节点的最终嵌入维度是 结构维度 + 文本维度
        if args.use_llm_embeddings:
            input_dim += args.latdim # 也可以写成 input_dim *= 2
            
        # 分类器接收的是拼接后的药物和基因嵌入，所以总维度是 input_dim * 2
        self.lin1 = nn.Linear(input_dim * 2, 128)
        self.lin2 = nn.Linear(128, args.num_classes)

    def forward(self, dEmbeds, gEmbeds):
        embeds = t.concat((dEmbeds, gEmbeds), 1)
        embeds = F.relu(self.lin1(embeds))
        embeds = F.dropout(embeds, p=0.4, training=self.training)
        ret = self.lin2(embeds)
        return ret
