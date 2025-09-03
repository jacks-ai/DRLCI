import torch as t
import torch.nn.functional as F


def calcRegLoss(model):
    """
    L2 正则化常用于训练神经网络，以防止模型过拟合
    Calculate the regularization loss by summing the L2 norm of model parameters.

    Parameters:
    model (torch.nn.Module): The neural network model.

    Returns:
    ret (float): The regularization loss.
    """
    ret = 0
    for W in model.parameters():
        ret += W.norm(2).square()
    return ret


def contrastLoss(embeds1, embeds2, nodes, temp):
    """
    Calculate the contrastive loss for embeddings.

    Parameters:
    embeds1 (torch.Tensor): The first set of embeddings.
    embeds2 (torch.Tensor): The second set of embeddings.
    nodes (torch.Tensor): Indices of nodes used for contrastive loss.
    temp (float): Temperature parameter for the contrastive loss.

    Returns:
    loss (torch.Tensor): The contrastive loss.
    """
    # 对嵌入向量进行l2正则化
    # 在归一化以后，点积等于余弦相似度
    embeds1 = F.normalize(embeds1 + 1e-8, p=2)
    embeds2 = F.normalize(embeds2 + 1e-8, p=2)
    pckEmbeds1 = embeds1[nodes]  # 提取出指定节点的嵌入
    pckEmbeds2 = embeds2[nodes]
    # dim=-1表示直接计算的是向量的点积（两个向量的相似度）
    # pckEmbeds1 * pckEmbeds2是逐元素相乘
    nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
    # embeds2.T 表示 矩阵或向量的转置
    # 每一行表示每个节点与所有节点的相似度，然后求所有相似度总和
    deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1) + 1e-8
    # 表示计算平均值
    return -t.log(nume / deno).mean()


def ce(pred, target):
    """
    Calculate the cross-entropy loss between predicted and target values.

    Parameters:
    pred (torch.Tensor): Predicted values.
    target (torch.Tensor): Target values.

    Returns:
    loss (torch.Tensor): The cross-entropy loss.
    """
    return F.cross_entropy(pred, target)


def l2_norm(x):
    """
    Calculate L2 normalization of a tensor.

    Parameters:
    x (torch.Tensor): The input tensor.

    Returns:
    normalized_x (torch.Tensor): The L2 normalized tensor.
    """
    epsilon = t.FloatTensor([1e-12]).cuda() #保证l2范数不为0
    # 确保输出张量维度与原始张量一致
    # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.
    return x / (t.max(t.norm(x, dim=1, keepdim=True), epsilon))
