import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import pandas as pd
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader

class DataHandler:
    def __init__(self):
        self.data = args.data

    def map_data(self, data):
        """
        Map data to proper indices in case they are not in a continues [0, N) range
        Parameters
        ----------
        data : np.int32 arrays
        Returns
        -------
        mapped_data : np.int32 arrays
        n : length of mapped_data
        """
        # 将所有值去重
        uniq = list(set(data))

        # enumerate可以遍历出所有索引和值
        # old 是原始的唯一值（来自 uniq 列表），值->键
        # new 是新的整数编码（即 enumerate 生成的索引） 索引——>值
        id_dict = {old: new for new, old in enumerate(sorted(uniq))}
        # 所有值转换为对应下标
        data = np.array([id_dict[x] for x in data])
        n = len(uniq)

        return data, id_dict, n

    def load_data_from_database(self, dataset, mode='transductive', testing=True, relation_map=None,
                                post_relation_map=None):
        """
        将原始的类别数据转换为整数的编码形式 因为一些机器学习算法需要整数的形式
        Loads official train/test split and uses 10% of training samples for validaiton
        For each split computes 1-of-num_classes labels. Also computes training
        adjacency matrix. Assumes flattening happens everywhere in row-major fashion.
        """

        dtypes = {
            'd_nodes': np.str_, 'g_nodes': np.int32,
            'relations': np.float32}

        filename_train = '../Data/' + dataset + '/' + mode + '/train.csv'
        filename_test = '../Data/' + dataset + '/' + mode + '/test.csv'

        data_train = pd.read_csv(
            filename_train, header=None,
            names=['d_nodes', 'g_nodes', 'relations'], dtype=dtypes)

        data_test = pd.read_csv(
            filename_test, header=None,
            names=['d_nodes', 'g_nodes', 'relations'], dtype=dtypes)

        # 将这个 NumPy 数组 转换为一个 Python 列表
        # 等价于data_train.values
        data_array_train = data_train.values.tolist()
        # 将前一步转换得到的 Python 列表 再次转换为 NumPy数组
        data_array_train = np.array(data_array_train)
        data_array_test = data_test.values.tolist()
        data_array_test = np.array(data_array_test)
        # 将下载好的NumPy数组拼接起来
        data_array = np.concatenate([data_array_train, data_array_test], axis=0)

        # 从下载好的数据中将数据分割为三类，并将NumPy数组类型进行转换
        d_nodes_relations = data_array[:, 0].astype(dtypes['d_nodes'])
        g_nodes_relations = data_array[:, 1].astype(dtypes['g_nodes'])
        relations = data_array[:, 2].astype(dtypes['relations'])

        if relation_map is not None:
            for i, x in enumerate(relations):
                relations[i] = relation_map[x]

        d_nodes_relations, d_dict, num_drugs = self.map_data(d_nodes_relations)
        g_nodes_relations, g_dict, num_genes = self.map_data(g_nodes_relations)

        d_nodes_relations, g_nodes_relations = d_nodes_relations.astype(np.int64), g_nodes_relations.astype(np.int32)
        relations = relations.astype(np.float64)

        d_nodes = d_nodes_relations
        g_nodes = g_nodes_relations

        neutral_relation = -1  # int(np.ceil(np.float(num_classes)/2.)) - 1
        # assumes that relations_train contains at least one example of every relation type
        # 键值对倒转  值：键
        relation_dict = {r: i for i, r in enumerate(np.sort(np.unique(relations)).tolist())}

        # 初始化矩阵
        labels = np.full((num_drugs, num_genes), neutral_relation, dtype=np.int32)
        # d_nodes[i], g_nodes[i]本来就是存储的药物与基因的索引值，用来制定矩阵中哪些位置将被赋值
        # 遍历relations中的值，将interaction的索引值存储进labels
        labels[d_nodes, g_nodes] = np.array([relation_dict[r] for r in relations])

        # 这里for是遍历索引 检查一遍矩阵的赋值操作 防止计算错误出现
        # 如果不相等，断言会 抛出异常，并停止程序的执行，通常会显示出错误信息，表明在某个位置的值不符合预期
        for i in range(len(d_nodes)):
            assert (labels[d_nodes[i], g_nodes[i]] == relation_dict[relations[i]])

        # 将数组labels 重塑为一维数组，并根据原始数组的总元素数自动推断出新的大小
        # 转一维以后才能更好的拆分为训练 验证 测试三部分
        labels = labels.reshape([-1])

        # number of test and validation edges, see cf-nade code
        num_train = data_array_train.shape[0]
        num_test = data_array_test.shape[0]
        # 训练集中取20%作为验证集
        num_val = int(np.ceil(num_train * 0.2))
        num_train = num_train - num_val
        # zip可以将两个集合中的元素两两配对
        pairs_nonzero = np.array([[d, g] for d, g in zip(d_nodes, g_nodes)])
        # d * num_genes + g 这样可以将二维坐标对应到拉伸的一维坐标
        idx_nonzero = np.array([d * num_genes + g for d, g in pairs_nonzero])

        # 检查labels拉伸到一维后数据是否正确
        for i in range(len(relations)):
            assert (labels[idx_nonzero[i]] == relation_dict[relations[i]])

        # 二元组与一元组都要分割为训练集与测试集
        idx_nonzero_train = idx_nonzero[0:num_train + num_val]
        idx_nonzero_test = idx_nonzero[num_train + num_val:]

        pairs_nonzero_train = pairs_nonzero[0:num_train + num_val]
        pairs_nonzero_test = pairs_nonzero[num_train + num_val:]

        # Internally shuffle training set (before splitting off validation set)

        # 生成一个从 0 到 len(idx_nonzero_train)-1 的整数序列
        rand_idx = list(range(len(idx_nonzero_train)))
        # 用于设置随机数生成器的种子  确保每次运行代码时打乱的序列是相同的
        np.random.seed(42)
        # 打乱下标元素顺序
        np.random.shuffle(rand_idx)
        # 按照打乱下标进行重新排列
        idx_nonzero_train = idx_nonzero_train[rand_idx]
        pairs_nonzero_train = pairs_nonzero_train[rand_idx]

        # 打乱完以后又将两者再次拼接
        idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
        pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

        val_idx = idx_nonzero[0:num_val]
        train_idx = idx_nonzero[num_val:num_train + num_val]
        test_idx = idx_nonzero[num_train + num_val:]

        # 检查分割后长度是否一致
        assert (len(test_idx) == num_test)

        # 拆分为验证 训练 测试
        val_pairs_idx = pairs_nonzero[0:num_val]
        train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
        test_pairs_idx = pairs_nonzero[num_train + num_val:num_train + num_val + num_test]

        # 拆为药物与基因
        d_test_idx, g_test_idx = test_pairs_idx.transpose()
        d_val_idx, g_val_idx = val_pairs_idx.transpose()
        d_train_idx, g_train_idx = train_pairs_idx.transpose()

        # create labels
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        # 如果是验证模式，就合并索引
        if not args.validate:
            d_train_idx = np.hstack([d_train_idx, d_val_idx])
            g_train_idx = np.hstack([g_train_idx, g_val_idx])
            train_labels = np.hstack([train_labels, val_labels])
            # for adjacency matrix construction
            train_idx = np.hstack([train_idx, val_idx])

        # np.unique() 是去重加排序 存储所有interaction类别
        class_values = np.sort(np.unique(relations))

        # make training adjacency matrix
        # 初始化训练矩阵 一个大小为 num_drugs * num_genes 的零矩阵
        relation_mx_train = np.zeros(num_drugs * num_genes, dtype=np.float32)
        relation_mx_test = np.zeros(num_drugs * num_genes, dtype=np.float32)

        if post_relation_map is None:
            relation_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
            relation_mx_test[test_idx] = labels[test_idx].astype(np.float32) + 1.
        else:
            relation_mx_train[train_idx] = np.array(
                [post_relation_map[r] for r in class_values[labels[train_idx]]]) + 1.

        # 一维转二维 并转为CSR格式的稀疏矩阵
        relation_mx_train = sp.csr_matrix(relation_mx_train.reshape(num_drugs, num_genes))
        relation_mx_test = sp.csr_matrix(relation_mx_test.reshape(num_drugs, num_genes))

        # make external testing set
        # 设置'LINCS'为测试集，那么就将前面分割的测试集数据全部丢弃，从另一个文件读
        if dataset == 'LINCS':
            filename_external_test = '../Data/' + dataset + '/' + mode + '/external_test.csv'
            data_external_test = pd.read_csv(
                filename_external_test, header=None,
                names=['d_nodes', 'g_nodes', 'relations'], dtype=dtypes)
            data_array_external_test = data_external_test.values.tolist()
            data_array_external_test = np.array(data_array_external_test)

            d_nodes_external_relations = data_array_external_test[:, 0].astype(dtypes['d_nodes'])
            g_nodes_external_relations = data_array_external_test[:, 1].astype(dtypes['g_nodes'])

            external_test_relations = data_array_external_test[:, 2].astype(dtypes['relations'])
            external_test_relations = external_test_relations.astype(np.float64)

            d_external_test_nodes = d_nodes_external_relations
            g_external_test_nodes = g_nodes_external_relations

            d_external_test_idx = np.array([d_dict[d] for d in d_external_test_nodes])
            g_external_test_idx = np.array([g_dict[g] for g in g_external_test_nodes])
            external_test_labels = np.array([relation_dict[r] for r in external_test_relations])

            d_test_idx, g_test_idx, test_labels = d_external_test_idx, g_external_test_idx, external_test_labels

        return relation_mx_train, relation_mx_test, train_labels, d_train_idx, g_train_idx, \
            val_labels, d_val_idx, g_val_idx, test_labels, d_test_idx, g_test_idx, class_values

    def normalizeAdj(self, mat):
        """
        归一化邻接矩阵
        Normalize an adjacency matrix using the degree normalization technique.

        Parameters:
        mat (sparse matrix): The input adjac  ency matrix to be normalized.

        Returns:
        sparse matrix: The normalized adjacency matrix.
        """
        # 计算每一行的和，也就是每一个节点的度
        degree = np.array(mat.sum(axis=-1))
        # 对于每个节点的度 计算其倒数的平方根，转为一维
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        # 将度为0的节点无穷大值替换为0 isinf可以找出无穷大的值
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        # 创建一个对角矩阵 根据一维数组创建一个对角矩阵
        dInvSqrtMat = sp.diags(dInvSqrt)
        # 将结果矩阵转换为 COO（Coordinate）格式 也就是稀疏矩阵格式 dot是矩阵乘法 transpose()是转置
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        """
        Convert a SciPy sparse matrix into a PyTorch sparse tensor and apply normalization.

        Parameters:
        mat (sparse matrix): The input sparse matrix to be converted and normalized.

        Returns:
        torch.sparse.FloatTensor: A PyTorch sparse tensor with applied normalization.
        """
        # 创建一个大小为 args.drug*args.drug大小的稀疏矩阵
        a = sp.csr_matrix((args.drug, args.drug))
        b = sp.csr_matrix((args.gene, args.gene))
        # vstack()垂直方向（按行）拼接矩阵的函数    hstack按列拼接
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        # 将矩阵二值化大于1的值直接转为1
        mat = (mat != 0) * 1.0
        # eye函数创建一个矩阵（对角线元素为 1，其余元素为 0）
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)
        # make cuda tensor
        # row col属性只存在于coo稀疏矩阵  表示所有非零矩阵的横坐标与纵坐标
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        # 存储方式要比稠密矩阵高效的多
        # 输入是一个稀疏矩阵了。接着转一个稀疏张量
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def LoadData(self):
        """
        This method loads the dataset, preprocesses it, and creates data loaders for training and testing.
        """
        # 邻接矩阵 标签 坐标（药物 基因）
        relation_mx_train, relation_mx_test, train_labels, d_train_idx, g_train_idx, \
            val_labels, d_val_idx, g_val_idx, test_labels, d_test_idx, g_test_idx, class_values \
            = self.load_data_from_database(args.data)

        # Apply thresholding to the adjacency matrices
        # 将 trnMat 和 tstMat 中大于或等于 1 的元素设置为 1，其他的元素保留为 0
        trnMat, tstMat = relation_mx_train, relation_mx_test
        trnMat[trnMat >= 1] = 1
        tstMat[tstMat >= 1] = 1

        if type(trnMat) != coo_matrix:
            # 将 trnMat 转换为 稀疏坐标格式（COO 格式）
            trnMat = sp.coo_matrix(trnMat)
        if type(tstMat) != coo_matrix:
            tstMat = sp.coo_matrix(tstMat)

        args.drug, args.gene = trnMat.shape
        args.num_classes = len(class_values)
        # 存储处理好的稀疏张量
        self.torchBiAdj = self.makeTorchAdj(trnMat)

        trnData = TrnData(train_labels, d_train_idx, g_train_idx)
        # num_workers 指定了用于数据加载的子进程数量 0表示加载将在主线程中进行，不适用多线程
        # batch_size 是批量的大小  trnData已经打乱了，shuffle=False
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=False,
                                               num_workers=0)  # already shuffled training set
        if args.validate:
            tstData = TstData(val_labels, d_val_idx, g_val_idx)
        else:
            tstData = TstData(test_labels, d_test_idx, g_test_idx)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False,
                                               num_workers=0)


# Data loader for training data
class TrnData(data.Dataset):
    def __init__(self, train_labels, d_train_idx, g_train_idx):
        self.train_labels = train_labels
        self.d_train_idx = d_train_idx
        self.g_train_idx = g_train_idx

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        return self.d_train_idx[idx], self.g_train_idx[idx], self.train_labels[idx]


# Data loader for testing data
class TstData(data.Dataset):
    def __init__(self, test_labels, d_test_idx, g_test_idx):
        self.test_labels = test_labels
        self.d_test_idx = d_test_idx
        self.g_test_idx = g_test_idx

    def __len__(self):
        return len(self.test_labels)

    def __getitem__(self, idx):
        return self.d_test_idx[idx], self.g_test_idx[idx], self.test_labels[idx]
