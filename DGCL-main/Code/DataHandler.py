import numpy as np
from scipy.sparse import coo_matrix
from Params import args
import scipy.sparse as sp
import pandas as pd
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import Counter
from multiprocessing import Pool, cpu_count
import time


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
        uniq = list(set(data))

        id_dict = {old: new for new, old in enumerate(sorted(uniq))}
        data = np.array([id_dict[x] for x in data])
        n = len(uniq)

        return data, id_dict, n

    def filter_data_by_count(self, data, max_count):
        """
        删除第一列出现次数大于指定次数的行。

        参数:
        data (np.ndarray): 输入的NumPy数组。
        max_count (int): 允许的最大出现次数。

        返回:
        np.ndarray: 删除了指定行的数组。
        """
        # 提取第一列
        first_column = data[:, 0]

        # 使用np.unique统计每个元素的出现次数
        unique_elements, counts = np.unique(first_column, return_counts=True)

        # 筛选出出现次数小于等于max_count的元素
        elements_to_keep = unique_elements[counts <= max_count]

        # 根据筛选结果删除相应的行
        filtered_data = data[np.isin(first_column, elements_to_keep)]

        return filtered_data

    def load_data_from_database(self, dataset, mode='transductive', testing=True, relation_map=None,
                                post_relation_map=None):
        """
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
        data_array_train = data_train.values.tolist()
        data_array_train = np.array(data_array_train)
        data_array_test = data_test.values.tolist()
        data_array_test = np.array(data_array_test)

        # print(data_array_test.shape[0])
        # data_array_test = self.filter_data_by_count(data_array_test, 10)
        # print(data_array_test.shape[0])

        data_array = np.concatenate([data_array_train, data_array_test], axis=0)

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
        relation_dict = {r: i for i, r in enumerate(np.sort(np.unique(relations)).tolist())}

        labels = np.full((num_drugs, num_genes), neutral_relation, dtype=np.int32)
        labels[d_nodes, g_nodes] = np.array([relation_dict[r] for r in relations])

        for i in range(len(d_nodes)):
            assert (labels[d_nodes[i], g_nodes[i]] == relation_dict[relations[i]])

        labels = labels.reshape([-1])

        # number of test and validation edges, see cf-nade code
        num_train = data_array_train.shape[0]
        num_test = data_array_test.shape[0]
        num_val = int(np.ceil(num_train * 0.2))
        num_train = num_train - num_val

        pairs_nonzero = np.array([[d, g] for d, g in zip(d_nodes, g_nodes)])
        idx_nonzero = np.array([d * num_genes + g for d, g in pairs_nonzero])

        for i in range(len(relations)):
            assert (labels[idx_nonzero[i]] == relation_dict[relations[i]])

        idx_nonzero_train = idx_nonzero[0:num_train + num_val]
        idx_nonzero_test = idx_nonzero[num_train + num_val:]

        pairs_nonzero_train = pairs_nonzero[0:num_train + num_val]
        pairs_nonzero_test = pairs_nonzero[num_train + num_val:]

        # Internally shuffle training set (before splitting off validation set)
        rand_idx = list(range(len(idx_nonzero_train)))
        np.random.seed(42)
        np.random.shuffle(rand_idx)
        idx_nonzero_train = idx_nonzero_train[rand_idx]
        pairs_nonzero_train = pairs_nonzero_train[rand_idx]

        idx_nonzero = np.concatenate([idx_nonzero_train, idx_nonzero_test], axis=0)
        pairs_nonzero = np.concatenate([pairs_nonzero_train, pairs_nonzero_test], axis=0)

        val_idx = idx_nonzero[0:num_val]
        train_idx = idx_nonzero[num_val:num_train + num_val]
        test_idx = idx_nonzero[num_train + num_val:]

        assert (len(test_idx) == num_test)

        val_pairs_idx = pairs_nonzero[0:num_val]
        train_pairs_idx = pairs_nonzero[num_val:num_train + num_val]
        test_pairs_idx = pairs_nonzero[num_train + num_val:num_train + num_val + num_test]

        d_test_idx, g_test_idx = test_pairs_idx.transpose()
        d_val_idx, g_val_idx = val_pairs_idx.transpose()
        d_train_idx, g_train_idx = train_pairs_idx.transpose()

        # create labels
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        test_labels = labels[test_idx]

        sorted_labels = sorted(Counter(train_labels).items(), key=lambda x: x[0])
        sorted_data = sorted(sorted_labels, key=lambda x: x[1])
        weights = {key: len(sorted_labels) - idx for idx, (key, _) in enumerate(sorted_data)}
        weighted_data = [(key, weights[key]) for key, _ in sorted_labels]

        # 只在args.device不存在时设置设备
        if not hasattr(args, 'device'):
            use_cuda = args.gpu >= 0 and t.cuda.is_available()
            device = 'cuda:{}'.format(args.gpu) if use_cuda else 'cpu'
            args.device = device

        args.class_weights = t.tensor(
            [value / sum([value for _, value in weighted_data]) for key, value in weighted_data]).to(args.device)

        if not args.validate:
            d_train_idx = np.hstack([d_train_idx, d_val_idx])
            g_train_idx = np.hstack([g_train_idx, g_val_idx])
            train_labels = np.hstack([train_labels, val_labels])
            # for adjacency matrix construction

            train_idx = np.hstack([train_idx, val_idx])

        class_values = np.sort(np.unique(relations))

        # make training adjacency matrix
        relation_mx_train = np.zeros(num_drugs * num_genes, dtype=np.float32)
        relation_mx_test = np.zeros(num_drugs * num_genes, dtype=np.float32)
        if post_relation_map is None:
            relation_mx_train[train_idx] = labels[train_idx].astype(np.float32) + 1.
            relation_mx_test[test_idx] = labels[test_idx].astype(np.float32) + 1.
        else:
            relation_mx_train[train_idx] = np.array(
                [post_relation_map[r] for r in class_values[labels[train_idx]]]) + 1.

        relation_mx_train = sp.csr_matrix(relation_mx_train.reshape(num_drugs, num_genes))
        relation_mx_test = sp.csr_matrix(relation_mx_test.reshape(num_drugs, num_genes))

        # make external testing set
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
        Normalize an adjacency matrix using the degree normalization technique.

        Parameters:
        mat (sparse matrix): The input adjacency matrix to be normalized.

        Returns:
        sparse matrix: The normalized adjacency matrix.
        """
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def makeTorchAdj(self, mat):
        """
        Convert a SciPy sparse matrix into a PyTorch sparse tensor and apply normalization.

        Parameters:
        mat (sparse matrix): The input sparse matrix to be converted and normalized.

        Returns:
        torch.sparse.FloatTensor: A PyTorch sparse tensor with applied normalization.
        """
        a = sp.csr_matrix((args.drug, args.drug))
        b = sp.csr_matrix((args.gene, args.gene))
        mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
        mat = (mat != 0) * 1.0
        mat = (mat + sp.eye(mat.shape[0])) * 1.0
        mat = self.normalizeAdj(mat)

        # make cuda tensor
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        # t.sparse.FloatTensor
        return t.sparse_coo_tensor(idxs, vals, shape).cuda()

    def LoadData(self):
        """
        This method loads the dataset, preprocesses it, and creates data loaders for training and testing.
        """
        relation_mx_train, relation_mx_test, train_labels, d_train_idx, g_train_idx, \
            val_labels, d_val_idx, g_val_idx, test_labels, d_test_idx, g_test_idx, class_values = self.load_data_from_database(
            args.data)

        # Apply thresholding to the adjacency matrices
        trnMat, tstMat = relation_mx_train, relation_mx_test
        trnMat_label = trnMat.copy()
        trnMat_label.data = trnMat_label.data - 1
        trnMat_label = trnMat_label.tocoo()

        trnMat[trnMat >= 1] = 1
        tstMat[tstMat >= 1] = 1

        if type(trnMat) != coo_matrix:
            trnMat = sp.coo_matrix(trnMat)
        if type(tstMat) != coo_matrix:
            tstMat = sp.coo_matrix(tstMat)
        args.drug, args.gene = trnMat.shape
        args.num_classes = len(class_values)
        self.torchBiAdj = self.makeTorchAdj(trnMat)

        trnData = TrnData(train_labels, d_train_idx, g_train_idx, trnMat, trnMat_label)
        self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=False,
                                               num_workers=0, )  # already shuffled training set
        if args.validate:
            tstData = TstData(val_labels, d_val_idx, g_val_idx)
        else:
            tstData = TstData(test_labels, d_test_idx, g_test_idx)
        self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)


# Data loader for training data
class TrnData(data.Dataset):
    def __init__(self, train_labels, d_train_idx, g_train_idx, coomat, coomat_label):
        self.train_labels = train_labels
        self.d_train_idx = d_train_idx
        self.g_train_idx = g_train_idx
        self.dokmat = coomat.todok()  # 稀疏矩阵(DOK格式)
        self.dokmat_label = coomat_label  # 带标签的稀疏矩阵
        self.negs = np.zeros((len(d_train_idx), args.num_neg)).astype(np.int32)

        self.negs_mul_gene_label = []
        self.negs_mul_gene =  []
        self.negs_mul_drug_label = []
        self.negs_mul_drug = []

    def negSampling(self):
        """
        多核并行优化的负采样函数：批量采样 + 向量化过滤 + 多进程并行
        性能提升：利用40个CPU核心并行处理
        """
        start_time = time.time()
        # 属性根本没有注入或为空
        if not hasattr(self, 'positive_genes_dict') or self.positive_genes_dict is None:
            raise RuntimeError(
                "positive_genes_dict 未设置。请先读取缓存并注入"
            )

        # 确定使用的进程数（最多使用35个核心，留5个给系统）
        num_processes = min(35, cpu_count())

        # 将样本分块，每块分配给一个进程
        total_samples = len(self.d_train_idx)
        chunk_size = max(1, total_samples // num_processes)

        # 准备并行处理的参数
        chunks = []
        for i in range(0, total_samples, chunk_size):
            end_idx = min(i + chunk_size, total_samples)
            chunk_drugs = self.d_train_idx[i:end_idx]
            chunk_indices = list(range(i, end_idx))
            chunks.append((chunk_drugs, chunk_indices, self.positive_genes_dict))

        # 并行处理
        parallel_start = time.time()
        with Pool(processes=num_processes) as pool:
            results = pool.map(self._process_chunk, chunks)

        # 合并结果
        for chunk_results, chunk_indices in results:
            for idx, neg_samples in enumerate(chunk_results):
                self.negs[chunk_indices[idx]] = neg_samples

        total_time = time.time() - start_time
        parallel_time = time.time() - parallel_start
        print(
            f"使用 {num_processes} 个CPU核心进行并行负采样,总耗时: {total_time:.2f}秒，并行处理: {parallel_time:.2f}秒")

    @staticmethod
    def _process_chunk(chunk_data):
        """
        静态方法：处理一个数据块的负采样
        用于多进程并行处理
        """
        chunk_drugs, chunk_indices, positive_genes_dict = chunk_data

        # 批量生成候选负样本的参数
        oversample_factor = 5  # 增加过采样因子以减少补充次数
        total_candidates = args.num_neg * oversample_factor

        chunk_results = []

        for drug in chunk_drugs:
            pos_genes = positive_genes_dict[drug]
            pos_genes_array = np.array(list(pos_genes))  # 转换为numpy数组提高效率

            # 批量生成候选负样本
            candidates = np.random.randint(0, args.gene, size=total_candidates)

            # 向量化过滤：移除正样本（优化版本）
            if len(pos_genes) > 0:
                mask = ~np.isin(candidates, pos_genes_array)
                valid_negatives = candidates[mask]
            else:
                valid_negatives = candidates

            # 如果有效负样本不够，继续生成（减少循环次数）
            max_attempts = 3  # 最多尝试3次
            attempt = 0
            while len(valid_negatives) < args.num_neg and attempt < max_attempts:
                needed = args.num_neg - len(valid_negatives)
                additional_size = max(needed * 3, total_candidates)  # 动态调整生成数量
                additional_candidates = np.random.randint(0, args.gene, size=additional_size)

                if len(pos_genes) > 0:
                    additional_mask = ~np.isin(additional_candidates, pos_genes_array)
                    additional_valid = additional_candidates[additional_mask]
                else:
                    additional_valid = additional_candidates

                valid_negatives = np.concatenate([valid_negatives, additional_valid])
                attempt += 1

            # 取前num_neg个作为最终的负样本
            if len(valid_negatives) >= args.num_neg:
                result = valid_negatives[:args.num_neg]
            else:
                # 如果还是不够，用重复采样填充（很少发生）
                result = np.pad(valid_negatives, (0, args.num_neg - len(valid_negatives)),
                                mode='wrap')[:args.num_neg]

            chunk_results.append(result)

        return chunk_results, chunk_indices

    def negMul_gene(self):
        for i in range(len(self.d_train_idx)):
            u = self.d_train_idx[i]
            mask = self.dokmat_label.row == u
            filtered_label = self.dokmat_label.data[mask]
            # filtered_drugs = self.dokmat_label.row[mask]
            filtered_genes = self.dokmat_label.col[mask]
            self.negs_mul_gene_label.append(filtered_label)
            self.negs_mul_gene.append(filtered_genes)

    def negMul_drug(self):
        for i in range(len(self.g_train_idx)):
            u = self.g_train_idx[i]
            mask = self.dokmat_label.col == u
            filtered_label = self.dokmat_label.data[mask]

            filtered_drugs = self.dokmat_label.row[mask]
            self.negs_mul_drug_label.append(filtered_label)
            self.negs_mul_drug.append(filtered_drugs)

    def padded_matrix(self):
        def pad_matrix(matrix, max_len):
            # 直接分配填充后的矩阵，并一次性填充
            padded_matrix = np.full((len(matrix), max_len), -1, dtype=int)
            for i, row in enumerate(matrix):
                padded_matrix[i, :len(row)] = row
            return padded_matrix

        # 计算 gene 的最大长度
        max_len_gene = max(
            max(len(row) for row in self.negs_mul_gene_label),
            max(len(row) for row in self.negs_mul_gene)
        )

        # 填充 negs_mul_gene_label 和 negs_mul_gene
        self.negs_mul_gene_label = pad_matrix(self.negs_mul_gene_label, max_len_gene)
        self.negs_mul_gene = pad_matrix(self.negs_mul_gene, max_len_gene)

        # 计算 drug 的最大长度
        max_len_drug = max(
            max(len(row) for row in self.negs_mul_drug_label),
            max(len(row) for row in self.negs_mul_drug)
        )

        # 填充 negs_mul_drug_label 和 negs_mul_drug
        self.negs_mul_drug_label = pad_matrix(self.negs_mul_drug_label, max_len_drug)
        self.negs_mul_drug = pad_matrix(self.negs_mul_drug, max_len_drug)

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        # 检查局部负样本列表是否已初始化（在预计算阶段这些列表为空）
        if len(self.negs_mul_gene_label) == 0:
            # 预计算阶段：只返回基本信息（4个元素）
            return self.d_train_idx[idx], self.g_train_idx[idx], self.train_labels[idx], self.negs[idx]
        else:
            # 训练阶段：返回完整信息（8个元素）
            return self.d_train_idx[idx], self.g_train_idx[idx], self.train_labels[idx], self.negs[idx], \
                self.negs_mul_gene_label[idx], self.negs_mul_gene[idx], self.negs_mul_drug_label[idx], \
            self.negs_mul_drug[idx]


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
