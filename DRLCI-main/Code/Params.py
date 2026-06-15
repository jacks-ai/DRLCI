import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=None, type=float, help='learning rate（未指定时按数据集：DGIdb=5e-3, DrugBank=1e-3）')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--tstBat', default=100000, type=int, help='number of interactions in a testing batch')
    parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer 权重衰减正则化')
    parser.add_argument('--epoch', default=150, type=int, help='number of epochs')
    parser.add_argument('--iteration', type=int, default='7', help='iteration')
    parser.add_argument('--latdim', default=128, type=int, help='embedding size')
    parser.add_argument('--hyperNum', default=128, type=int, help='number of hyperedges')
    parser.add_argument('--gnn_layer', default=6, type=int, help='number of gnn layers')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--keepRate', default=0.75, type=float, help='ratio of edges to keep')
    parser.add_argument('--temp', default=0.1, type=float, help='temperature')
    parser.add_argument('--mult', default=1e-1, type=float, help='multiplication factor')
    parser.add_argument('--ssl_reg', default=1e-5, type=float, help='weight for ssl（Self-Supervised Learning） loss')
    parser.add_argument('--data', default='DGIdb', type=str, help='DrugBank DGIdb LINCS  name of dataset')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='0', type=int, help='indicates which gpu to use')
    parser.add_argument('--multi_gpu', action='store_true', default=False, help='use dual GPUs for parallel computation')
    parser.add_argument('--gpu_list', type=str, default='0', help='list of GPUs to use, separated by comma')
    parser.add_argument('--seed', default=43, type=int, help='seed')
    parser.add_argument('--is_debug', type=bool, default=False, help='is_debug')
    parser.add_argument('--dense', action='store_true', default=False, help='dense')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='if set , use validation mode which splits all relations into \
	                        train/val/test and evaluate on val only;\
	                        otherwise, use testing mode which splits all relations into train/test')
    parser.add_argument('--num_neg', type=int, default=100, help='生成全局负样本最大数量')
    # parser.add_argument('--num_hard_neg', type=int, default=20, help='选择困难负样本的数量')
    parser.add_argument('--num_two_hop', type=int, default=50, help='每个基因选择的二跳邻居数量')
    parser.add_argument('--num_neg_mul', type=int, default=None, help='num_neg_mul（自动根据数据集设置）')

    parser.add_argument('--one_hop_max_ratio', type=float, default=0.1, help='一跳邻居在困难负样本中的最大比例')

    parser.add_argument('--one_hop_weight', type=float, default=None, help='一跳困难负样本权重倍数（未指定时按数据集自动设置）')
    parser.add_argument('--two_hop_weight', type=float, default=None, help='二跳困难负样本权重倍数（未指定时按数据集自动设置）')
    parser.add_argument('--common_neg_weight', type=float, default=1.0, help='二跳困难负样本权重倍数')

    parser.add_argument(
        '--use_causal_intervention_mask',
        type=lambda v: str(v).lower() in {'1', 'true', 'yes'},
        default=True,
        help='是否对 GAT 注意力施加因果干预掩码（按一跳困难负样本切断药物–基因边）；False 时不掩码，与无干预前向一致'
    )

    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='梯度裁剪的最大范数')
    parser.add_argument('--score_clamp_min', type=float, default=-10.0, help='分数裁剪的最小值，防止exp爆炸')
    parser.add_argument('--score_clamp_max', type=float, default=10.0, help='分数裁剪的最大值，防止exp爆炸')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='数值稳定性的小值，防止除零和log(0)')

    parser.add_argument('--pretrained_drug_embed_path', type=str, default=None, help='预训练药物嵌入路径（自动根据数据集设置）')
    parser.add_argument('--pretrained_gene_embed_path', type=str, default=None, help='预训练基因嵌入路径（自动根据数据集设置）')
    parser.add_argument('--use_llm_embeddings', type=lambda v: str(v).lower() in {'1', 'true', 'yes'}, default=True,
                        help='是否启用预训练大模型嵌入； False 时改为随机初始化')
    parser.add_argument('--use_joint_encoding', type=lambda v: str(v).lower() in {'1', 'true', 'yes'}, default=True, # 联合编码（未指定时按数据集自动设置）
                        help='True=联合编码(joint_embeddings_*.npy)，False=实体级药物/基因嵌入')
    parser.add_argument('--joint_embed_train_path', type=str, default=None, help='联合编码训练集嵌入路径')
    parser.add_argument('--joint_embed_test_path', type=str, default=None, help='联合编码测试集嵌入路径')
    parser.add_argument('--log_dir', type=str, default="/mnt/data/huangpeng/DGCL/DGCL-main/log",
                        help='训练日志与结果统一保存目录')
    
    # 错误案例统计/导出控制开关
    parser.add_argument(
        '--enable_error_logging',
        type=lambda v: str(v).lower() in {'1', 'true', 'yes'},
        default=False,
        help='是否启用错误案例统计与CSV导出（默认关闭）'
    )
    
    # 方案 B：GCN + BioLinkBERT 融合配置
    parser.add_argument('--use_bert_fusion', type=int, default=0,
                        help='是否启用 GCN + BioLinkBERT 融合 (0=禁用, 1=启用方案B)')
    parser.add_argument('--bert_model_path', type=str, 
                        default="/mnt/data/huangpeng/DGCL/DGCL-main/Code/bert/best_biolinkbert_only_0208_110605.pt",
                        help='预训练的 BioLinkBERT 模型路径')
    parser.add_argument('--bert_cache_path', type=str,
                        default="/mnt/data/huangpeng/DGCL/mymodel/BioLinkBERT",
                        help='BioLinkBERT  预训练权重缓存路径')
    
    args = parser.parse_args()
    
    # 根据数据集自动设置学习率
    if args.lr is None:
        args.lr = 5e-3 if args.data == 'DGIdb' else 1e-3
    
    # 根据数据集自动设置是否使用联合编码：
    # DGIdb 上为 True，DrugBank 上为 False（除非命令行显式指定）
    if args.use_joint_encoding is None:
        args.use_joint_encoding = True if args.data == 'DGIdb' else False
    
    #  根据数据集自动设置嵌入路径  /mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/gene_text/bert_dgidb_gene_emd_cls.npy
    if args.pretrained_drug_embed_path is None:
        if args.data == 'DGIdb':
            args.pretrained_drug_embed_path = "/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/drug_text/ft_bert_dgidb_drug_emd_cls.npy"
        else:  # DrugBank
            args.pretrained_drug_embed_path = "/mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/drug_text/ft_bert_drugbank_drug_emd_cls.npy"
    
    if args.pretrained_gene_embed_path is None:
        if args.data == 'DGIdb':
            args.pretrained_gene_embed_path = "/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/gene_text/ft_bert_dgidb_gene_emd_cls.npy"
        else:  # DrugBank
            args.pretrained_gene_embed_path = "/mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/gene_text/ft_bert_drugbank_gene_emd_cls.npy"

    # 联合编码路径（use_joint_encoding=True 时使用） 最有效的文本嵌入
    if getattr(args, 'use_joint_encoding', False):
        if args.joint_embed_train_path is None:
            args.joint_embed_train_path = f"/mnt/data/huangpeng/DGCL/DGCL-main/Data/{args.data}/transductive/joint_embeddings_train.npy"
        if args.joint_embed_test_path is None:
            args.joint_embed_test_path = f"/mnt/data/huangpeng/DGCL/DGCL-main/Data/{args.data}/transductive/joint_embeddings_test.npy"

    #   根据数据集自动设置 num_neg_mul
    if args.num_neg_mul is None:
        if args.data == 'DrugBank':
            args.num_neg_mul = 0
        else:  # DGIdb
            args.num_neg_mul = 50
    
    # 根据数据集自动设置困难负样本权重（除非命令行显式指定）
    # DGIdb:  一跳=3.5, 二跳=2.5
    # DrugBank: 一跳=1.5, 二跳=2.5
    if args.one_hop_weight is None:
        args.one_hop_weight = 4.0 if args.data == 'DGIdb' else 1.5
    if args.two_hop_weight is None:
        args.two_hop_weight = 3.0
    
    return args


args = ParseArgs()
