import argparse


def ParseArgs():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=5e-3, type=float, help='learning rate')
    parser.add_argument('--batch', default=4096, type=int, help='batch size')
    parser.add_argument('--tstBat', default=100000, type=int, help='number of interactions in a testing batch')
    parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer 权重衰减正则化')
    parser.add_argument('--epoch', default=450, type=int, help='number of epochs')
    parser.add_argument('--latdim', default=128, type=int, help='embedding size')
    parser.add_argument('--hyperNum', default=128, type=int, help='number of hyperedges')
    parser.add_argument('--gnn_layer', default=4, type=int, help='number of gnn layers')
    parser.add_argument('--load_model', default=None, help='model name to load')
    parser.add_argument('--keepRate', default=0.75, type=float, help='ratio of edges to keep')
    parser.add_argument('--temp', default=0.1, type=float, help='temperature')
    parser.add_argument('--mult', default=1e-1, type=float, help='multiplication factor')
    parser.add_argument('--ssl_reg', default=1e-5, type=float, help='weight for ssl（Self-Supervised Learning） loss')
    parser.add_argument('--data', default='DGIdb', type=str, help='DrugBank DGIdb name of dataset')
    parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')
    parser.add_argument('--gpu', default='1', type=int, help='indicates which gpu to use')
    parser.add_argument('--multi_gpu', action='store_true', default=False, help='use dual GPUs for parallel computation')
    parser.add_argument('--gpu_list', type=str, default='1', help='list of GPUs to use, separated by comma')
    parser.add_argument('--seed', default=43, type=int, help='seed')
    parser.add_argument('--iteration', type=int, default='1', help='iteration')
    parser.add_argument('--is_debug', type=bool, default=False, help='is_debug')
    parser.add_argument('--dense', action='store_true', default=False, help='dense')
    parser.add_argument('--validate', action='store_true', default=False,
                        help='if set , use validation mode which splits all relations into \
	                        train/val/test and evaluate on val only;\
	                        otherwise, use testing mode which splits all relations into train/test')
    parser.add_argument('--num_neg', type=int, default=100, help='生成全局负样本最大数量')
    # parser.add_argument('--num_hard_neg', type=int, default=20, help='选择困难负样本的数量')
    parser.add_argument('--num_two_hop', type=int, default=50, help='每个基因选择的二跳邻居数量')
    parser.add_argument('--num_neg_mul', type=int, default= 50, help='num_neg_mul')

    parser.add_argument('--one_hop_max_ratio', type=float, default=0.1, help='一跳邻居在困难负样本中的最大比例')

    parser.add_argument('--one_hop_weight', type=float, default=3.0, help='一跳困难负样本权重倍数')
    parser.add_argument('--two_hop_weight', type=float, default=2.0, help='二跳困难负样本权重倍数')
    parser.add_argument('--common_neg_weight', type=float, default=1.0, help='二跳困难负样本权重倍数')

    parser.add_argument('--clip_grad_norm', type=float, default=5.0, help='梯度裁剪的最大范数')
    parser.add_argument('--score_clamp_min', type=float, default=-10.0, help='分数裁剪的最小值，防止exp爆炸')
    parser.add_argument('--score_clamp_max', type=float, default=10.0, help='分数裁剪的最大值，防止exp爆炸')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='数值稳定性的小值，防止除零和log(0)')

    parser.add_argument('--pretrained_drug_embed_path', type=str, default="/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/drug_text/bert_dgidb_emd_cls.npy", help='')
    #                                                                                  /mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/drug_text/bert_dgidb_emd_cls.npy
    #                                                                                  /mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/drug_text/bert_drugbank_emd_cls.npy
    parser.add_argument('--pretrained_gene_embed_path', type=str, default="/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/gene_text/bert_dgidb_gene_emd_cls.npy", help='')
    #                                                                                   /mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/gene_text/bert_dgidb_gene_emd_cls.npy
    #                                                                                   /mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/gene_text/bert_drugbank_gene_emd_cls.npy
    parser.add_argument('--use_llm_embeddings', type=lambda v: str(v).lower() in {'1', 'true', 'yes'}, default=False,
                        help='是否启用预训练大模型嵌入； False 时改为随机初始化')
    parser.add_argument('--log_dir', type=str, default="/mnt/data/huangpeng/DGCL/DGCL-main/log",
                        help='训练日志与结果统一保存目录')
    
    # 方案 B：GCN + BioLinkBERT 融合配置
    parser.add_argument('--use_bert_fusion', type=int, default=1,
                        help='是否启用 GCN + BioLinkBERT 融合 (0=禁用, 1=启用方案B)')
    parser.add_argument('--bert_model_path', type=str, 
                        default="/mnt/data/huangpeng/DGCL/DGCL-main/Code/bert/best_biolinkbert_only_0208_110605.pt",
                        help='预训练的 BioLinkBERT 模型路径')
    parser.add_argument('--bert_cache_path', type=str,
                        default="/mnt/data/huangpeng/DGCL/mymodel/BioLinkBERT",
                        help='BioLinkBERT  预训练权重缓存路径')
    
    return parser.parse_args()


args = ParseArgs()
