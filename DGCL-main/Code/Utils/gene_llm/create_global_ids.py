import pandas as pd
import os
import json
import argparse
import hashlib


def compute_checksum(items):
    concatenated = '\n'.join(items)
    return hashlib.sha1(concatenated.encode('utf-8')).hexdigest()
# 第一步，生成包含数据集中所有唯一药物和基因的全局ID列表
def create_global_ids(args):
    print("=" * 80)
    print(f"开始为数据集 '{args.data}' 生成全局ID列表")
    print("=" * 80)

    base_path = os.path.join(args.data_dir, args.data, 'transductive')
    train_path = os.path.join(base_path, 'train.csv')
    test_path = os.path.join(base_path, 'test.csv')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"错误: 在 '{base_path}' 中未找到 train.csv 或 test.csv")
        return

    print(f"\n[1/3] 从以下文件读取数据:\n  - {train_path}\n  - {test_path}")

    try:
        df_train = pd.read_csv(train_path, header=None)
        df_test = pd.read_csv(test_path, header=None)
        df_combined = pd.concat([df_train, df_test], ignore_index=True)
    except Exception as e:
        print(f"读取CSV文件时出错: {e}")
        return

    print("\n[2/3] 提取、去重并排序ID...")

    # 提取药物ID（第0列）和基因ID（第1列）
    # 排序规则与 DataHandler.map_data 一致：
    # - 药物: DataHandler 中 d_nodes 为 np.str_，使用字符串排序
    # - 基因: DataHandler 中 g_nodes 为 np.int32，使用数值排序
    all_drug_ids = pd.unique(df_combined[0].astype(str)).tolist()
    all_drug_ids.sort()

    all_gene_ids_raw = pd.unique(df_combined[1]).tolist()
    all_gene_ids = [str(g) for g in sorted(all_gene_ids_raw, key=lambda x: int(x))]  # JSON 保存为字符串

    drug_checksum = compute_checksum(all_drug_ids)
    gene_checksum = compute_checksum(all_gene_ids)

    print(f"✓ 发现 {len(all_drug_ids)} 个唯一药物ID, checksum={drug_checksum}")
    print(f"✓ 发现 {len(all_gene_ids)} 个唯一基因ID, checksum={gene_checksum}")

    output_data = {
        'drug_ids': all_drug_ids,
        'gene_ids': all_gene_ids,
        'drug_ids_checksum': drug_checksum,
        'gene_ids_checksum': gene_checksum,
    }

    output_dir = os.path.join(args.data_dir, args.data)
    output_path = os.path.join(output_dir, 'global_ids_DGIdb.json')  # global_ids_2.json

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"\n[3/3] 保存全局ID列表到: {output_path}")
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2)
        print("✓ 保存成功！")
    except Exception as e:
        print(f"保存JSON文件时出错: {e}")

    print("\n" + "=" * 80)
    print("✓ 全局ID列表生成完毕！")
    print("=" * 80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从数据集中创建全局的药物和基因ID列表")
    parser.add_argument('--data_dir', type=str, 
                        default='/mnt/data/huangpeng/DGCL/DGCL-main/Data',
                        help='数据集的根目录')
    parser.add_argument('--data', type=str, default='DGIdb',
                        help='要处理的数据集名称 (例如: DrugBank, DGIdb)')
    
    args = parser.parse_args()
    create_global_ids(args)
