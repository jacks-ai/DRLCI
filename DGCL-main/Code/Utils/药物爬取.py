import pandas as pd
import json
import os


def extract_and_save_drug_texts(base_path):
    """
    根据 train.csv 中的药物ID，从 drugbank_llm_features.csv 中提取对应的文本描述，
    并保存为 JSON 文件。

    Args:
        base_path (str): 服务器上的项目根目录路径。
    """
    # 1. 定义文件路径
    train_csv_path = "/mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/transductive/train.csv"
    features_csv_path = os.path.join(base_path, 'drugbank_llm_features.csv')
    # output_dir = os.path.join(base_path, 'Data/DrugBank/drug_text_train')
    output_json_path = os.path.join(base_path, 'drug_text_train.json')

    print(f"源文件 (train): {train_csv_path}")
    print(f"特征文件: {features_csv_path}")
    print(f"输出路径: {output_json_path}")

    # 确保输出目录存在
    # os.makedirs(output_dir, exist_ok=True)

    try:
        # 2. 从 train.csv 读取所需的药物ID
        print("\n正在读取 train.csv 并提取唯一的药物ID...")
        train_df = pd.read_csv(train_csv_path, header=None)
        # 提取第一列（索引为0）的所有唯一药物ID
        drug_ids_needed = train_df[0].unique().tolist()
        print(f"✅ 成功提取 {len(drug_ids_needed)} 个唯一的药物ID。")
        print(f"示例ID: {drug_ids_needed[:5]}")

        # 3. 从 drugbank_llm_features.csv 读取特征数据
        print("\n正在加载药物特征文件 (drugbank_llm_features.csv)...")
        features_df = pd.read_csv(features_csv_path)
        print("✅ 特征文件加载完毕。")

        # 关键步骤：将DrugID设置为索引，极大提升查询效率
        features_df.set_index('DrugID', inplace=True)

        # 4. 筛选出需要的药物文本
        print("\n正在匹配ID并提取文本描述...")
        # 使用 .loc 和 reindex 高效筛选出所有需要的ID
        # reindex 会自动处理不在索引中的ID（结果为NaN），然后用 dropna 清除
        relevant_features = features_df.loc[features_df.index.isin(drug_ids_needed)]

        if 'LLM_Input_Text' not in relevant_features.columns:
            print("❌ 错误: 特征文件中找不到 'LLM_Input_Text' 列。")
            return

        # 5. 转换为字典格式 {DrugID: LLM_Input_Text}
        drug_text_dict = relevant_features['LLM_Input_Text'].to_dict()
        print(f"✅ 成功匹配到 {len(drug_text_dict)} 个药物的文本描述。")

        # 6. 保存结果到 JSON 文件
        print(f"\n正在保存结果到 {output_json_path} ...")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(drug_text_dict, f, ensure_ascii=False, indent=2)
        print("🎉 保存成功！")

        # 7. 打印结果示例
        print("\n--- 提取结果示例 ---")
        for i, (drug_id, text) in enumerate(list(drug_text_dict.items())[:3]):
            print(f"ID [{drug_id}]: {text[:150]}...")

    except FileNotFoundError as e:
        print(f"❌ 文件未找到错误: {e}")
        print("请确保您在服务器上运行此脚本，并且路径正确。")
    except Exception as e:
        print(f"❌ 发生未知错误: {e}")


if __name__ == '__main__':
    # ⚠️ 请确保这个路径是您在服务器上的项目根目录
    server_base_path = '/mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/drug_text_train'
    extract_and_save_drug_texts(server_base_path)