import lxml.etree as ET
import pandas as pd
import time
import re


def parse_drugbank_ultimate_fallback(xml_file_path):
    """
    全能版 DrugBank XML 解析。
    策略：凑够 3 个字段即停止，优先级：
    1. Core (机理/适应症/代谢)
    2. Supp (描述/药效)
    3. Class (化学分类)
    4. Categories (药物类别 - 新增)
    5. SMILES (化学结构 - 新增)
    6. Groups (分组 - 新增)
    """
    print(f"🚀 开始解析文件: {xml_file_path} ...")
    start_time = time.time()

    # 1. 定义命名空间
    ns = {'db': 'http://www.drugbank.ca'}

    data_list = []

    # 统计计数器
    stats = {
        'total': 0,
        'success_ge_3': 0,  # 提取字段 >= 3
        'insufficient_lt_3': 0  # 提取字段 < 3
    }

    # 2. 使用 iterparse 增量解析
    context = ET.iterparse(xml_file_path, events=('end',))

    for event, elem in context:
        if elem.tag == f"{{{ns['db']}}}drug":
            stats['total'] += 1

            # --- A. 提取 DrugID ---
            drug_id = None
            for id_tag in elem.findall('db:drugbank-id', ns):
                if id_tag.get('primary') == 'true':
                    drug_id = id_tag.text
                    break

            if not drug_id:
                elem.clear()
                continue

            # --- 辅助函数：清洗文本 ---
            def clean_str(s):
                if not s: return ""
                # 去除换行和引用标签 [L1234]
                s = s.replace('\n', ' ').replace('\r', '').strip()
                s = re.sub(r'\[.*?\]', '', s).strip()
                return s

            def get_clean_text_from_node(tag_name):
                node = elem.find(f'db:{tag_name}', ns)
                if node is not None and node.text:
                    return clean_str(node.text)
                return ""

            name = get_clean_text_from_node('name')

            # 初始化 Prompt 构建
            text_parts = [f"Drug Name: {name}."]
            # 用于记录已添加的字段类型，防止重复（尽管当前逻辑通常不会重复）
            added_fields = set(['Name'])
            # 当前有效字段计数 (Name不算在凑够3个的任务里，还是算?
            # 原脚本逻辑：name是基础，count是统计后面附加的特征。这里保持原逻辑：used_fields_count 指的是描述性特征的数量)
            used_fields_count = 0

            raw_data = {'Name': name}

            # ============================================================
            # 优先级 1: 核心字段 (Core)
            # ============================================================
            core_fields = {
                'Mechanism of Action': 'mechanism-of-action',
                'Indication': 'indication',
                'Metabolism': 'metabolism'
            }
            for display, tag in core_fields.items():
                val = get_clean_text_from_node(tag)
                raw_data[f'Raw_{display.replace(" ", "_")}'] = val
                if val:
                    text_parts.append(f"{display}: {val}")
                    used_fields_count += 1
                    added_fields.add(display)

            # ============================================================
            # 优先级 2: 补充字段 (Supplementary)
            # ============================================================
            if used_fields_count < 3:
                supp_fields = {
                    'Description': 'description',
                    'Pharmacodynamics': 'pharmacodynamics'
                }
                for display, tag in supp_fields.items():
                    if used_fields_count >= 3: break
                    val = get_clean_text_from_node(tag)
                    raw_data[f'Raw_{display}'] = val
                    if val and display not in added_fields:
                        text_parts.append(f"{display}: {val}")
                        used_fields_count += 1
                        added_fields.add(display)

            # ============================================================
            # 优先级 3: 化学分类 (Classification)
            # ============================================================
            if used_fields_count < 3:
                # Classification 比较特殊，位于 classification/description
                class_node = elem.find('db:classification/db:description', ns)
                val = ""
                if class_node is not None and class_node.text:
                    val = clean_str(class_node.text)

                raw_data['Raw_Classification'] = val
                if val and 'Classification' not in added_fields:
                    text_parts.append(f"Classification: {val}")
                    used_fields_count += 1
                    added_fields.add('Classification')

            # ============================================================
            # 优先级 4: Categories (新增兜底)
            # ============================================================
            if used_fields_count < 3:
                # 提取 Categories，通常只需提取名字
                # XML结构: <categories><category><category>Name</category><mesh-id>...</mesh-id></category>...</categories>
                cats = []
                cat_nodes = elem.findall('db:categories/db:category/db:category', ns)
                for cat in cat_nodes:
                    if cat.text:
                        cats.append(clean_str(cat.text))

                if cats:
                    # 取前8个，防止过长
                    cats_str = ", ".join(cats[:8])
                    text_parts.append(f"Categories: {cats_str}")
                    used_fields_count += 1
                    added_fields.add('Categories')

            # ============================================================
            # 优先级 5: SMILES (新增兜底 - 化学结构)
            # ============================================================
            if used_fields_count < 3:
                smiles = ""
                # 遍历 calculated-properties 寻找 kind 为 SMILES 的
                props = elem.findall('db:calculated-properties/db:property', ns)
                for prop in props:
                    kind = prop.find('db:kind', ns)
                    if kind is not None and kind.text == "SMILES":
                        val_node = prop.find('db:value', ns)
                        if val_node is not None and val_node.text:
                            smiles = val_node.text.strip()
                            break

                if smiles:
                    text_parts.append(f"SMILES: {smiles}")
                    used_fields_count += 1
                    added_fields.add('SMILES')

            # ============================================================
            # 优先级 6: Groups (新增兜底 - 分组)
            # ============================================================
            if used_fields_count < 3:
                groups = []
                group_nodes = elem.findall('db:groups/db:group', ns)
                for g in group_nodes:
                    if g.text:
                        groups.append(clean_str(g.text))

                if groups:
                    groups_str = ", ".join(groups)
                    text_parts.append(f"Groups: {groups_str}")
                    used_fields_count += 1
                    added_fields.add('Groups')

            # --- 统计逻辑 ---
            if used_fields_count >= 3:
                stats['success_ge_3'] += 1
            else:
                stats['insufficient_lt_3'] += 1

            full_text_llm = " ".join(text_parts)

            # --- 存入 ---
            storage_data = {
                'DrugID': drug_id,
                'Name': name,
                'LLM_Input_Text': full_text_llm,
                'Field_Count': used_fields_count  # 方便后续分析
            }
            # 合并 Raw Data (如果有需要保留原始字段用于检查)
            # storage_data.update(raw_data)

            data_list.append(storage_data)

            if stats['total'] % 1000 == 0:
                print(f"✅ 已扫描 {stats['total']} 个药物...")

            # --- 清理内存 ---
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    end_time = time.time()
    print(f"\n🎉 解析完成！")
    print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")

    # --- 最终统计输出 ---
    print("\n" + "=" * 40)
    print("📊 字段提取统计报告")
    print("=" * 40)
    print(f"💊 总药物处理数: {stats['total']}")
    print(f"✅ 成功提取 >= 3 个字段的药物数: {stats['success_ge_3']}")
    print(f"⚠️ 提取字段 < 3 个 (信息不足) 的药物数: {stats['insufficient_lt_3']}")
    print("=" * 40 + "\n")

    return pd.DataFrame(data_list)


# ==========================================
# 运行部分
# ==========================================

# ⚠️ 请确认路径
xml_path = r"D:\桌面\研\drugbank\full database.xml"
output_csv = r"D:\桌面\研\论文\实验代码\DGCL-main\DGCL-main\Code\Utils\drugbank_llm_features_v2.csv"

# 1. 运行解析
try:
    df_drugbank = parse_drugbank_ultimate_fallback(xml_path)

    # 2. 预览
    print("\n--- [LLM 输入文本预览 (前2条)] ---")
    if not df_drugbank.empty:
        for i in range(min(2, len(df_drugbank))):
            row = df_drugbank.iloc[i]
            print(f"ID: {row['DrugID']} | Fields Found: {row['Field_Count']}")
            print(f"Content: {row['LLM_Input_Text'][:200]} ...\n")

    # 3. 保存
    df_drugbank.to_csv(output_csv, index=False)
    print(f"💾 数据已保存至: {output_csv}")

except FileNotFoundError:
    print("❌ 错误：找不到 XML 文件，请检查路径。")
except Exception as e:
    print(f"❌ 发生错误: {e}")