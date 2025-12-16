import lxml.etree as ET
import pandas as pd
import time


def parse_drugbank_selected_fields(xml_file_path):
    """
    高效解析 DrugBank XML，提取用于 Drug-Gene 预测的四大核心字段。
    """
    print(f"🚀 开始解析文件: {xml_file_path} ...")
    start_time = time.time()

    # 1. 定义命名空间 (DrugBank XML 的硬性要求)
    ns = {'db': 'http://www.drugbank.ca'}

    data_list = []

    # 2. 使用 iterparse 增量解析 (events='end' 表示在标签闭合时触发)
    context = ET.iterparse(xml_file_path, events=('end',))

    count = 0

    for event, elem in context:
        # 只在遇到 </drug> 结束标签时处理
        if elem.tag == f"{{{ns['db']}}}drug":

            # --- A. 提取 Primary DrugBank ID (连接键) ---
            drug_id = None
            # 遍历所有 ID，找到 primary 的那个
            for id_tag in elem.findall('db:drugbank-id', ns):
                if id_tag.get('primary') == 'true':
                    drug_id = id_tag.text
                    break

            # 如果没有 ID (极少情况)，跳过
            if not drug_id:
                elem.clear()
                continue

            # --- B. 提取四大核心字段 ---
            # 定义辅助函数：安全获取文本并去除换行符
            def get_clean_text(tag_name):
                node = elem.find(f'db:{tag_name}', ns)
                if node is not None and node.text:
                    # 去除多余的换行和空格，变成一行文本
                    return node.text.replace('\n', ' ').replace('\r', '').strip()
                return ""  # 如果没有内容，返回空字符串

            name = get_clean_text('name')
            mechanism = get_clean_text('mechanism-of-action')  # 核心：靶点信息
            metabolism = get_clean_text('metabolism')  # 核心：酶信息
            indication = get_clean_text('indication')  # 背景：疾病信息

            # --- C. 构造用于 LLM 的组合文本 (Prompt) ---
            # 只有当字段不为空时才加入，避免产生 "Mechanism: ." 这种无效句子
            text_parts = [f"Drug Name: {name}."]

            if indication:
                text_parts.append(f"Indication: {indication}")
            if mechanism:
                text_parts.append(f"Mechanism of Action: {mechanism}")
            if metabolism:
                text_parts.append(f"Metabolism: {metabolism}")

            # 将所有部分拼接成一个长字符串
            full_text_llm = " ".join(text_parts)

            # --- D. 存入数据 ---
            data_list.append({
                'DrugID': drug_id,
                'Name': name,
                'LLM_Input_Text': full_text_llm,  # 这一列直接喂给 OpenBioLLM
                # 以下列保留方便人工检查，训练时不需要
                'Raw_Mechanism': mechanism,
                'Raw_Metabolism': metabolism,
                'Raw_Indication': indication
            })

            count += 1
            if count % 1000 == 0:
                print(f"✅ 已处理 {count} 个药物...")

            # --- E. 关键步骤：清理内存 ---
            # 处理完一个 drug 标签后，将其从内存中清除
            elem.clear()
            # 还要清除根节点的引用，否则内存还是会缓慢增长
            while elem.getprevious() is not None:
                del elem.getparent()[0]

    end_time = time.time()
    print(f"\n🎉 解析完成！共处理 {count} 个药物。")
    print(f"⏱️ 耗时: {end_time - start_time:.2f} 秒")

    return pd.DataFrame(data_list)


# ==========================================
# 运行部分
# ==========================================

# ⚠️ 替换成你实际的 xml 文件名
xml_path = r"D:\桌面\研\drugbank\full database.xml"

# 1. 运行解析
try:
    df_drugbank = parse_drugbank_selected_fields(xml_path)

    # 2. 预览生成的 LLM 输入文本
    print("\n--- [Llama-3 输入文本预览] ---")
    if not df_drugbank.empty:
        sample_text = df_drugbank.iloc[0]['LLM_Input_Text']
        print(f"ID: {df_drugbank.iloc[0]['DrugID']}")
        print(f"Text Length: {len(sample_text.split())} words")
        print(f"Content:\n{sample_text[:500]} ...")  # 只打印前500字符

    # 3. 保存为 CSV (方便后续加载)
    output_csv = r"D:\桌面\研\论文\实验代码\DGCL-main\DGCL-main\Code\Utils\drugbank_llm_features.csv"
    df_drugbank.to_csv(output_csv, index=False)
    print(f"\n💾 数据已保存至: {output_csv}")

except FileNotFoundError:
    print("❌ 错误：找不到 XML 文件，请检查路径。")
except Exception as e:
    print(f"❌ 发生错误: {e}")