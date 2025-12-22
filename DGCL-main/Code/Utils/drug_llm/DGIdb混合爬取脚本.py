import os
import time
import pandas as pd
import pubchempy as pcp
import requests
from chembl_webresource_client.new_client import new_client

# 初始化 ChEMBL 客户端
mechanism = new_client.mechanism
molecule = new_client.molecule


TRANSDUCTIVE_DIR = r"/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/transductive"
DRUGBANK_FEATURES_CSV = r"/mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/drug_text/drugbank_llm_features.csv"
OUTPUT_CSV = "/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/drug_text/mixed_drug_descriptions.csv"


def get_pubchem_description_via_api(cid):
    """
    通过 PubChem PUG REST API 获取详细的文本描述
    这是 pubchempy 库默认不提供的，必须单独请求
    """
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/description/JSON"
    try:
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if 'InformationList' in data and 'Information' in data['InformationList']:
                # 提取第一条非空的 Description
                infos = data['InformationList']['Information']
                for info in infos:
                    if 'Description' in info:
                        return info['Description']
    except Exception:
        pass
    return ""


def load_drug_ids_from_transductive(transductive_dir):
    """
    读取 transductive/train.csv 和 test.csv 的首列，收集所有药物 ID。
    """
    drug_ids = set()
    for filename in ("train.csv", "test.csv"):
        file_path = os.path.join(transductive_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                drug_id = line.split(",", 1)[0].strip()
                if drug_id:
                    drug_ids.add(drug_id)
    return sorted(drug_ids)


def load_drugbank_llm_text(csv_path, target_ids):
    """
    从预先解析好的 drugbank_llm_features.csv 读取 DrugBank 药物的描述。
    仅返回 target_ids 对应的内容，key 保持原 ID（不区分大小写）。
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 DrugBank 描述文件: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'DrugID' not in df.columns or 'LLM_Input_Text' not in df.columns:
        raise ValueError("CSV 中必须包含 'DrugID' 和 'LLM_Input_Text' 两列。")

    lookup = {str(row['DrugID']).upper(): row['LLM_Input_Text'] for _, row in df.iterrows()}
    descriptions = {}
    missing_ids = []
    for drug_id in target_ids:
        upper_id = drug_id.upper()
        text = lookup.get(upper_id)
        if text:
            descriptions[drug_id] = text
        else:
            missing_ids.append(drug_id)

    if missing_ids:
        raise ValueError(
            f"DrugBank CSV 中缺少 {len(missing_ids)} 个目标药物ID: {missing_ids[:10]}..."
        )

    return descriptions


def fetch_drug_info(query_id):
    """
    智能路由函数：根据 ID 类型自动选择数据源
    """
    query_id = str(query_id).strip()

    # === 1. 处理 ChEMBL ID ===
    if query_id.upper().startswith("CHEMBL"):
        try:
            # 获取名字
            mol = molecule.get(query_id)
            name = mol.get('pref_name', query_id) if mol else query_id

            # 获取机制
            mech_data = mechanism.filter(molecule_chembl_id=query_id)
            mech_texts = [m['mechanism_of_action'] for m in mech_data if m.get('mechanism_of_action')]
            mech_str = ". ".join(mech_texts)

            # 构造文本
            if mech_str:
                return f"Drug Name: {name}. Mechanism: {mech_str}"
            else:
                # 如果没有机制，返回名字
                return f"Drug Name: {name}. Source: ChEMBL."

        except Exception as e:
            return f"Drug Name: {query_id}. Error: {str(e)}"

    # === 2. 处理 PubChem CID (纯数字) 或 药物名称 ===
    else:
        cid = None
        input_name = query_id

        try:
            # A. 如果是纯数字，直接作为 CID
            if query_id.isdigit():
                cid = int(query_id)
            # B. 如果是名称 (如 XL-765)，先搜索 CID
            else:
                compounds = pcp.get_compounds(query_id, 'name')
                if compounds:
                    cid = compounds[0].cid
                    input_name = compounds[0].synonyms[0] if compounds[0].synonyms else query_id
                else:
                    return f"Drug Name: {query_id}. Info: Not found in PubChem."

            # C. 利用 CID 获取详细信息
            if cid:
                # 1. 获取基本属性
                c = pcp.Compound.from_cid(cid)
                name = c.synonyms[0] if c.synonyms else input_name

                # 2. [关键步骤] 获取详细文本描述 (Description)
                desc_text = get_pubchem_description_via_api(cid)

                # 3. 构造文本
                parts = [f"Drug Name: {name}."]
                if desc_text:
                    parts.append(f"Description: {desc_text}")
                else:
                    # 如果没有描述，用分子式兜底
                    parts.append(f"Formula: {c.molecular_formula}.")

                return " ".join(parts)

        except Exception as e:
            return f"Drug Name: {query_id}. Error: {str(e)}"


def main():
    drug_ids = load_drug_ids_from_transductive(TRANSDUCTIVE_DIR)
    print(f"📦 共收集到 {len(drug_ids)} 个药物 ID")

    db_ids = [drug_id for drug_id in drug_ids if drug_id.upper().startswith("DB")]
    drugbank_texts = {}
    if db_ids:
        print(f"🔍 正在从 DrugBank CSV 读取 {len(db_ids)} 个 DB 开头的药物描述...")
        drugbank_texts = load_drugbank_llm_text(DRUGBANK_FEATURES_CSV, db_ids)
        print(f"✅ 成功读取 {len(drugbank_texts)} 个 DrugBank 药物描述")

    results = []
    missing_ids = set()

    print("🚀 开始混合爬取药物描述...\n")
    for did in drug_ids:
        print(f"处理: {did} ...")
        if did.upper().startswith("DB"):
            text = drugbank_texts.get(did, "")
            if not text:
                missing_ids.add(did)
                text = f"Drug Name: {did}. Source: DrugBank. Description unavailable."
        else:
            text = fetch_drug_info(did)
            if not text or "Not found" in text or "Error" in text:
                missing_ids.add(did)
        results.append({'ID': did, 'LLM_Text': text})
        print(f" -> {text[:200]}...")
        print("-" * 30)
        time.sleep(1)

    df = pd.DataFrame(results).sort_values("ID")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ 所有数据已保存至 {OUTPUT_CSV}")

    if missing_ids:
        print(f"⚠️ 共 {len(missing_ids)} 个药物未能获取有效描述:")
        print(missing_ids)
    else:
        print("🎉 所有药物均获取到描述信息")


if __name__ == "__main__":
    main()