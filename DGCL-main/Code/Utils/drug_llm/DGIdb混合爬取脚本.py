import os
import sys
import time
import atexit
import pandas as pd
from pandas.errors import EmptyDataError, ParserError
import pubchempy as pcp
import requests
from datetime import datetime
from chembl_webresource_client.new_client import new_client

# 初始化 ChEMBL 客户端
mechanism = new_client.mechanism
molecule = new_client.molecule

# ================= 配置路径 =================
TRANSDUCTIVE_DIR = r"/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/transductive"
DRUGBANK_FEATURES_CSV = r"/mnt/data/huangpeng/DGCL/DGCL-main/Data/DrugBank/drug_text/drugbank_llm_features.csv"
OUTPUT_CSV = "/mnt/data/huangpeng/DGCL/DGCL-main/Data/DGIdb/drug_text/mixed_drug_descriptions.csv"
LOG_DIR = "/mnt/data/huangpeng/DGCL/DGCL-main/Logs"

# 不允许写入 CSV 的错误提示摘要（可按需扩充）
DISALLOWED_OUTPUT_SNIPPETS = [
    ". Info:",  # 统一屏蔽 “Drug Name ... Info: ...” 的失败提示
    ". Error:",  # 屏蔽所有 “Drug Name ... Error: ...” 异常信息
    "No mapped CID found in PubChem",
    "Not found in PubChem",
    "Not found as CID or SID",
    "HTTPSConnectionPool",
    "Source: DrugBank. Description unavailable",
    "Source: ChEMBL (No Data)",
]

# ===========================================


def get_pubchem_description_via_api(cid):
    """
    通过 PubChem PUG REST API 获取详细的文本描述
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


def get_best_structural_representation(compound):
    """
    获取最佳的结构表示字段，替代复杂的 IUPAC 命名或简单的分子式。
    优先级：Isomeric SMILES > Canonical SMILES > Molecular Formula
    """
    try:
        # 优先获取带立体化学信息的 SMILES
        if getattr(compound, 'smiles', None):
            return f"SMILES: {compound.smiles}"
        elif getattr(compound, 'canonical_smiles', None):
            return f"SMILES: {compound.canonical_smiles}"
        elif getattr(compound, 'molecular_formula', None):
            return f"Formula: {compound.molecular_formula}"
    except Exception:
        pass
    return ""


def should_skip_output(text):
    """检测描述中是否包含禁止写入 CSV 的错误提示，返回 (是否跳过, 触发片段, 具体文本片段)"""
    if not text:
        return True, "empty text", ""
    upper_text = text.upper()
    for snippet in DISALLOWED_OUTPUT_SNIPPETS:
        snippet_upper = snippet.upper()
        idx = upper_text.find(snippet_upper)
        if idx != -1:
            detail = text[idx: idx + 200]
            return True, snippet, detail.strip()
    return False, "", ""


class TeeStream:
    """双向输出流：同时打印到控制台和写入日志文件"""

    def __init__(self, original, logfile):
        self.original = original
        self.logfile = logfile

    def write(self, data):
        if data:
            self.original.write(data)
            self.logfile.write(data)

    def flush(self):
        self.original.flush()
        self.logfile.flush()


def setup_run_logging():
    """初始化日志系统"""
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"dgidb_scrape_{timestamp}.log")
    log_file = open(log_path, "w", encoding="utf-8")

    sys.stdout = TeeStream(sys.stdout, log_file)
    sys.stderr = TeeStream(sys.stderr, log_file)

    def close_log():
        try:
            log_file.close()
        except Exception:
            pass

    atexit.register(close_log)
    return log_path


def load_drug_ids_from_transductive(transductive_dir):
    """读取训练/测试集中的药物ID"""
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
    """读取预处理好的 DrugBank 本地数据"""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到 DrugBank 描述文件: {csv_path}")

    # 跳过报错的行，并打印警告，这样你可以看到到底是哪几行出问题了
    df = pd.read_csv(csv_path, on_bad_lines='warn')
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
        print(f"⚠警告: DrugBank CSV 中缺少 {len(missing_ids)} 个目标药物ID。")
    else:
        print(f"✅ DrugBank 所有目标ID均在本地查到。")

    return descriptions


def fetch_drug_info(query_id):
    """
    智能路由函数：根据 ID 类型自动选择数据源
    【优化版逻辑】：
    1. ChEMBL: 获取 Name + Mechanism + SMILES
    2. PubChem: 获取 Name + Description + SMILES (全量获取，不再互斥)
    """
    query_id = str(query_id).strip()

    # === 1. 处理 ChEMBL ID ===
    if query_id.upper().startswith("CHEMBL"):
        try:
            # A. 获取分子基础信息
            mol = molecule.get(query_id)
            if not mol:
                return f"Drug Name: {query_id}. Source: ChEMBL (No Data)."

            name = mol.get('pref_name', query_id)

            # [新增] 尝试获取 SMILES 结构
            smiles = ""
            if mol.get('molecule_structures'):
                smiles = mol['molecule_structures'].get('canonical_smiles', "")

            # B. 获取机制
            mech_data = mechanism.filter(molecule_chembl_id=query_id)
            mech_texts = [m['mechanism_of_action'] for m in mech_data if m.get('mechanism_of_action')]
            mech_str = ". ".join(mech_texts)

            # C. 组装 Prompt
            parts = [f"Drug Name: {name}."]
            if mech_str:
                parts.append(f"Mechanism: {mech_str}")
            if smiles:
                parts.append(f"SMILES: {smiles}")

            # 兜底
            if len(parts) == 1:
                parts.append("Source: ChEMBL.")

            return " ".join(parts)

        except Exception as e:
            return f"Drug Name: {query_id}. Error: {str(e)}"

    # === 2. 处理 PubChem CID / SID / Name ===
    else:
        cid = None
        input_name = query_id
        is_sid = False

        try:
            # A. 尝试处理纯数字 ID (可能是 CID 也可能是 SID)
            if query_id.isdigit():
                input_id = int(query_id)
                try:
                    # 尝试作为 CID
                    c = pcp.Compound.from_cid(input_id)
                    _ = c.molecular_formula  # 触发验证
                    cid = input_id
                except (pcp.BadRequestError, pcp.NotFoundError, pcp.PubChemHTTPError):
                    # 尝试作为 SID
                    try:
                        print(f"   -> ID {input_id} 不是 CID，尝试作为 SID 查询...")
                        s = pcp.Substance.from_sid(input_id)
                        if s.standardized_cid:
                            cid = s.standardized_cid
                            is_sid = True
                            print(f"   -> 映射成功: SID {input_id} -> CID {cid}")
                        else:
                            return f"Drug Name: SID {input_id}. Info: No mapped CID found in PubChem."
                    except Exception:
                        return f"Drug Name: {input_id}. Info: Not found as CID or SID."

            # B. 处理药物名称
            else:
                compounds = pcp.get_compounds(query_id, 'name')
                if compounds:
                    cid = compounds[0].cid
                    input_name = compounds[0].synonyms[0] if compounds[0].synonyms else query_id
                else:
                    return f"Drug Name: {query_id}. Info: Not found in PubChem."

            # C. 利用 CID 提取全量信息 (Description + SMILES)
            if cid:
                c = pcp.Compound.from_cid(cid)
                name = c.synonyms[0] if c.synonyms else input_name

                # 1. 获取文本描述
                desc_text = get_pubchem_description_via_api(cid)

                parts = [f"Drug Name: {name}."]
                if is_sid:
                    parts.append(f"(Derived from SID {query_id})")

                # 2. [优化] 总是加入描述 (如果有)
                if desc_text:
                    parts.append(f"Description: {desc_text}")

                # 3. [优化] 总是加入结构 SMILES (无论有没有描述，作为最强补充)
                struct_info = get_best_structural_representation(c)
                if struct_info:
                    parts.append(struct_info)

                return " ".join(parts)

        except Exception as e:
            return f"Drug Name: {query_id}. Error: {str(e)}"


def main():
    log_path = setup_run_logging()
    print(f"📝 日志输出文件: {log_path}")

    # 1. 读取待处理 ID
    drug_ids = load_drug_ids_from_transductive(TRANSDUCTIVE_DIR)
    print(f"📦 共收集到 {len(drug_ids)} 个药物 ID")

    # 2. 读取 DrugBank 本地缓存
    db_ids = [drug_id for drug_id in drug_ids if drug_id.upper().startswith("DB")]
    drugbank_texts = {}
    if db_ids:
        print(f"🔍 正在从 DrugBank CSV 读取 {len(db_ids)} 个 DB 开头的药物描述...")
        try:
            drugbank_texts = load_drugbank_llm_text(DRUGBANK_FEATURES_CSV, db_ids)
            print(f"✅ 成功读取 {len(drugbank_texts)} 条缓存。")
        except Exception as e:
            print(f"⚠️ 读取 DrugBank CSV 失败: {e}")

    # 3. 断点续传逻辑
    processed_ids = set()

    if os.path.exists(OUTPUT_CSV):
        try:
            # 尝试读取，建议显式指定编码，防止中文乱码导致读取失败
            existing_df = pd.read_csv(OUTPUT_CSV, encoding='utf-8-sig')

            if 'DrugID' in existing_df.columns:
                processed_ids = set(existing_df['DrugID'].astype(str))
                print(f"🔄 检测到已存在结果，将跳过 {len(processed_ids)} 个已处理 ID。")
            else:
                print(f"文件存在但缺少 'DrugID' 列，将从头开始处理。")
                processed_ids = set()

        except EmptyDataError:
            print(f"读取失败：文件 {OUTPUT_CSV} 是空的。")
        except ParserError as e:
            print("读取失败：文件格式损坏或分隔符解析错误（ParserError）。")
            print(f"   详细错误信息: {e}")
            print("   建议：检查 CSV 末尾是否存在未完成的行，或尝试临时使用 pandas.read_csv(..., on_bad_lines='skip') 定位问题行。")
        except PermissionError:
            print(f"读取失败：没有权限读取该文件，请检查文件是否被 Excel 等其他程序占用。")
        except UnicodeDecodeError:
            print(f"读取失败：编码不匹配。请尝试更改 encoding 参数（如 'gbk' 或 'utf-8'）。")
        except Exception as e:
            # 捕获其他未预见的错误，并打印出具体的异常类型
            print(f"读取现有文件时发生未知错误类型 {type(e).__name__}: {e}")

    missing_ids = set()
    write_header = not os.path.exists(OUTPUT_CSV)

    print("\n🚀 开始混合爬取药物描述...\n")

    for did in drug_ids:
        if did in processed_ids:
            continue

        print(f"处理: {did} ...")

        skip_reason = ""

        # 策略 A: DrugBank (本地)
        if did.upper().startswith("DB"):
            text = drugbank_texts.get(did, "")
            if not text:
                missing_ids.add(did)
                text = f"Drug Name: {did}. Source: DrugBank. Description unavailable."

        # 策略 B: 网络爬取
        else:
            text = fetch_drug_info(did)
            if not text:
                missing_ids.add(did)

        # 实时保存 (Append Mode)
        skip, reason, detail = should_skip_output(text)
        if skip:
            missing_ids.add(did)
            detail_msg = detail if detail else text
            print(f"⚠️ 文本包含禁止写入的错误提示（{reason}）: {detail_msg}")
            print(f"⚠️ 跳过保存到 CSV: {did}")
            continue

        new_row = {'DrugID': did, 'LLM_Text': text}

        try:
            pd.DataFrame([new_row]).to_csv(OUTPUT_CSV, mode='a', header=write_header, index=False)
            write_header = False
            processed_ids.add(did)
        except Exception as e:
            print(f"❌ 写入文件失败: {e}")

        # 打印预览并休眠
        print(f" -> {text[:200]}...")
        print("-" * 30)
        time.sleep(1)  # ⚠ 必须保留，防止 API 封禁

    print(f"\n✅ 所有任务完成！数据已保存至 {OUTPUT_CSV}")
    if missing_ids:
        print(f"⚠ 共 {len(missing_ids)} 个药物未能获取有效描述。")
        print(sorted(missing_ids))
    else:
        print("🎉 完美！所有药物均获取到描述信息。")


if __name__ == "__main__":
    main()