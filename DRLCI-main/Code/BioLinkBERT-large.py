import os

# ========================================================
# 第一步：强制覆盖环境变量，彻底封杀 F 盘路径
# 必须写在导入 transformers 之前！
# ========================================================
custom_home = r"/mnt/data/huangpeng/DGCL/mymodel/BioLinkBERT"

os.environ['HF_HOME'] = custom_home
os.environ['HF_HUB_CACHE'] = custom_home
os.environ['TRANSFORMERS_CACHE'] = custom_home
os.environ['XDG_CACHE_HOME'] = custom_home
# 针对你报错中的 xet 错误，强制重定向日志
os.environ['HF_HUB_LOG_DIR'] = os.path.join(custom_home, "logs")

# 确保文件夹存在
if not os.path.exists(custom_home):
    os.makedirs(custom_home, exist_ok=True)
# ========================================================

from transformers import AutoTokenizer, AutoModel

model_name = "michiyasunaga/BioLinkBERT-large"

print(f"正在下载至: {custom_home}，请稍候...")

# 2. 此时再运行，程序会认为 D 盘才是它的“家”
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_home)
model = AutoModel.from_pretrained(model_name, cache_dir=custom_home)

# 3. 永久保存一份标准格式到该目录下
tokenizer.save_pretrained(custom_home)
model.save_pretrained(custom_home)

print("✅ 任务圆满完成！")