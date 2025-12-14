import requests
from huggingface_hub import snapshot_download
import os
from huggingface_hub.constants import default_cache_path
# 使用这种官方给的下载方式AutoModelForCausalLM一直下不动
# 用snapshot_download 反而可以下载了（只下载不加载，这样就对显存没要求
# 把之前下载的清空一下，下载变快了

model_id = "aaditya/Llama3-OpenBioLLM-8B"
# F:\my_models\Llama3-OpenBioLLM-8B
cache_dir = r"F:\my_models\Llama3-OpenBioLLM-8B"

os.makedirs(cache_dir, exist_ok=True)

print("=" * 80)
print("缓存位置检查")
print("=" * 80)

print(f"\n[1] HuggingFace 默认缓存路径:")
print(f"    {default_cache_path}")

print(f"\n[2] 指定的缓存目录:")
print(f"    {cache_dir}")

print(f"\n[3] 检查缓存目录内容:")
if os.path.exists(default_cache_path):
    items = os.listdir(default_cache_path)
    print(f"    默认缓存目录存在，包含 {len(items)} 个项目")
    if items:
        print(f"    内容: {items[:5]}" + (f" ... 还有 {len(items)-5} 个" if len(items) > 5 else ""))
else:
    print(f"    默认缓存目录不存在")

if os.path.exists(cache_dir):
    items = os.listdir(cache_dir)
    print(f"    指定缓存目录存在，包含 {len(items)} 个项目")
    if items:
        print(f"    内容: {items[:5]}" + (f" ... 还有 {len(items)-5} 个" if len(items) > 5 else ""))
else:
    print(f"    指定缓存目录不存在（将在下载时创建）")

print(f"\n模型ID: {model_id}")
print(f"网络连接: {requests.get('https://huggingface.co').status_code}")
print("=" * 80)
try:
    print("\n[1/1] 使用snapshot_download下载完整模型...")
    print("这将下载模型权重和Tokenizer，不加载到内存中")
    repo_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=False
    )
    
    print(f"✓ 模型下载完成")
    print(f"\n{'=' * 60}")
    print(f"模型信息:")
    print(f"  - 模型ID: {model_id}")
    print(f"  - 本地路径: {repo_path}")
    print(f"  - 缓存目录: {cache_dir}")
    print(f"\n模型已保存到: {cache_dir}")
    print("✓ 模型下载成功！可用于后续在服务器上加载")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ 下载失败: {e}")
    raise
