import requests
from huggingface_hub import snapshot_download
import os
from huggingface_hub.constants import default_cache_path

model_id = "aaditya/Llama3-OpenBioLLM-8B"
# F:\my_models\Llama3-OpenBioLLM-8B
cache_dir = r"F:\my_models\Llama3-OpenBioLLM-8B"

os.makedirs(cache_dir, exist_ok=True)

print(f"开始下载模型到: {cache_dir}")
print(f"模型ID: {model_id}")
print(requests.get("https://huggingface.co").status_code)
print(default_cache_path)
print("=" * 60)

try:
    print("\n[1/1] 使用snapshot_download下载完整模型...")
    print("这将下载模型权重和Tokenizer，不加载到内存中")
    repo_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
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
