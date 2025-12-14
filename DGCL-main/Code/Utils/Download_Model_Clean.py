import requests
from huggingface_hub import snapshot_download
import os
import shutil
from huggingface_hub.constants import default_cache_path

model_id = "aaditya/Llama3-OpenBioLLM-8B"
cache_dir = r"F:\my_models\Llama3-OpenBioLLM-8B"

print("=" * 80)
print("模型下载脚本（带清理选项）")
print("=" * 80)

print(f"\n[1] 缓存位置检查")
print(f"    默认缓存: {default_cache_path}")
print(f"    指定缓存: {cache_dir}")

# 检查是否需要清理
print(f"\n[2] 清理选项")
if os.path.exists(cache_dir):
    items = os.listdir(cache_dir)
    print(f"    缓存目录存在，包含 {len(items)} 个项目")
    
    clean = input("    是否清理缓存重新下载？(y/n): ").strip().lower()
    if clean == 'y':
        print("    正在清理缓存...")
        try:
            shutil.rmtree(cache_dir)
            os.makedirs(cache_dir, exist_ok=True)
            print("    ✓ 缓存已清理")
        except Exception as e:
            print(f"    ✗ 清理失败: {e}")
    else:
        print("    保留现有缓存，继续下载")
else:
    os.makedirs(cache_dir, exist_ok=True)
    print(f"    缓存目录不存在，已创建")

print(f"\n[3] 网络连接检查")
try:
    status = requests.get("https://huggingface.co", timeout=5).status_code
    print(f"    HuggingFace 连接: {status}")
except Exception as e:
    print(f"    ✗ 连接失败: {e}")

print("\n" + "=" * 80)
print("开始下载模型...")
print("=" * 80)

try:
    print(f"\n下载模型: {model_id}")
    print("这将只下载模型文件，不加载到内存中\n")
    
    repo_path = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=False,
        force_download=False
    )
    
    print(f"\n✓ 模型下载完成")
    print(f"\n{'=' * 80}")
    print(f"下载信息:")
    print(f"  - 模型ID: {model_id}")
    print(f"  - 本地路径: {repo_path}")
    print(f"  - 缓存目录: {cache_dir}")
    print(f"\n模型已保存到: {cache_dir}")
    print("✓ 下载成功！可用于后续在服务器上加载")
    print("=" * 80)
    
except Exception as e:
    print(f"✗ 下载失败: {e}")
    print(f"\n排查步骤:")
    print(f"  1. 检查网络连接")
    print(f"  2. 检查磁盘空间（需要约16GB）")
    print(f"  3. 尝试清理缓存重新下载")
    raise
