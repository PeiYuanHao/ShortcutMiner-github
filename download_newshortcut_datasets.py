import os
import sys
import shutil
import subprocess
from pathlib import Path


# =========================
# 1. 自动检查 kagglehub
# =========================
try:
    import kagglehub
except ImportError:
    print("[INFO] 未检测到 kagglehub，正在自动安装一次...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
    import kagglehub


# =========================
# 2. 保存位置
# =========================
BASE_DIR = Path(r"E:\小龙虾")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# kagglehub 默认会先下载到缓存目录
# 这里把缓存也放到 E:\小龙虾 下面，方便管理
os.environ["KAGGLEHUB_CACHE"] = str(BASE_DIR / "_kagglehub_cache")


# =========================
# 3. 三个数据集
# =========================
DATASETS = {
    "plant_disease_recognition": "rashikrahmanpritom/plant-disease-recognition-dataset",
    "mango_fruit_disease": "shuvokumarbasak4004/mango-fruit-early-diseases-detection",
    "skin_cancer_workshop_ham": "aviraj09/workshop-dataset-ham",
}


# =========================
# 4. 复制下载结果到目标目录
# =========================
def copy_dataset(src_dir: Path, dst_dir: Path):
    dst_dir.mkdir(parents=True, exist_ok=True)

    for item in src_dir.iterdir():
        target = dst_dir / item.name

        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


# =========================
# 5. 下载函数
# =========================
def download_one(folder_name: str, dataset_slug: str):
    save_dir = BASE_DIR / folder_name

    print("\n" + "=" * 80)
    print(f"[START] 下载数据集：{dataset_slug}")
    print(f"[SAVE ] 保存到：{save_dir}")
    print("=" * 80)

    try:
        downloaded_path = kagglehub.dataset_download(dataset_slug)
        downloaded_path = Path(downloaded_path)

        print(f"[OK] KaggleHub 下载缓存位置：{downloaded_path}")

        copy_dataset(downloaded_path, save_dir)

        print(f"[DONE] 已复制到目标文件夹：{save_dir}")

    except Exception as e:
        print(f"[ERROR] 下载失败：{dataset_slug}")
        print("原因可能是：")
        print("1. 这个数据集需要登录或同意协议；")
        print("2. 网络访问 Kaggle 不稳定；")
        print("3. 数据集链接发生变化。")
        print(f"具体错误：{e}")


# =========================
# 6. 主程序
# =========================
def main():
    print("[INFO] 开始下载三组 Kaggle 公开数据集")
    print(f"[INFO] 根目录：{BASE_DIR}")

    for folder_name, dataset_slug in DATASETS.items():
        download_one(folder_name, dataset_slug)

    print("\n" + "=" * 80)
    print("[FINISH] 全部任务执行完成")
    print("=" * 80)

    print("\n最终文件夹：")
    for folder_name in DATASETS.keys():
        print(BASE_DIR / folder_name)


if __name__ == "__main__":
    main()