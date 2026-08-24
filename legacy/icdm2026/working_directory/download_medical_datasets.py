# -*- coding: utf-8 -*-
"""
下载 4 个医学捷径学习数据集到 E:\小龙虾

目录结构：
E:\小龙虾
├── HAM10000
├── BreakHis
├── Kvasir_v2
└── PH2

说明：
1. HAM10000、Kvasir_v2、PH2 主要通过 KaggleHub 下载。
2. BreakHis 优先走官网直链，失败后走 KaggleHub 镜像。
3. 第一次使用 KaggleHub 时，某些 Kaggle 数据集可能需要登录或接受数据集协议。
"""

import os
import sys
import time
import shutil
import tarfile
import subprocess
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


ROOT = Path(r"E:\小龙虾")

# 如果你需要代理，取消下面两行注释，并改成自己的端口
# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"


def pip_install(package: str):
    """自动安装缺失的 Python 包"""
    print(f"\n[检查依赖] {package}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", package])


def ensure_packages():
    """安装 KaggleHub"""
    try:
        import kagglehub  # noqa
        print("[OK] kagglehub 已安装")
    except ImportError:
        pip_install("kagglehub")


def make_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def download_file(url: str, out_file: Path, chunk_size: int = 1024 * 1024):
    """普通直链下载，带进度显示"""
    make_dir(out_file.parent)

    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"[跳过] 已存在：{out_file}")
        return

    print(f"[下载] {url}")
    print(f"[保存] {out_file}")

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0"
        }
    )

    with urlopen(req, timeout=60) as response:
        total = response.headers.get("Content-Length")
        total = int(total) if total is not None else None

        downloaded = 0
        start = time.time()

        with open(out_file, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break

                f.write(chunk)
                downloaded += len(chunk)

                if total:
                    percent = downloaded / total * 100
                    mb_done = downloaded / 1024 / 1024
                    mb_total = total / 1024 / 1024
                    print(
                        f"\r进度：{percent:6.2f}%  "
                        f"{mb_done:.1f}/{mb_total:.1f} MB",
                        end=""
                    )
                else:
                    mb_done = downloaded / 1024 / 1024
                    print(f"\r已下载：{mb_done:.1f} MB", end="")

        print()
        cost = time.time() - start
        print(f"[完成] 用时 {cost:.1f} 秒")


def extract_tar_gz(tar_path: Path, out_dir: Path):
    """解压 .tar.gz"""
    marker = out_dir / ".extract_done"

    if marker.exists():
        print(f"[跳过] 已解压：{out_dir}")
        return

    print(f"[解压] {tar_path}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)

    marker.write_text("done", encoding="utf-8")
    print(f"[完成] 解压到：{out_dir}")


def download_kaggle_dataset(name: str, handle_list, out_dir: Path, force_download: bool = False):
    """
    使用 KaggleHub 下载数据集。
    handle_list 可以放多个候选镜像，前一个失败就试下一个。
    """
    import kagglehub

    make_dir(out_dir)

    print("\n" + "=" * 80)
    print(f"[KaggleHub] 开始下载：{name}")
    print(f"[目标目录] {out_dir}")

    for handle in handle_list:
        try:
            print(f"[尝试] {handle}")

            # 最新 kagglehub 支持 output_dir 参数，会直接下载到指定目录
            path = kagglehub.dataset_download(
                handle,
                output_dir=str(out_dir),
                force_download=force_download
            )

            print(f"[成功] {name}")
            print(f"[返回路径] {path}")
            return True

        except TypeError:
            # 兼容旧版 kagglehub：如果不支持 output_dir，就先下载到缓存，再复制到目标目录
            print("[提示] 当前 kagglehub 版本可能不支持 output_dir，尝试缓存下载后复制。")
            try:
                cache_path = Path(kagglehub.dataset_download(handle))
                print(f"[缓存路径] {cache_path}")

                for item in cache_path.iterdir():
                    dst = out_dir / item.name
                    if dst.exists():
                        print(f"[跳过] 已存在：{dst}")
                        continue

                    if item.is_dir():
                        shutil.copytree(item, dst)
                    else:
                        shutil.copy2(item, dst)

                print(f"[成功] {name}")
                return True

            except Exception as e:
                print(f"[失败] {handle}")
                print(e)

        except Exception as e:
            print(f"[失败] {handle}")
            print(e)

    print(f"[警告] {name} 下载失败，所有候选镜像都不可用。")
    return False


def download_breakhis():
    """
    BreakHis：
    优先使用官网直链；
    如果官网直链失败，则尝试 Kaggle 镜像。
    """
    out_dir = ROOT / "BreakHis"
    make_dir(out_dir)

    print("\n" + "=" * 80)
    print("[BreakHis] 开始下载")

    official_url = "http://www.inf.ufpr.br/vri/databases/BreaKHis_v1.tar.gz"
    tar_file = out_dir / "BreaKHis_v1.tar.gz"

    try:
        download_file(official_url, tar_file)
        extract_tar_gz(tar_file, out_dir)
        print("[成功] BreakHis 官方版本下载完成")
        return True

    except (HTTPError, URLError, TimeoutError, Exception) as e:
        print("[警告] BreakHis 官方直链失败，改用 Kaggle 镜像")
        print(e)

        return download_kaggle_dataset(
            name="BreakHis",
            handle_list=[
                "ambarish/breakhis",
            ],
            out_dir=out_dir
        )


def main():
    print("=" * 80)
    print("医学捷径学习数据集下载脚本")
    print(f"根目录：{ROOT}")
    print("=" * 80)

    make_dir(ROOT)
    ensure_packages()

    results = {}

    # 1. HAM10000
    results["HAM10000"] = download_kaggle_dataset(
        name="HAM10000",
        handle_list=[
            "kmader/skin-cancer-mnist-ham10000",
        ],
        out_dir=ROOT / "HAM10000"
    )

    # 2. BreakHis
    results["BreakHis"] = download_breakhis()

    # 3. Kvasir v2
    results["Kvasir_v2"] = download_kaggle_dataset(
        name="Kvasir v2",
        handle_list=[
            "plhalvorsen/kvasir-v2-a-gastrointestinal-tract-dataset",
            "yasserhessein/the-kvasir-dataset",
        ],
        out_dir=ROOT / "Kvasir_v2"
    )

    # 4. PH2
    results["PH2"] = download_kaggle_dataset(
        name="PH2",
        handle_list=[
            "spacesurfer/ph2-dataset",
        ],
        out_dir=ROOT / "PH2"
    )

    print("\n" + "=" * 80)
    print("下载结果汇总")
    print("=" * 80)

    for name, ok in results.items():
        status = "成功" if ok else "失败"
        print(f"{name:12s}: {status}")

    print("\n数据集保存位置：")
    print(r"E:\小龙虾\HAM10000")
    print(r"E:\小龙虾\BreakHis")
    print(r"E:\小龙虾\Kvasir_v2")
    print(r"E:\小龙虾\PH2")

    print("\n如果 Kaggle 数据集失败，一般是没有登录 Kaggle 或没接受数据集协议。")
    print("可以先在浏览器打开对应 Kaggle 数据集页面，登录后点一次下载/接受协议，再重新运行本脚本。")


if __name__ == "__main__":
    main()