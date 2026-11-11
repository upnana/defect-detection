"""从 Hugging Face 下载 shrish23/screw-dataset 到本地目录。"""
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Download screw-dataset from Hugging Face")
    parser.add_argument(
        "--output",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "screw-dataset"),
        help="Local output directory",
    )
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("请先安装: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    print(f"正在下载 shrish23/screw-dataset 的 dataset/ 目录到 {out} ...")
    snapshot_download(
        repo_id="shrish23/screw-dataset",
        repo_type="dataset",
        local_dir=str(out),
        allow_patterns=["dataset/**"],
    )
    print(f"已保存至 {out / 'dataset'}")


if __name__ == "__main__":
    main()
