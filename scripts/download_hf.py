import argparse
from huggingface_hub import snapshot_download


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="hf snapshot downloader",
        description="a utility to download from huggingface hub",
    )
    parser.add_argument("model")
    parser.print_help()

    args = parser.parse_args()

    snapshot_download(args.model)