#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from huggingface_hub import snapshot_download

DEFAULT_MODELS_DIR = Path("/home/ubuntu/vito/project/LLM/models").resolve()

DEFAULT_MODEL_REGISTRY: List[Dict[str, Any]] = [
    {
        "local_name": "LLaMA-Omni-8B",
        "sources": [
            ("ICTNLP/Llama-3.1-8B-Omni", None),
        ],
        "ignore_patterns": ["*.msgpack", "*.h5", "*.ot"],
    },
    {
        "local_name": "spirit-lm-base-7b",
        "sources": [
            ("facebook/spirit-lm-base-7b", None),
            ("adamo1139/Meta_Spirit-LM-ungated", ["*base*"]),
        ],
        "ignore_patterns": ["*.msgpack", "*.h5", "*.ot"],
    },
]


def download_single_model(
    local_name: str,
    sources: List[Tuple[str, Optional[List[str]]]],
    models_dir: Path,
    ignore_patterns: Optional[List[str]] = None,
    token: Optional[str] = None,
    max_workers: int = 8,
) -> bool:
    target_dir = (models_dir / local_name).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"[TASK] Downloading model: {local_name}")
    print(f"[PATH] Target directory : {target_dir}")
    print("=" * 70)

    for idx, (repo_id, allow_patterns) in enumerate(sources, 1):
        try:
            print(f"-> [{idx}/{len(sources)}] Attempting download from repository: {repo_id}")
            if allow_patterns:
                print(f"   Allow patterns : {allow_patterns}")
            if ignore_patterns:
                print(f"   Ignore patterns: {ignore_patterns}")

            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                token=token,
                max_workers=max_workers,
            )
            print(f"✅ [{local_name}] Download completed! (Source: {repo_id})\n")
            return True

        except Exception as e:
            print(f"⚠️ [{local_name}] Failed to download from {repo_id}: {repr(e)}")
            if idx < len(sources):
                print("-> Switching to the next fallback source...\n")

    print(f"❌ [{local_name}] All sources failed. Please check network connection, repo ID, or HF_TOKEN permissions.\n")
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Universal Hugging Face Model Downloader"
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Specify a single Hugging Face repo_id to download directly",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Custom local directory name",
    )
    parser.add_argument(
        "--allow",
        nargs="+",
        default=None,
        help="File patterns to include",
    )
    parser.add_argument(
        "--ignore",
        nargs="+",
        default=["*.msgpack", "*.h5", "*.ot"],
        help="File patterns to exclude",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=DEFAULT_MODELS_DIR,
        help="Root directory for storing models",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face Access Token",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    models_dir = args.dir.resolve()
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.repo:
        local_name = args.name if args.name else args.repo.split("/")[-1]
        download_single_model(
            local_name=local_name,
            sources=[(args.repo, args.allow)],
            models_dir=models_dir,
            ignore_patterns=args.ignore,
            token=args.token,
            max_workers=args.workers,
        )
        return

    print(f"Starting batch download for {len(DEFAULT_MODEL_REGISTRY)} models from registry...\n")
    summary = {}
    for item in DEFAULT_MODEL_REGISTRY:
        ok = download_single_model(
            local_name=item["local_name"],
            sources=item["sources"],
            models_dir=models_dir,
            ignore_patterns=item.get("ignore_patterns", args.ignore),
            token=args.token,
            max_workers=args.workers,
        )
        summary[item["local_name"]] = "SUCCESS" if ok else "FAILED"

    print("=" * 70)
    print("📊 Download Summary:")
    for name, status in summary.items():
        print(f"  - {name}: {status}")
    print("=" * 70)


if __name__ == "__main__":
    main()