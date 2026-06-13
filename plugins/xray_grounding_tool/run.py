#!/usr/bin/env python3
"""CLI wrapper for xray_grounding_tool."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from logic import execute


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chest X-ray analysis with GPT-4o visual grounding."
    )
    parser.add_argument("image_path", help="Path to chest X-ray image (JPG/PNG)")
    parser.add_argument(
        "--no-save", action="store_true", help="Do not save annotated output image"
    )
    parser.add_argument(
        "--output-dir", default=None, help="Directory to save annotated image"
    )
    parser.add_argument("--api-key", default=None, help="OpenAI API key")
    args = parser.parse_args()

    payload = {
        "image_path": args.image_path,
        "save_annotated": not args.no_save,
    }
    if args.output_dir:
        payload["output_dir"] = args.output_dir
    if args.api_key:
        payload["api_key"] = args.api_key

    result = execute(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if result.get("error"):
        sys.exit(1)

    if result["abnormal"]:
        print(f"\n[ABNORMAL] {result['impression']}")
        for f in result["findings"]:
            print(f"  - {f}")
        if result.get("annotated_image_path"):
            print(f"\nAnnotated image saved: {result['annotated_image_path']}")
    else:
        print(f"\n[NORMAL] {result['impression']}")


if __name__ == "__main__":
    main()
