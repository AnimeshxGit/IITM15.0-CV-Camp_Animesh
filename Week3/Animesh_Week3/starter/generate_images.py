"""
generate_images.py -- SCRIPT 1 of 2
Loads your model via model_loader.py, then generates an image for every row in
prompts.csv, saving each to <out_dir>/<case_number>.png. Resumable: skips any
image already present. Blank outputs are logged; evaluate.py scores them as
failures.

Usage:
    python generate_images.py --data_dir ./data_public --out_dir ./generated
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from metrics import is_degenerate
from model_loader import load_model, generate


def read_prompts(data_dir):
    with open(os.path.join(data_dir, "prompts.csv")) as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="./data_public")
    ap.add_argument("--out_dir", default="./generated")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rows = read_prompts(args.data_dir)

    print("Loading model via model_loader.load_model() ...")
    pipe = load_model()

    generated, skipped, blank = 0, 0, []
    for i, row in enumerate(rows, 1):
        out_path = os.path.join(args.out_dir, f"{row['case_number']}.png")
        if os.path.exists(out_path):
            skipped += 1
            continue
        img = generate(pipe, row["prompt"], int(row["seed"]))
        img.save(out_path)
        generated += 1
        if is_degenerate(img):
            blank.append(row["case_number"])
            print(f"  [{i}/{len(rows)}] {row['case_number']}  [BLANK OUTPUT]")
        else:
            print(f"  [{i}/{len(rows)}] {row['case_number']}")

    print(f"\nDone. Generated {generated}, skipped {skipped}, {len(rows)} total.")
    if blank:
        print(f"{len(blank)} blank output(s): {', '.join(blank)}")
    print(f"Next: python evaluate.py --data_dir {args.data_dir} "
          f"--generated_dir {args.out_dir}")


if __name__ == "__main__":
    main()
