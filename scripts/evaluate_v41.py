"""Evaluate the downloaded V4.1 runs for both checkpoint selections."""

import argparse
import json
from pathlib import Path

from evaluate_paper import ROOT, run_suite
from paper_experiments import build_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation/v4.1")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    manifest_path = output.with_name(f"{output.name}_manifest.json")
    if not manifest_path.exists():
        build_manifest(args.run_root.resolve(), manifest_path, smoke=False)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["protocol_id"] = "v4.1_gaussian"
    run_suite(manifest, manifest_path.parent, output, v41=True)


if __name__ == "__main__":
    main()
