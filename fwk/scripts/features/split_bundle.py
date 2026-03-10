import sys
from pathlib import Path
from typing import Optional, Tuple
import argparse

import pandas as pd

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent.parent
sys.path.insert(0, str(_project_root / "data_norm_features"))

BUNDLE_DIR = Path("/home/brian/sing/data/bundle")


def split_bundle(
    p_input_path: Path,
    p_train_pct: float,
    p_output_dir: Optional[Path] = None,
    p_prefix: Optional[str] = None,
) -> Tuple[Path, Path]:
    """
    Split a bundle parquet file into train and test sets.

    Args:
        p_input_path: Path to input parquet bundle
        p_train_pct: Percentage for training set (0.0-1.0)
        p_output_dir: Output directory (default: same as input)
        p_prefix: Prefix for output filenames (default: use input stem)

    Returns:
        Tuple of (train_path, test_path)
    """
    if p_train_pct <= 0 or p_train_pct >= 1:
        raise ValueError(f"p_train_pct must be between 0 and 1, got {p_train_pct}")

    output_dir = p_output_dir or p_input_path.parent

    print(f"Loading bundle: {p_input_path}")
    df = pd.read_parquet(p_input_path)

    total_rows = len(df)
    train_rows = int(total_rows * p_train_pct)
    test_rows = total_rows - train_rows

    print(f"Total rows: {total_rows}")
    print(f"Train split: {p_train_pct * 100:.1f}% ({train_rows} rows)")
    print(f"Test split: {(1 - p_train_pct) * 100:.1f}% ({test_rows} rows)")

    train_df = df.iloc[:train_rows].copy()
    test_df = df.iloc[train_rows:].copy()

    prefix = p_prefix or p_input_path.stem
    train_path = output_dir / f"{prefix}_train.parquet"
    test_path = output_dir / f"{prefix}_test.parquet"

    train_df.to_parquet(train_path)
    test_df.to_parquet(test_path)

    train_size_mb = train_path.stat().st_size / (1024 * 1024)
    test_size_mb = test_path.stat().st_size / (1024 * 1024)

    print(f"\nTrain bundle saved: {train_path} ({train_size_mb:.2f} MB)")
    print(f"Test bundle saved: {test_path} ({test_size_mb:.2f} MB)")

    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(
        description="Split a bundle parquet into train and test sets"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(BUNDLE_DIR / "etf_features_bundle.parquet"),
        help="Input bundle parquet path",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.8,
        help="Percentage for training set (0.0-1.0, default: 0.8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Prefix for output filenames (default: use input stem)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("Bundle Split Tool")
    print("=" * 60)

    split_bundle(
        p_input_path=input_path,
        p_train_pct=args.train_pct,
        p_output_dir=output_dir,
        p_prefix=args.prefix,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(BUNDLE_DIR / "etf_features_bundle.parquet"),
        help="Input bundle parquet path",
    )
    parser.add_argument(
        "--train-pct",
        type=float,
        default=0.8,
        help="Percentage for training set (0.0-1.0, default: 0.8)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: same as input)",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else None

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    print("=" * 60)
    print("Bundle Split Tool")
    print("=" * 60)

    split_bundle(
        p_input_path=input_path,
        p_train_pct=args.train_pct,
        p_output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
