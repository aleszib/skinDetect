"""Command line entry point for SkinTrack."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from skintrack.io.photos import write_photo_import_manifest
from skintrack.overlap.candidates import write_overlap_candidate_manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""

    parser = argparse.ArgumentParser(prog="skintrack")
    subparsers = parser.add_subparsers(dest="command")

    import_parser = subparsers.add_parser(
        "import-photos",
        help="Scan a local folder of images and write a JSON manifest.",
    )
    import_parser.add_argument("input_dir", help="Input directory to scan.")
    import_parser.add_argument(
        "--output",
        required=True,
        help="Output path for the JSON manifest.",
    )
    import_parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Scan only the top level of INPUT_DIR instead of recursing.",
    )

    overlap_parser = subparsers.add_parser(
        "rank-overlap-candidates",
        help="Rank likely overlapping photo pairs from a manifest JSON file.",
    )
    overlap_parser.add_argument("manifest_path", help="Photo import manifest JSON file.")
    overlap_parser.add_argument(
        "--output",
        required=True,
        help="Output path for the overlap candidate JSON file.",
    )
    overlap_parser.add_argument(
        "--include-nonimported",
        action="store_true",
        help="Include unreadable or unsupported manifest records with heavy penalties.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the SkinTrack CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0

    if args.command == "import-photos":
        write_photo_import_manifest(
            Path(args.input_dir),
            Path(args.output),
            recursive=not args.no_recursive,
        )
        return 0

    if args.command == "rank-overlap-candidates":
        write_overlap_candidate_manifest(
            Path(args.manifest_path),
            Path(args.output),
            include_nonimported=args.include_nonimported,
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation only.
    raise SystemExit(main())
