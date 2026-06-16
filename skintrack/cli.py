"""Command line entry point for SkinTrack."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from skintrack.io.photos import write_photo_import_manifest
from skintrack.overlap.candidates import write_overlap_candidate_manifest
from skintrack.regions.manual import write_validated_candidate_region_manifest
from skintrack.regions.projection import write_projected_candidate_region_manifest
from skintrack.registration.geometric import write_registration_manifest
from skintrack.tracking.temporal import write_candidate_track_manifest


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

    registration_parser = subparsers.add_parser(
        "register-candidate-pairs",
        help="Estimate geometric registration for ranked overlap candidate pairs.",
    )
    registration_parser.add_argument(
        "candidate_path",
        help="Overlap candidate JSON file.",
    )
    registration_parser.add_argument(
        "--manifest",
        required=True,
        help="Photo import manifest JSON file.",
    )
    registration_parser.add_argument(
        "--output",
        required=True,
        help="Output path for the registration JSON file.",
    )
    registration_parser.add_argument(
        "--debug-dir",
        help="Optional directory for technical debug visualization images.",
    )

    regions_parser = subparsers.add_parser(
        "validate-candidate-regions",
        help="Validate manual candidate-region JSON against a photo import manifest.",
    )
    regions_parser.add_argument(
        "regions_path",
        help="Manual candidate-region JSON file.",
    )
    regions_parser.add_argument(
        "--manifest",
        required=True,
        help="Photo import manifest JSON file.",
    )
    regions_parser.add_argument(
        "--output",
        required=True,
        help="Output path for the validated candidate-region JSON file.",
    )
    regions_parser.add_argument(
        "--overlay-dir",
        help="Optional directory for neutral technical overlay images.",
    )

    projection_parser = subparsers.add_parser(
        "project-candidate-regions",
        help="Project validated candidate regions through registered image pairs.",
    )
    projection_parser.add_argument(
        "--validated-regions",
        required=True,
        help="Validated candidate-region JSON file.",
    )
    projection_parser.add_argument(
        "--registrations",
        required=True,
        help="Registration JSON file.",
    )
    projection_parser.add_argument(
        "--manifest",
        required=True,
        help="Photo import manifest JSON file.",
    )
    projection_parser.add_argument(
        "--output",
        required=True,
        help="Output path for the projected candidate-region JSON file.",
    )
    projection_parser.add_argument(
        "--overlay-dir",
        help="Optional directory for neutral technical projection overlays.",
    )

    tracking_parser = subparsers.add_parser(
        "track-candidate-regions",
        help="Group validated and projected candidate regions into temporal tracks.",
    )
    tracking_parser.add_argument(
        "--manifest",
        required=True,
        help="Photo import manifest JSON file.",
    )
    tracking_parser.add_argument(
        "--validated-regions",
        required=True,
        help="Validated candidate-region JSON file.",
    )
    tracking_parser.add_argument(
        "--projections",
        required=True,
        help="Projected candidate-region JSON file.",
    )
    tracking_parser.add_argument(
        "--output",
        required=True,
        help="Output path for the candidate track JSON file.",
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

    if args.command == "register-candidate-pairs":
        write_registration_manifest(
            Path(args.manifest),
            Path(args.candidate_path),
            Path(args.output),
            debug_dir=Path(args.debug_dir) if args.debug_dir is not None else None,
        )
        return 0

    if args.command == "validate-candidate-regions":
        write_validated_candidate_region_manifest(
            Path(args.manifest),
            Path(args.regions_path),
            Path(args.output),
            overlay_dir=Path(args.overlay_dir) if args.overlay_dir is not None else None,
        )
        return 0

    if args.command == "project-candidate-regions":
        write_projected_candidate_region_manifest(
            Path(args.manifest),
            Path(args.validated_regions),
            Path(args.registrations),
            Path(args.output),
            overlay_dir=Path(args.overlay_dir) if args.overlay_dir is not None else None,
        )
        return 0

    if args.command == "track-candidate-regions":
        write_candidate_track_manifest(
            Path(args.manifest),
            Path(args.validated_regions),
            Path(args.projections),
            Path(args.output),
        )
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":  # pragma: no cover - manual invocation only.
    raise SystemExit(main())
