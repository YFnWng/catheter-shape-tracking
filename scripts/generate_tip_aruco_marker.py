#!/usr/bin/env python3
"""Generate a compact, print-ready ArUco marker for tip-pose calibration.

The PDF preserves the requested marker size when printed at 100% / Actual Size.
The PNG is a high-resolution reference and is not dimensionally controlled.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from shape_tracking.boards import ARUCO_DICT, get_dictionary  # noqa: E402


PAGE_SIZES_MM = {
    "letter": (215.9, 279.4),
    "a4": (210.0, 297.0),
}
DEFAULT_MARKER_ID = 33
DEFAULT_MARKER_SIZE_MM = 15.0
DEFAULT_QUIET_ZONE_MM = 3.0
RESERVED_ID_RANGE = (0, 32)


def marker_image(marker_id: int, pixels: int, dictionary) -> np.ndarray:
    """Return the marker itself, excluding its required white quiet zone."""
    image = np.empty((pixels, pixels), dtype=np.uint8)
    cv2.aruco.generateImageMarker(
        dictionary, marker_id, pixels, image, borderBits=1)
    return image


def reference_image(
        marker: np.ndarray,
        marker_size_mm: float,
        quiet_zone_mm: float) -> np.ndarray:
    border_pixels = max(
        1, int(round(marker.shape[0] * quiet_zone_mm / marker_size_mm)))
    return cv2.copyMakeBorder(
        marker, border_pixels, border_pixels, border_pixels, border_pixels,
        cv2.BORDER_CONSTANT, value=255)


def detected_ids(image: np.ndarray, dictionary) -> list[int]:
    margin = max(20, image.shape[0] // 20)
    padded = cv2.copyMakeBorder(
        image, margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=255)
    detector = cv2.aruco.ArucoDetector(dictionary)
    _, ids, _ = detector.detectMarkers(padded)
    if ids is None:
        return []
    return sorted(int(value) for value in ids.reshape(-1))


def write_print_pdf(
        path: Path,
        marker: np.ndarray,
        marker_id: int,
        marker_size_mm: float,
        quiet_zone_mm: float,
        copies: int,
        page_name: str) -> None:
    page_width_mm, page_height_mm = PAGE_SIZES_MM[page_name]
    target_size_mm = marker_size_mm + 2.0 * quiet_zone_mm
    label_space_mm = 8.0
    cell_gap_mm = 12.0
    cell_width_mm = target_size_mm + cell_gap_mm
    cell_height_mm = target_size_mm + label_space_mm + cell_gap_mm

    usable_width_mm = page_width_mm - 30.0
    columns = max(1, min(copies, int(usable_width_mm // cell_width_mm)))
    rows = int(np.ceil(copies / columns))
    occupied_width_mm = columns * cell_width_mm
    occupied_height_mm = rows * cell_height_mm
    if occupied_height_mm > page_height_mm - 45.0:
        raise ValueError(
            f"{copies} copies of a {target_size_mm:.1f} mm target do not fit "
            f"on {page_name}")

    figure = Figure(
        figsize=(page_width_mm / 25.4, page_height_mm / 25.4),
        dpi=150, facecolor="white")
    x_origin_mm = (page_width_mm - occupied_width_mm) / 2.0
    y_top_mm = page_height_mm - 24.0

    for index in range(copies):
        row, column = divmod(index, columns)
        outer_x_mm = (
            x_origin_mm + column * cell_width_mm + cell_gap_mm / 2.0)
        outer_y_mm = (
            y_top_mm - (row + 1) * cell_height_mm + label_space_mm
            + cell_gap_mm / 2.0)
        marker_x_mm = outer_x_mm + quiet_zone_mm
        marker_y_mm = outer_y_mm + quiet_zone_mm

        axes = figure.add_axes([
            marker_x_mm / page_width_mm,
            marker_y_mm / page_height_mm,
            marker_size_mm / page_width_mm,
            marker_size_mm / page_height_mm,
        ])
        axes.imshow(
            marker, cmap="gray", vmin=0, vmax=255,
            interpolation="nearest")
        axes.set_axis_off()

        cut_outline = Rectangle(
            (outer_x_mm / page_width_mm, outer_y_mm / page_height_mm),
            target_size_mm / page_width_mm,
            target_size_mm / page_height_mm,
            transform=figure.transFigure, fill=False, linewidth=0.35,
            linestyle=(0, (2, 2)), edgecolor="0.65")
        figure.add_artist(cut_outline)
        figure.text(
            (outer_x_mm + target_size_mm / 2.0) / page_width_mm,
            (outer_y_mm - 3.0) / page_height_mm,
            f"DICT_4X4_50 ID {marker_id}",
            ha="center", va="center", fontsize=6)

    figure.text(
        0.5, 0.965, "Catheter-tip optical calibration marker",
        ha="center", va="center", fontsize=11)
    figure.text(
        0.5, 0.035,
        "Print at 100% / Actual Size; disable Fit/Shrink. "
        f"Black marker edge must measure {marker_size_mm:.2f} mm.",
        ha="center", va="center", fontsize=8)
    with PdfPages(path) as pdf:
        pdf.savefig(figure)


def generate(
        output_dir: Path,
        marker_id: int = DEFAULT_MARKER_ID,
        marker_size_mm: float = DEFAULT_MARKER_SIZE_MM,
        quiet_zone_mm: float = DEFAULT_QUIET_ZONE_MM,
        pixels: int = 1200,
        copies: int = 6,
        page_name: str = "letter",
        prefix: str = "tip_aruco") -> tuple[Path, Path, Path]:
    if marker_size_mm <= 0.0:
        raise ValueError("marker size must be positive")
    if quiet_zone_mm < marker_size_mm / 8.0:
        raise ValueError(
            "quiet zone must be at least one marker bit "
            f"({marker_size_mm / 8.0:.3f} mm for a 4x4 marker)")
    if pixels < 160:
        raise ValueError("pixels must be at least 160")
    if copies < 1:
        raise ValueError("copies must be at least one")
    if page_name not in PAGE_SIZES_MM:
        raise ValueError(f"unknown page size: {page_name}")

    dictionary = get_dictionary()
    capacity = int(dictionary.bytesList.shape[0])
    if marker_id < 0 or marker_id >= capacity:
        raise ValueError(
            f"marker ID must be between 0 and {capacity - 1}")
    if RESERVED_ID_RANGE[0] <= marker_id <= RESERVED_ID_RANGE[1]:
        raise ValueError(
            f"marker ID {marker_id} is reserved by the base/field boards; "
            f"use {RESERVED_ID_RANGE[1] + 1}-{capacity - 1}")

    marker = marker_image(marker_id, pixels, dictionary)
    reference = reference_image(marker, marker_size_mm, quiet_zone_mm)
    observed = detected_ids(reference, dictionary)
    if observed != [marker_id]:
        raise RuntimeError(
            f"generated-marker verification failed: expected [{marker_id}], "
            f"detected {observed}")

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{prefix}_id_{marker_id}_{marker_size_mm:g}mm"
    png_path = output_dir / f"{stem}.png"
    pdf_path = output_dir / f"{stem}_{page_name}.pdf"
    manifest_path = output_dir / f"{stem}_manifest.json"
    if not cv2.imwrite(str(png_path), reference):
        raise OSError(f"failed to write {png_path}")
    write_print_pdf(
        pdf_path, marker, marker_id, marker_size_mm, quiet_zone_mm,
        copies, page_name)

    target_size_mm = marker_size_mm + 2.0 * quiet_zone_mm
    manifest = {
        "purpose": "rigid optical-to-EM catheter-tip pose calibration",
        "dictionary": "DICT_4X4_50",
        "opencv_dictionary_constant": int(ARUCO_DICT),
        "marker_id": marker_id,
        "marker_size_mm": marker_size_mm,
        "quiet_zone_each_side_mm": quiet_zone_mm,
        "cut_target_size_mm": [target_size_mm, target_size_mm],
        "copies": copies,
        "reference_png": png_path.name,
        "print_pdf": pdf_path.name,
        "print_instruction": (
            "Print the PDF at 100% / Actual Size with all scaling disabled; "
            "measure the black marker edge before use."),
        "mounting_instruction": (
            "Mount one copy to a rigid, flat, matte tab fixed to the tip "
            "housing. Keep the complete white quiet zone visible."),
        "reserved_marker_ids": {
            "base_boards": [1, 16],
            "field_generator_board_in_use": [17, 24],
            "reserved_second_field_board": [25, 32],
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path, pdf_path, png_path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path,
        default=REPOSITORY / "generated_tip_marker",
        help="output directory (default: repository/generated_tip_marker)")
    parser.add_argument(
        "--marker-id", type=int, default=DEFAULT_MARKER_ID,
        help=f"DICT_4X4_50 marker ID (default: {DEFAULT_MARKER_ID})")
    parser.add_argument(
        "--marker-size-mm", type=float, default=DEFAULT_MARKER_SIZE_MM,
        help=f"black marker edge length (default: {DEFAULT_MARKER_SIZE_MM})")
    parser.add_argument(
        "--quiet-zone-mm", type=float, default=DEFAULT_QUIET_ZONE_MM,
        help=(
            "white margin on every side "
            f"(default: {DEFAULT_QUIET_ZONE_MM})"))
    parser.add_argument(
        "--pixels", type=int, default=1200,
        help="black marker PNG width/height in pixels (default: 1200)")
    parser.add_argument(
        "--copies", type=int, default=6,
        help="number of copies on the print sheet (default: 6)")
    parser.add_argument(
        "--page", choices=sorted(PAGE_SIZES_MM), default="letter",
        help="print-ready PDF page size (default: letter)")
    parser.add_argument(
        "--prefix", default="tip_aruco",
        help="output filename prefix (default: tip_aruco)")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest, pdf, png = generate(
        args.output_dir, marker_id=args.marker_id,
        marker_size_mm=args.marker_size_mm,
        quiet_zone_mm=args.quiet_zone_mm, pixels=args.pixels,
        copies=args.copies, page_name=args.page, prefix=args.prefix)
    print(f"Print-ready PDF: {pdf}")
    print(f"Reference PNG: {png}")
    print(f"Manifest: {manifest}")
    print(
        f"Marker: DICT_4X4_50 ID {args.marker_id}; "
        f"{args.marker_size_mm:.2f} mm black edge; "
        f"{args.marker_size_mm + 2.0 * args.quiet_zone_mm:.2f} mm cut target")


if __name__ == "__main__":
    main()
