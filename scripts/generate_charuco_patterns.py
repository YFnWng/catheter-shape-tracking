#!/usr/bin/env python3
'''Generate print-ready ChArUco targets compatible with shape_tracking.

The output PDF preserves a 100 x 100 mm chessboard when printed at 100% / Actual
Size. PNG files are high-resolution references and should not be used for
dimensionally controlled printing.
'''
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / 'src'
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from shape_tracking.boards import (  # noqa: E402
    ARUCO_DICT,
    BOARD_ID_OFFSETS,
    MARKER_LENGTH_M,
    MARKERS_PER_BOARD,
    SQUARE_LENGTH_M,
    SQUARES_X,
    SQUARES_Y,
    get_dictionary,
)


PAGE_SIZES_MM = {
    'letter': (279.4, 215.9),  # landscape 11 x 8.5 inch
    'a4': (297.0, 210.0),      # landscape
}
BOARD_SIZE_MM = SQUARES_X * SQUARE_LENGTH_M * 1000.0
MARKER_SIZE_MM = MARKER_LENGTH_M * 1000.0
DEFAULT_FIRST_ID = max(BOARD_ID_OFFSETS) + MARKERS_PER_BOARD


def board_image(first_id: int, pixels: int, dictionary):
    ids = np.arange(
        first_id, first_id + MARKERS_PER_BOARD, dtype=np.int32)
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M,
        dictionary, ids)
    image = board.generateImage(
        (pixels, pixels), marginSize=0, borderBits=1)
    return image, ids


def detected_ids(image, dictionary):
    margin = max(40, image.shape[0] // 20)
    padded = cv2.copyMakeBorder(
        image, margin, margin, margin, margin,
        cv2.BORDER_CONSTANT, value=255)
    detector = cv2.aruco.ArucoDetector(dictionary)
    _, ids, _ = detector.detectMarkers(padded)
    if ids is None:
        return []
    return sorted(int(value) for value in ids.reshape(-1))


def write_print_pdf(path: Path, patterns, page_name: str):
    page_width_mm, page_height_mm = PAGE_SIZES_MM[page_name]
    gap_mm = 10.0
    page_width_in = page_width_mm / 25.4
    page_height_in = page_height_mm / 25.4
    with PdfPages(path) as pdf:
        for start in range(0, len(patterns), 2):
            page_patterns = patterns[start:start + 2]
            occupied_mm = (
                len(page_patterns) * BOARD_SIZE_MM
                + max(0, len(page_patterns) - 1) * gap_mm)
            x_start_mm = (page_width_mm - occupied_mm) / 2.0
            y_start_mm = (page_height_mm - BOARD_SIZE_MM) / 2.0 + 5.0
            figure = Figure(
                figsize=(page_width_in, page_height_in), dpi=100,
                facecolor='white')
            for column, pattern in enumerate(page_patterns):
                x_mm = x_start_mm + column * (BOARD_SIZE_MM + gap_mm)
                axes = figure.add_axes([
                    x_mm / page_width_mm,
                    y_start_mm / page_height_mm,
                    BOARD_SIZE_MM / page_width_mm,
                    BOARD_SIZE_MM / page_height_mm,
                ])
                axes.imshow(
                    pattern['image'], cmap='gray', vmin=0, vmax=255,
                    interpolation='nearest')
                axes.set_axis_off()
                center_x = (x_mm + BOARD_SIZE_MM / 2.0) / page_width_mm
                label_y = (y_start_mm - 7.0) / page_height_mm
                figure.text(
                    center_x, label_y,
                    f"{pattern['name']}  IDs {pattern['ids'][0]}-"
                    f"{pattern['ids'][-1]}",
                    ha='center', va='center', fontsize=8)
            figure.text(
                0.5, 0.035,
                'Print at 100% / Actual Size. Chessboard outer edge must measure '
                f'{BOARD_SIZE_MM:.1f} x {BOARD_SIZE_MM:.1f} mm.',
                ha='center', va='center', fontsize=8)
            pdf.savefig(figure)


def generate(output_dir: Path, count: int, first_id: int, pixels: int,
             page_name: str, prefix: str):
    if count < 1:
        raise ValueError('count must be at least one')
    if pixels < 400:
        raise ValueError('pixels must be at least 400')
    dictionary = get_dictionary()
    capacity = int(dictionary.bytesList.shape[0])
    final_id = first_id + count * MARKERS_PER_BOARD - 1
    if first_id < 0 or final_id >= capacity:
        raise ValueError(
            f'requested marker IDs {first_id}-{final_id}, but the dictionary '
            f'contains IDs 0-{capacity - 1}')
    existing_ids = {
        marker_id
        for offset in BOARD_ID_OFFSETS
        for marker_id in range(offset, offset + MARKERS_PER_BOARD)
    }
    requested_ids = set(range(first_id, final_id + 1))
    overlap = sorted(existing_ids & requested_ids)
    if overlap:
        raise ValueError(
            f'requested marker IDs overlap existing boards: {overlap}')

    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = []
    for index in range(count):
        offset = first_id + index * MARKERS_PER_BOARD
        image, ids = board_image(offset, pixels, dictionary)
        expected = ids.tolist()
        observed = detected_ids(image, dictionary)
        if observed != expected:
            raise RuntimeError(
                f'generated-board verification failed: expected {expected}, '
                f'detected {observed}')
        name = f'{prefix}_{index + 1}'
        png_name = f'{name}_ids_{expected[0]}-{expected[-1]}.png'
        if not cv2.imwrite(str(output_dir / png_name), image):
            raise OSError(f'failed to write {output_dir / png_name}')
        patterns.append({
            'name': name,
            'ids': expected,
            'image': image,
            'png': png_name,
        })

    pdf_name = f'{prefix}_{count}_boards_{page_name}.pdf'
    write_print_pdf(output_dir / pdf_name, patterns, page_name)
    manifest = {
        'dictionary': 'DICT_4X4_50',
        'opencv_dictionary_constant': int(ARUCO_DICT),
        'board_size_mm': [BOARD_SIZE_MM, BOARD_SIZE_MM],
        'squares': [SQUARES_X, SQUARES_Y],
        'square_length_mm': SQUARE_LENGTH_M * 1000.0,
        'marker_length_mm': MARKER_SIZE_MM,
        'print_pdf': pdf_name,
        'print_instruction': 'Print at 100% / Actual Size; disable scaling.',
        'patterns': [
            {key: value for key, value in pattern.items() if key != 'image'}
            for pattern in patterns
        ],
        'existing_board_id_ranges': [
            [offset, offset + MARKERS_PER_BOARD - 1]
            for offset in BOARD_ID_OFFSETS
        ],
    }
    manifest_path = output_dir / f'{prefix}_manifest.json'
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    return manifest_path, output_dir / pdf_name


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output-dir', type=Path,
        default=REPOSITORY / 'generated_charuco',
        help='output directory (default: repository/generated_charuco).')
    parser.add_argument(
        '--count', type=int, default=2,
        help='number of distinct boards; two supports a multi-face fixture.')
    parser.add_argument(
        '--first-id', type=int, default=DEFAULT_FIRST_ID,
        help=f'first marker ID (default {DEFAULT_FIRST_ID}; existing end at 16).')
    parser.add_argument(
        '--pixels', type=int, default=2000,
        help='PNG width/height in pixels (default 2000).')
    parser.add_argument(
        '--page', choices=sorted(PAGE_SIZES_MM), default='letter',
        help='print-ready PDF page size, landscape orientation.')
    parser.add_argument(
        '--prefix', default='field_generator_charuco',
        help='output filename and board-label prefix.')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    manifest, pdf = generate(
        args.output_dir, args.count, args.first_id, args.pixels,
        args.page, args.prefix)
    print(f'Print-ready PDF: {pdf}')
    print(f'Manifest: {manifest}')
    print(
        f'Board geometry: {BOARD_SIZE_MM:.1f} x {BOARD_SIZE_MM:.1f} mm; '
        f'{SQUARES_X}x{SQUARES_Y} squares; '
        f'{SQUARE_LENGTH_M * 1000.0:.2f} mm square; '
        f'{MARKER_SIZE_MM:.2f} mm marker')


if __name__ == '__main__':
    main()
