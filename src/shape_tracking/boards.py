"""ChArUco board definition, decoded from charuco10_DICT4X4_two_boards_LETTER.pdf.

Two independent 4x4 ChArUco boards, each 100 mm square when printed at 100%:
    square length = 25.0 mm, marker length = 18.75 mm (0.75 ratio),
    dictionary DICT_4X4_50, board #1 ids 1..8, board #2 ids 9..16.
    (IDs verified by detecting markers in the decoded PDF renders — the generator
    started numbering at 1, not 0.)

*** Print the PDF at 100% / "Actual size" (no fit-to-page), then measure a
printed square with calipers and set SQUARE_LENGTH_M accordingly: pose scale
depends entirely on this number. ***
"""

import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "OpenCV not found. Install with:  pip install opencv-contrib-python"
    ) from e

if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "CharucoDetector"):
    raise ImportError(
        "cv2.aruco.CharucoDetector missing. Install opencv-contrib-python>=4.7:\n"
        "  pip uninstall -y opencv-python opencv-contrib-python\n"
        "  pip install opencv-contrib-python"
    )

# --------------------------------------------------------------------------- #
# Board configuration  (edit here if you re-print or re-generate the board)
# --------------------------------------------------------------------------- #
SQUARES_X = 4
SQUARES_Y = 4
SQUARE_LENGTH_M = 0.025          # 25.0 mm  -- MEASURE YOUR PRINT AND CONFIRM
MARKER_LENGTH_M = 0.01875        # 18.75 mm (0.75 * square)
ARUCO_DICT = cv2.aruco.DICT_4X4_50
BOARD_ID_OFFSETS = (1, 9)        # board#1 ids 1..8, board#2 ids 9..16
FIELD_GENERATOR_ID_OFFSET = 17   # field-generator board ids 17..24
FIELD_GENERATOR_BOARD_INDEX = 2
MARKERS_PER_BOARD = (SQUARES_X * SQUARES_Y) // 2   # = 8

AXIS_LEN_M = 0.03                # length of drawn pose axes (30 mm)


class BoardEntry:
    """A single ChArUco board plus its detector and index."""

    __slots__ = ("index", "offset", "board", "detector")

    def __init__(self, index, offset, board, detector):
        self.index = index
        self.offset = offset
        self.board = board
        self.detector = detector


def get_dictionary():
    return cv2.aruco.getPredefinedDictionary(ARUCO_DICT)


def build_boards(dictionary=None, additional=()):
    """Build the two base boards plus optional ``(index, id_offset)`` boards."""
    if dictionary is None:
        dictionary = get_dictionary()
    boards = []
    specifications = list(enumerate(BOARD_ID_OFFSETS)) + list(additional)
    seen_indices = set()
    seen_ids = set()
    for index, off in specifications:
        index, off = int(index), int(off)
        ids_set = set(range(off, off + MARKERS_PER_BOARD))
        if index in seen_indices:
            raise ValueError(f'duplicate ChArUco board index {index}')
        overlap = sorted(seen_ids & ids_set)
        if overlap:
            raise ValueError(f'overlapping ChArUco marker IDs: {overlap}')
        seen_indices.add(index)
        seen_ids.update(ids_set)
        ids = np.arange(off, off + MARKERS_PER_BOARD, dtype=np.int32)
        board = cv2.aruco.CharucoBoard(
            (SQUARES_X, SQUARES_Y), SQUARE_LENGTH_M, MARKER_LENGTH_M,
            dictionary, ids)
        detector = cv2.aruco.CharucoDetector(board)
        boards.append(BoardEntry(index, off, board, detector))
    return dictionary, boards
