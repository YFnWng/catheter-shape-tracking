import json
from pathlib import Path
import sqlite3
import struct
from tempfile import TemporaryDirectory
import unittest

from shape_tracking.sequence import (
    _decode_std_string, load_collection_markers, select_frame_records)
from shape_tracking.session import FrameRecord


class SequenceTests(unittest.TestCase):
    def test_decodes_ros2_std_string_cdr(self):
        text = json.dumps({"event": "run_start", "stamp_ns": 123})
        encoded = text.encode("utf-8") + b"\0"
        serialized = b"\x00\x01\x00\x00" + struct.pack("<I", len(encoded)) + encoded
        self.assertEqual(_decode_std_string(serialized), text)

    def test_selects_trajectory_marker_window(self):
        records = [
            FrameRecord(index, timestamp)
            for index, timestamp in enumerate((50, 100, 150, 200, 250))]
        markers = {
            "run_start": {"stamp_ns": 100},
            "return_start": {"stamp_ns": 200},
            "run_end": {"stamp_ns": 250},
        }
        selected = select_frame_records(
            records, markers, "trajectory", None, None, stride=1,
            max_frames=None)
        self.assertEqual(
            [record.timestamp_ns for record in selected], [100, 150])

    def test_loads_collection_markers_from_robot_bag_directory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            bag = root / "robot_bag"
            bag.mkdir()
            database = bag / "bag_0.db3"
            connection = sqlite3.connect(database)
            connection.executescript(
                "create table topics(id integer primary key, name text);"
                "create table messages(id integer primary key, "
                "topic_id integer, timestamp integer, data blob);"
                "insert into topics values(1, '/collection/events');")
            text = json.dumps({"event": "run_start", "stamp_ns": 123})
            encoded = text.encode("utf-8") + b"\0"
            serialized = (
                b"\x00\x01\x00\x00"
                + struct.pack("<I", len(encoded)) + encoded)
            connection.execute(
                "insert into messages values(1, 1, 124, ?)",
                (serialized,))
            connection.commit()
            connection.close()
            markers = load_collection_markers(root)
            self.assertEqual(markers["run_start"]["stamp_ns"], 123)
            self.assertEqual(
                markers["run_start"]["receipt_timestamp_ns"], 124)


if __name__ == "__main__":
    unittest.main()
