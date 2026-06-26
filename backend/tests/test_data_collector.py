import json

import pytest

from data_collector import DataCollector


def _landmarks():
    return [{"x": i / 20, "y": i / 25, "z": 0.0} for i in range(21)]


def test_save_sample_rejects_path_traversal_label(tmp_path):
    collector = DataCollector(tmp_path)

    with pytest.raises(ValueError):
        collector.save_sample(_landmarks(), "../escape")

    assert not (tmp_path.parent / "escape").exists()


def test_save_sample_accepts_known_label_inside_data_dir(tmp_path):
    collector = DataCollector(tmp_path)

    filepath = collector.save_sample(_landmarks(), "A")

    assert str(filepath).startswith(str(tmp_path.resolve()))
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["label"] == "A"


@pytest.mark.parametrize("landmarks", [[], [{"x": 0, "y": 0, "z": 0}], [None] * 21])
def test_save_sample_rejects_invalid_landmarks(tmp_path, landmarks):
    collector = DataCollector(tmp_path)

    with pytest.raises(ValueError):
        collector.save_sample(landmarks, "A")
