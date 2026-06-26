import math
from pathlib import Path

import pytest

from professional_letter_classifier import ProfessionalLetterClassifier


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "professional_model.h5"


def _landmarks():
    return [
        {
            "x": 0.25 + (idx % 5) * 0.04,
            "y": 0.30 + (idx // 5) * 0.05,
            "z": 0.0,
        }
        for idx in range(21)
    ]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="professional model artifact is not present")
def test_numpy_classifier_loads_h5_contract_and_predicts():
    classifier = ProfessionalLetterClassifier(backend="numpy")

    assert classifier.load_model(MODEL_PATH)

    status = classifier.status()
    assert status["ready"] is True
    assert status["expected_input_shape"] == [None, 72]
    assert status["output_shape"] == [None, status["label_count"]]
    assert status["label_count"] == 27

    prediction = classifier.predict(_landmarks())
    assert prediction["success"] is True
    assert prediction["letter"] in classifier.labels
    assert 0.0 <= prediction["confidence"] <= 1.0
    assert math.isclose(sum(prediction["all_probabilities"].values()), 1.0, rel_tol=1e-5)
    assert prediction["letter"] == "H"
    assert math.isclose(prediction["confidence"], 0.40143513679504395, rel_tol=1e-6)
    assert prediction["top_3"] == [
        {"letter": "H", "confidence": pytest.approx(0.40143513679504395, rel=1e-6)},
        {"letter": "Q", "confidence": pytest.approx(0.3006454110145569, rel=1e-6)},
        {"letter": "del", "confidence": pytest.approx(0.15696808695793152, rel=1e-6)},
    ]
