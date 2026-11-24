import numpy as np
import pytest

from signaldesk.policy import policy_metrics, select_threshold


def test_policy_maximizes_coverage_without_splitting_confidence_ties():
    confidence = np.array([0.9, 0.8, 0.8, 0.7, 0.6])
    correct = np.array([True, True, False, True, False])
    policy = select_threshold(confidence, correct, target_accuracy=0.75, min_accepted=2)
    assert policy["threshold"] == 0.7
    assert policy["validation"]["accepted"] == 4
    assert policy["validation"]["selective_accuracy"] == 0.75


def test_impossible_policy_reviews_everything():
    policy = select_threshold([0.2, 0.9], [False, False], min_accepted=1)
    assert not policy["target_met"]
    assert policy["threshold"] > 1
    assert policy["validation"]["coverage"] == 0
    assert policy["validation"]["selective_accuracy"] is None
    assert policy_metrics([], [], 0.5)["total"] == 0


@pytest.mark.parametrize("confidence,correct", [([], []), ([float("nan")], [True]), ([1.1], [True]), ([0.5], [])])
def test_policy_rejects_invalid_probabilities(confidence, correct):
    with pytest.raises(ValueError):
        select_threshold(confidence, correct)
