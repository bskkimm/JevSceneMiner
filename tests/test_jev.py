from types import SimpleNamespace

import numpy as np
import pytest

from jevsceneminer.cli import indicator_segments
from jevsceneminer.jev import AnswerCache, JevClassifier, JevError, Labels, cache_key

LABELS = Labels("0.1.1", "lateral?", {"follow_lane": "stay", "turn_left": "turn"},
                "longitudinal?", {"standing_still": "stopped", "driving_forward_keeping_speed": "steady"})


class FakeClient:
    def __init__(self):
        self.calls = 0

    async def system_one(self, state, questions, model=None):
        self.calls += 1
        if "FAIL" in state:
            raise RuntimeError("boom")
        lat = "turn_left" if "left" in state else "follow_lane"
        return SimpleNamespace(
            answers={"lateral": SimpleNamespace(choice=lat, probabilities={lat: 0.9}),
                     "longitudinal": SimpleNamespace(choice="standing_still", probabilities={"standing_still": 1.0})},
            model="jev-test", usage=SimpleNamespace(input_tokens=100))

    async def aclose(self):
        pass


def test_classifier_uses_the_cache(tmp_path):
    client = FakeClient()
    cache = AnswerCache(tmp_path / "c.jsonl")
    out = JevClassifier(LABELS, cache, client_factory=lambda: client, log=lambda _: None).classify(
        {1: "turning left", 2: "going straight"})
    assert out[1].lateral == "turn_left" and out[2].lateral == "follow_lane"
    assert out[1].model == "jev-test" and out[1].input_tokens == 100
    again = JevClassifier(LABELS, AnswerCache(tmp_path / "c.jsonl"), client_factory=lambda: client,
                          log=lambda _: None).classify({1: "turning left"})
    assert client.calls == 2 and again[1] == out[1]


def test_classifier_raises_after_failures(tmp_path):
    with pytest.raises(JevError):
        JevClassifier(LABELS, AnswerCache(tmp_path / "c.jsonl"), client_factory=FakeClient,
                      log=lambda _: None).classify({1: "ok", 2: "FAIL"})


def test_cache_key_changes_with_questions_and_script():
    other = Labels("0.1.1", "lateral?", {"follow_lane": "stay"}, "longitudinal?", LABELS.longitudinal)
    assert cache_key(None, LABELS, "a") != cache_key(None, other, "a")
    assert cache_key(None, LABELS, "a") != cache_key(None, LABELS, "b")
    assert cache_key(None, LABELS, "a") == cache_key(None, LABELS, "a")


def test_indicator_segments():
    s = SimpleNamespace(indicator_t=np.array([0, 1, 2, 3, 4]) * 10**9, indicator=np.array([1, 2, 2, 1, 3]))
    assert indicator_segments(s) == [[1.0, 3.0, 2], [4.0, 4.0, 3]]


class LateralOnlyClient(FakeClient):
    async def system_one(self, state, questions, model=None):
        assert set(questions) == {"lateral"}
        self.calls += 1
        return SimpleNamespace(answers={"lateral": SimpleNamespace(choice="turn_left", probabilities={"turn_left": 0.7})},
                               model="jev-test", usage=None)


def test_lateral_only_asks_one_question(tmp_path):
    labels = Labels(*[getattr(LABELS, f) for f in ("taxonomy_version", "lateral_instructions", "lateral",
                                                    "longitudinal_instructions", "longitudinal")], lateral_only=True)
    out = JevClassifier(labels, AnswerCache(tmp_path / "c.jsonl"), client_factory=LateralOnlyClient,
                        log=lambda _: None).classify({1: "x"})
    assert out[1].lateral == "turn_left" and out[1].longitudinal is None and out[1].longitudinal_probs == {}
    assert cache_key(None, labels, "x") != cache_key(None, LABELS, "x")
