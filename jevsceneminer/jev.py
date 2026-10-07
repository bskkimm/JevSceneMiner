"""Ask Jev which labels apply at NOW, for many scripts in parallel, with a disk cache.

Each call has two Choice questions (lateral, longitudinal) built from labels.yaml.
Answers are appended to a JSONL cache as they arrive, so an interrupted run resumes
where it stopped and re-stitching never calls Jev again.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import yaml
from typesafe_sdk import AsyncTypeSafeClient, Choice, RetryPolicy

# 529 = TypeSafe is overloaded ("high traffic, try again later"): retry it like a 503.
RETRY = RetryPolicy(max_retries=6, http_statuses={429, 500, 502, 503, 504, 529},
                    backoff_initial=1.0, backoff_max=30.0, timeout=30.0)


INDICATOR_CODES = {"left": 2, "right": 3}   # TurnIndicatorsReport: 2 = LEFT, 3 = RIGHT


@dataclass(frozen=True)
class Labels:
    version: str
    lateral_instructions: str
    lateral: dict[str, str]
    longitudinal_instructions: str
    longitudinal: dict[str, str]
    lateral_only: bool = False   # ask only the lateral question
    indicator: dict = field(default_factory=dict)   # lateral label -> indicator code it needs (2 | 3)
    traffic_side: str = "left"
    has_indicator: bool = True   # definitions written for logs that record the turn indicator

    def spec(self) -> dict:
        """Plain-JSON form of the questions, used in the cache key."""
        spec = {"lateral": [self.lateral_instructions, self.lateral]}
        if not self.lateral_only:
            spec["longitudinal"] = [self.longitudinal_instructions, self.longitudinal]
        return spec

    def questions(self) -> dict[str, Choice]:
        qs = {"lateral": Choice(instructions=self.lateral_instructions, criteria=self.lateral)}
        if not self.lateral_only:
            qs["longitudinal"] = Choice(instructions=self.longitudinal_instructions, criteria=self.longitudinal)
        return qs


def _sides(traffic_side: str) -> dict[str, str]:
    if traffic_side not in ("left", "right"):
        raise ValueError(f"traffic_side must be left or right, not {traffic_side!r}")
    far = "right" if traffic_side == "left" else "left"
    return {"curb": traffic_side, "far": far, "CURB": traffic_side.upper(), "FAR": far.upper()}


_VARIANT = re.compile(r"\[\[(.*?)(?:\|\|(.*?))?\]\]", re.S)


def _text(text: str, sides: dict, has_indicator: bool) -> str:
    """Fill in the traffic side and pick the [[with||without indicator]] variants."""
    text = _VARIANT.sub(lambda m: m.group(1) if has_indicator else (m.group(2) or ""), text)
    return " ".join(text.format(**sides).split())


def _definitions(entries: dict, sides: dict, has_indicator: bool) -> tuple[dict[str, str], dict[str, int]]:
    """Label -> definition text, and label -> indicator code it needs."""
    texts, needs = {}, {}
    for name, entry in entries.items():
        if isinstance(entry, dict):
            texts[name] = _text(entry["definition"], sides, has_indicator)
            if entry.get("indicator"):
                needs[name] = INDICATOR_CODES[sides.get(entry["indicator"], entry["indicator"])]
        else:
            texts[name] = _text(entry, sides, has_indicator)
    return texts, needs


def load_labels(path: Path, lateral_only: bool = False, traffic_side: str | None = None,
                has_indicator: bool = True) -> Labels:
    doc = yaml.safe_load(Path(path).read_text())
    side = traffic_side or doc.get("traffic_side", "left")
    sides = _sides(side)
    lateral, needs = _definitions(doc["lateral"]["labels"], sides, has_indicator)
    longitudinal, _ = _definitions(doc["longitudinal"]["labels"], sides, has_indicator)
    return Labels(
        lateral_only=lateral_only,
        version=str(doc["version"]),
        lateral_instructions=doc["lateral"]["instructions"],
        lateral=lateral,
        longitudinal_instructions=doc["longitudinal"]["instructions"],
        longitudinal=longitudinal,
        indicator=needs if has_indicator else {},
        traffic_side=side,
        has_indicator=has_indicator,
    )


@dataclass(frozen=True)
class Answer:
    lateral: str
    lateral_probs: dict[str, float]
    longitudinal: str | None          # None when only the lateral question was asked
    longitudinal_probs: dict[str, float]
    model: str | None = None
    input_tokens: int | None = None


class JevError(RuntimeError):
    """Jev calls failed after retries."""


def cache_key(model: str | None, labels: Labels, script: str) -> str:
    payload = json.dumps({"model": model or "default", "questions": labels.spec(), "state": script},
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


class AnswerCache:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._answers: dict[str, Answer] = {}
        # Successful responses written by this instance, including repeated keys.
        self.written_answers: list[Answer] = []
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                if line.strip():
                    row = json.loads(line)
                    self._answers[row["key"]] = Answer(**row["answer"])

    def get(self, key: str) -> Answer | None:
        return self._answers.get(key)

    def put(self, key: str, answer: Answer) -> None:
        self._answers[key] = answer
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(json.dumps({"key": key, "answer": asdict(answer)}, ensure_ascii=False) + "\n")
        self.written_answers.append(answer)


def parse_response(response) -> Answer:
    lat, lon = response.answers["lateral"], response.answers.get("longitudinal")
    usage = getattr(response, "usage", None)
    return Answer(
        lateral=lat.choice,
        lateral_probs={k: float(v) for k, v in lat.probabilities.items()},
        longitudinal=lon.choice if lon is not None else None,
        longitudinal_probs={k: float(v) for k, v in lon.probabilities.items()} if lon is not None else {},
        model=getattr(response, "model", None),
        input_tokens=getattr(usage, "input_tokens", None) if usage is not None else None,
    )


class JevClassifier:
    def __init__(self, labels: Labels, cache: AnswerCache, model: str | None = None, workers: int = 8,
                 client_factory: Callable[[], object] | None = None, log: Callable[[str], None] = print):
        self.labels = labels
        self.cache = cache
        self.model = model
        self.workers = workers
        self._client_factory = client_factory or (lambda: AsyncTypeSafeClient(retry=RETRY, timeout=30.0))
        self._log = log

    def classify(self, scripts: dict[int, str]) -> dict[int, Answer]:
        """Answers keyed like ``scripts``. Cached answers are reused without a call."""
        results: dict[int, Answer] = {}
        todo = []
        for idx, script in scripts.items():
            key = cache_key(self.model, self.labels, script)
            hit = self.cache.get(key)
            if hit is not None:
                results[idx] = hit
            else:
                todo.append((idx, key, script))
        if todo:
            self._log(f"  jev: {len(todo)} calls ({len(results)} cached)")
            asyncio.run(self._run(todo, results))
        return results

    async def _run(self, todo, results) -> None:
        client = self._client_factory()
        questions = self.labels.questions()
        sem = asyncio.Semaphore(self.workers)

        async def one(idx, key, script):
            async with sem:
                response = await client.system_one(state=script, questions=questions, model=self.model)
            return idx, key, parse_response(response)

        tasks = [asyncio.create_task(one(*item)) for item in todo]
        start, done = time.time(), 0
        try:
            for fut in asyncio.as_completed(tasks):
                idx, key, answer = await fut
                self.cache.put(key, answer)
                results[idx] = answer
                done += 1
                if done % 250 == 0 or done == len(todo):
                    rate = done / max(time.time() - start, 1e-9)
                    self._log(f"  jev: {done}/{len(todo)} done ({rate:.1f} calls/s)")
        except Exception as exc:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise JevError(f"Jev call failed after retries ({done} answers saved to the cache): {exc!r}") from exc
        finally:
            close = getattr(client, "aclose", None)
            if close is not None:
                await close()
