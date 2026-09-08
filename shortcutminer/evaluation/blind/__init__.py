"""Evaluation-only blind shortcut discovery framework.

Production workflow modules must never import this package.  The package may read
validated production outputs only after a workflow has completed.
"""

from shortcutminer.evaluation.blind.evaluator import BlindEvaluator
from shortcutminer.evaluation.blind.batch import BlindEvaluationBatch
from shortcutminer.evaluation.blind.generator import BlindChallengeGenerator
from shortcutminer.evaluation.blind.protocol import (
    BlindChallenge,
    BlindChallengeConfig,
    BlindEvaluationBatchResult,
    BlindEvaluationResult,
    EvaluationSnapshot,
    HiddenShortcutTruth,
)
from shortcutminer.evaluation.blind.snapshot import build_evaluation_snapshot

__all__ = [
    "BlindChallenge",
    "BlindChallengeConfig",
    "BlindChallengeGenerator",
    "BlindEvaluationBatch",
    "BlindEvaluationBatchResult",
    "BlindEvaluationResult",
    "BlindEvaluator",
    "EvaluationSnapshot",
    "HiddenShortcutTruth",
    "build_evaluation_snapshot",
]
