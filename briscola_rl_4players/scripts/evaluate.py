#!/usr/bin/env python3
"""Evaluate a Briscola checkpoint against baselines or another checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from briscola.baselines import GreedyTrickPolicy, RandomPolicy, SimpleHeuristicPolicy
from briscola.evaluation import EvaluationResult, evaluate_team_match, matchup_matrix
from briscola.features import BriscolaFeatureExtractor
from briscola.policies import LinearSoftmaxPolicy, Policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Briscola policy.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--opponent",
        default="random",
        help=(
            "Opponent to evaluate against: random, greedy, heuristic, "
            "or path to another checkpoint."
        ),
    )
    parser.add_argument("--seed-start", type=int, default=100_000)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic policy sampling instead of greedy evaluation.")
    parser.add_argument("--matrix", action="store_true", help="Evaluate learner, baselines, and pool snapshots as a matchup matrix.")
    parser.add_argument("--max-snapshots", type=int, default=5, help="Maximum pool snapshots to include in matrix mode.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    learner = policy_from_checkpoint(checkpoint, name="learner")
    seeds = list(range(args.seed_start, args.seed_start + args.games))
    greedy = not args.stochastic

    if args.matrix:
        policies = build_matrix_policies(checkpoint, max_snapshots=args.max_snapshots)
        matrix = matchup_matrix(policies, seeds=seeds, greedy=greedy)
        payload = {
            f"{name_a} vs {name_b}": result_to_dict(result)
            for (name_a, name_b), result in matrix.items()
        }
        print(json.dumps(payload, indent=2))
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return

    opponent = load_opponent(args.opponent)
    result = evaluate_team_match(
        policy_a=learner,
        policy_b=opponent,
        seeds=seeds,
        paired=True,
        greedy=greedy,
    )
    payload = {
        "checkpoint": str(args.checkpoint),
        "opponent": getattr(opponent, "name", args.opponent),
        "paired_deals": True,
        "greedy": greedy,
        **result_to_dict(result),
    }
    print(json.dumps(payload, indent=2))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def policy_from_checkpoint(
    checkpoint: dict[str, object],
    name: str = "checkpoint_policy",
) -> LinearSoftmaxPolicy:
    """Build the learner policy from a train.py checkpoint."""

    feature_names = checkpoint["feature_names"]
    learner = checkpoint["learner"]
    if not isinstance(feature_names, list) or not isinstance(learner, dict):
        raise ValueError("Invalid checkpoint format")
    feature_extractor = BriscolaFeatureExtractor(feature_names=list(feature_names))
    return LinearSoftmaxPolicy(
        theta=list(learner["theta"]),
        feature_extractor=feature_extractor,
        name=name,
    )


def snapshot_policy(
    checkpoint: dict[str, object],
    snapshot_index: int,
    name: str,
) -> LinearSoftmaxPolicy:
    feature_names = checkpoint["feature_names"]
    pool = checkpoint["pool"]
    if not isinstance(feature_names, list) or not isinstance(pool, list):
        raise ValueError("Invalid checkpoint format")
    snapshot = pool[snapshot_index]
    if not isinstance(snapshot, dict):
        raise ValueError("Invalid snapshot format")
    feature_extractor = BriscolaFeatureExtractor(feature_names=list(feature_names))
    return LinearSoftmaxPolicy(
        theta=list(snapshot["theta"]),
        feature_extractor=feature_extractor,
        name=name,
    )


def load_opponent(opponent: str) -> Policy:
    if opponent == "random":
        return RandomPolicy()
    if opponent == "greedy":
        return GreedyTrickPolicy()
    if opponent == "heuristic":
        return SimpleHeuristicPolicy()

    path = Path(opponent)
    if path.exists():
        return policy_from_checkpoint(load_checkpoint(path), name=path.stem)

    raise ValueError(
        "Unknown opponent. Use random, greedy, heuristic, or a checkpoint path."
    )


def build_matrix_policies(
    checkpoint: dict[str, object],
    max_snapshots: int,
) -> dict[str, Policy]:
    policies: dict[str, Policy] = {
        "learner": policy_from_checkpoint(checkpoint, name="learner"),
        "random": RandomPolicy(),
        "greedy": GreedyTrickPolicy(),
        "heuristic": SimpleHeuristicPolicy(),
    }

    pool = checkpoint.get("pool", [])
    if isinstance(pool, list):
        for index, snapshot in enumerate(pool[:max_snapshots]):
            if isinstance(snapshot, dict):
                name = str(snapshot.get("name", f"snapshot_{index}"))
                policies[f"pool_{index}_{name}"] = snapshot_policy(
                    checkpoint,
                    snapshot_index=index,
                    name=name,
                )
    return policies


def result_to_dict(result: EvaluationResult) -> dict[str, object]:
    return {
        "games": result.games,
        "win_rate": result.win_rate,
        "draw_rate": result.draw_rate,
        "loss_rate": result.loss_rate,
        "mean_point_difference": result.mean_point_difference,
        "standard_error": result.standard_error,
        "confidence_interval_95": list(result.confidence_interval_95),
    }


if __name__ == "__main__":
    main()

