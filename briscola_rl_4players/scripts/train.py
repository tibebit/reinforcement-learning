#!/usr/bin/env python3
"""Train a four-player Briscola policy with pool-based self-play.

The script intentionally uses only the Python standard library so it can run in
minimal university environments. It saves a JSON checkpoint containing the final
learner parameters and all frozen snapshots in the pool.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from briscola.baselines import GreedyTrickPolicy, RandomPolicy, SimpleHeuristicPolicy
from briscola.evaluation import evaluate_team_match
from briscola.features import BriscolaFeatureExtractor
from briscola.policies import LinearSoftmaxPolicy
from briscola.reinforce import RewardConfig
from briscola.self_play import SelfPlayConfig, SelfPlayTrainer, SnapshotPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a four-player Briscola RL policy with REINFORCE.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--snapshot-interval", type=int, default=10)
    parser.add_argument("--max-pool-size", type=int, default=20)
    parser.add_argument("--pool-best-count", type=int, default=None)
    parser.add_argument("--pool-recent-count", type=int, default=None)
    parser.add_argument(
        "--pool-eval-games",
        type=int,
        default=0,
        help="Evaluate each saved snapshot on fixed seeds; 0 disables scoring.",
    )
    parser.add_argument("--pool-eval-seed-start", type=int, default=900_000)
    parser.add_argument(
        "--reward-mode",
        choices=["combined_terminal", "terminal_win", "trick_point_dense"],
        default="combined_terminal",
    )
    parser.add_argument("--lambda-margin", type=float, default=0.2)
    parser.add_argument("--baseline", choices=["batch_mean", "none"], default="batch_mean")
    parser.add_argument("--max-update-norm", type=float, default=5.0)
    parser.add_argument("--init-scale", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "experiments/results/checkpoint.json")
    parser.add_argument("--log", type=Path, default=PROJECT_ROOT / "experiments/results/train_log.jsonl")
    parser.add_argument("--print-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)

    feature_extractor = BriscolaFeatureExtractor()
    learner = LinearSoftmaxPolicy.initialize(
        feature_extractor=feature_extractor,
        rng=random.Random(args.seed),
        scale=args.init_scale,
        name="learner",
    )
    pool = SnapshotPool(
        feature_extractor=feature_extractor,
        max_size=args.max_pool_size,
        best_count=args.pool_best_count,
        recent_count=args.pool_recent_count,
    )

    reward_config = RewardConfig(
        mode=args.reward_mode,
        lambda_margin=args.lambda_margin,
    )
    train_config = SelfPlayConfig(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        snapshot_interval=args.snapshot_interval,
        max_update_norm=args.max_update_norm,
        baseline=args.baseline,
        reward_config=reward_config,
    )
    trainer = SelfPlayTrainer(
        learner=learner,
        pool=pool,
        config=train_config,
        seed=args.seed,
        snapshot_score_fn=build_snapshot_score_fn(args),
    )

    with args.log.open("w", encoding="utf-8") as log_file:
        for update in range(1, args.updates + 1):
            stats = trainer.train_update()
            record = {
                "update": update,
                "pool_size": len(pool.snapshots),
                "best_snapshot": best_snapshot_record(pool),
                **asdict(stats),
            }
            log_file.write(json.dumps(record) + "\n")

            if update == 1 or update % args.print_every == 0 or update == args.updates:
                print(
                    "update={update:04d} "
                    "return={ret:+.4f} "
                    "score_diff={diff:+.2f} "
                    "grad={grad:.4f} "
                    "pool={pool_size}".format(
                        update=update,
                        ret=stats.mean_return,
                        diff=stats.mean_score_diff,
                        grad=stats.gradient_norm,
                        pool_size=len(pool.snapshots),
                    )
                )

    save_checkpoint(
        path=args.output,
        learner=learner,
        pool=pool,
        args=args,
        feature_extractor=feature_extractor,
    )
    print(f"Saved checkpoint to {args.output}")
    print(f"Saved training log to {args.log}")


def save_checkpoint(
    path: Path,
    learner: LinearSoftmaxPolicy,
    pool: SnapshotPool,
    args: argparse.Namespace,
    feature_extractor: BriscolaFeatureExtractor,
) -> None:
    """Write all data needed by evaluate.py to a JSON checkpoint."""

    config = vars(args).copy()
    config["output"] = str(args.output)
    config["log"] = str(args.log)

    checkpoint = {
        "format": "briscola_rl_4players_checkpoint_v1",
        "config": config,
        "feature_names": feature_extractor.feature_names,
        "learner": {
            "name": learner.name,
            "theta": learner.theta,
        },
        "pool": [
            {
                "name": snapshot.name,
                "theta": snapshot.theta,
                "update_index": snapshot.update_index,
                "evaluation_score": snapshot.evaluation_score,
            }
            for snapshot in pool.snapshots
        ],
    }
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def build_snapshot_score_fn(args: argparse.Namespace):
    """Create the optional external score used to preserve strong snapshots."""

    if args.pool_eval_games <= 0:
        return None

    seeds = list(
        range(args.pool_eval_seed_start, args.pool_eval_seed_start + args.pool_eval_games)
    )
    baselines = [RandomPolicy(), GreedyTrickPolicy(), SimpleHeuristicPolicy()]

    def score(policy: LinearSoftmaxPolicy, update_index: int) -> float:
        results = [
            evaluate_team_match(
                policy_a=policy,
                policy_b=baseline,
                seeds=seeds,
                paired=True,
                greedy=True,
            ).mean_point_difference
            for baseline in baselines
        ]
        return sum(results) / len(results)

    return score


def best_snapshot_record(pool: SnapshotPool) -> dict[str, object] | None:
    scored = [
        snapshot for snapshot in pool.snapshots if snapshot.evaluation_score is not None
    ]
    if not scored:
        return None
    best = max(scored, key=lambda snapshot: (snapshot.evaluation_score, snapshot.update_index))
    return {
        "name": best.name,
        "update_index": best.update_index,
        "evaluation_score": best.evaluation_score,
    }


if __name__ == "__main__":
    main()
