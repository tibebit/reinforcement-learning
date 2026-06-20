#!/usr/bin/env python3
"""Train a four-player Briscola policy with pool-based self-play.

The script intentionally uses only the Python standard library so it can run in
minimal university environments. It saves a JSON checkpoint containing the final
learner parameters and all frozen snapshots in the pool.
"""

from __future__ import annotations

import argparse
import importlib
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


REWARD_MODES = {"combined_terminal", "terminal_win", "trick_point_dense"}
BASELINES = {"batch_mean", "none"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a four-player Briscola RL policy with REINFORCE.",
    )
    parser.add_argument("--config", type=Path, default=None, help="Optional YAML config file.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--updates", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--snapshot-interval", type=int, default=None)
    parser.add_argument("--max-pool-size", type=int, default=None)
    parser.add_argument("--pool-best-count", type=int, default=None)
    parser.add_argument("--pool-recent-count", type=int, default=None)
    parser.add_argument(
        "--pool-eval-games",
        type=int,
        default=None,
        help="Evaluate each saved snapshot on fixed seeds; 0 disables scoring.",
    )
    parser.add_argument("--pool-eval-seed-start", type=int, default=None)
    parser.add_argument(
        "--reward-mode",
        choices=["combined_terminal", "terminal_win", "trick_point_dense"],
        default=None,
    )
    parser.add_argument("--lambda-margin", type=float, default=None)
    parser.add_argument("--baseline", choices=["batch_mean", "none"], default=None)
    parser.add_argument("--max-update-norm", type=float, default=None)
    parser.add_argument("--init-scale", type=float, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--print-every", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}

    yaml = importlib.import_module("yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Training config must be a YAML mapping")
    return data


def resolve_args(args: argparse.Namespace) -> argparse.Namespace:
    config = load_config(args.config)

    policy_type = str(config.get("policy_type", "linear_softmax"))
    if policy_type != "linear_softmax":
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    resolved = argparse.Namespace()
    resolved.config = args.config
    resolved.seed = _resolve_value(args.seed, config, "seed", 0, int)
    resolved.updates = _resolve_value(args.updates, config, "training_updates", 100, int)
    resolved.batch_size = _resolve_value(args.batch_size, config, "batch_size", 100, int)
    resolved.learning_rate = _resolve_value(
        args.learning_rate,
        config,
        "learning_rate",
        0.01,
        float,
    )
    resolved.snapshot_interval = _resolve_value(
        args.snapshot_interval,
        config,
        "snapshot_interval",
        10,
        int,
    )
    resolved.max_pool_size = _resolve_value(
        args.max_pool_size,
        config,
        "max_pool_size",
        20,
        int,
    )
    resolved.pool_best_count = _resolve_value(
        args.pool_best_count,
        config,
        "pool_best_count",
        None,
        int,
    )
    resolved.pool_recent_count = _resolve_value(
        args.pool_recent_count,
        config,
        "pool_recent_count",
        None,
        int,
    )
    resolved.pool_eval_games = _resolve_value(
        args.pool_eval_games,
        config,
        "pool_eval_games",
        0,
        int,
    )
    resolved.pool_eval_seed_start = _resolve_value(
        args.pool_eval_seed_start,
        config,
        "pool_eval_seed_start",
        900_000,
        int,
    )
    resolved.reward_mode = _resolve_value(
        args.reward_mode,
        config,
        "reward_mode",
        "combined_terminal",
        str,
    )
    resolved.lambda_margin = _resolve_value(
        args.lambda_margin,
        config,
        "lambda_margin",
        0.2,
        float,
    )
    resolved.baseline = _resolve_value(args.baseline, config, "baseline", "batch_mean", str)
    resolved.max_update_norm = _resolve_value(
        args.max_update_norm,
        config,
        "max_update_norm",
        5.0,
        float,
    )
    resolved.init_scale = _resolve_value(args.init_scale, config, "init_scale", 0.01, float)
    resolved.output = Path(
        _resolve_value(
            args.output,
            config,
            "output",
            PROJECT_ROOT / "experiments/results/checkpoint.json",
            Path,
        )
    )
    resolved.log = Path(
        _resolve_value(
            args.log,
            config,
            "log",
            PROJECT_ROOT / "experiments/results/train_log.jsonl",
            Path,
        )
    )
    resolved.print_every = _resolve_value(args.print_every, config, "print_every", 10, int)
    validate_resolved_args(resolved)
    return resolved


def validate_resolved_args(args: argparse.Namespace) -> None:
    if args.reward_mode not in REWARD_MODES:
        raise ValueError(f"Unsupported reward_mode: {args.reward_mode}")
    if args.baseline not in BASELINES:
        raise ValueError(f"Unsupported baseline: {args.baseline}")


def _resolve_value(
    cli_value: object,
    config: dict[str, object],
    key: str,
    default: object,
    cast,
):
    value = cli_value if cli_value is not None else config.get(key, default)
    if value is None:
        return None
    return cast(value)


def main() -> None:
    args = resolve_args(parse_args())
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

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

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
