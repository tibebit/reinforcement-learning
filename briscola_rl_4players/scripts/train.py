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

from briscola.features import BriscolaFeatureExtractor
from briscola.policies import LinearSoftmaxPolicy
from briscola.reinforce import RewardConfig
from briscola.self_play import SelfPlayConfig, SelfPlayTrainer, SnapshotPool


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

    policy_type = config.get("policy_type", "linear_softmax")
    if policy_type != "linear_softmax":
        raise ValueError(f"Unsupported policy_type: {policy_type}")

    resolved = argparse.Namespace()
    resolved.config = args.config
    resolved.seed = args.seed if args.seed is not None else int(config.get("seed", 0))
    resolved.updates = args.updates if args.updates is not None else int(config.get("training_updates", 100))
    resolved.batch_size = args.batch_size if args.batch_size is not None else int(config.get("batch_size", 100))
    resolved.learning_rate = (
        args.learning_rate if args.learning_rate is not None else float(config.get("learning_rate", 0.01))
    )
    resolved.snapshot_interval = (
        args.snapshot_interval if args.snapshot_interval is not None else int(config.get("snapshot_interval", 10))
    )
    resolved.max_pool_size = (
        args.max_pool_size if args.max_pool_size is not None else int(config.get("max_pool_size", 20))
    )
    resolved.reward_mode = args.reward_mode if args.reward_mode is not None else str(config.get("reward_mode", "combined_terminal"))
    resolved.lambda_margin = (
        args.lambda_margin if args.lambda_margin is not None else float(config.get("lambda_margin", 0.2))
    )
    resolved.baseline = args.baseline if args.baseline is not None else str(config.get("baseline", "batch_mean"))
    resolved.max_update_norm = (
        args.max_update_norm if args.max_update_norm is not None else float(config.get("max_update_norm", 5.0))
    )
    resolved.init_scale = args.init_scale if args.init_scale is not None else float(config.get("init_scale", 0.01))
    resolved.output = args.output if args.output is not None else Path(
        config.get("output", PROJECT_ROOT / "experiments/results/checkpoint.json")
    )
    resolved.log = args.log if args.log is not None else Path(
        config.get("log", PROJECT_ROOT / "experiments/results/train_log.jsonl")
    )
    resolved.print_every = (
        args.print_every if args.print_every is not None else int(config.get("print_every", 10))
    )
    return resolved


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
    )

    with args.log.open("w", encoding="utf-8") as log_file:
        for update in range(1, args.updates + 1):
            stats = trainer.train_update()
            record = {
                "update": update,
                "pool_size": len(pool.snapshots),
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
            }
            for snapshot in pool.snapshots
        ],
    }
    path.write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
