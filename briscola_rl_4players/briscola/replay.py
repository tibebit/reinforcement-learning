"""Replay serialization helpers."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .env import FourPlayerBriscolaEnv
from .policies import Policy


@dataclass
class Replay:
    seed: int
    start_player: int
    trump_suit: str
    actions: list[dict[str, object]] = field(default_factory=list)
    final_scores: tuple[int, int] | None = None


def record_match(
    policies_by_player: dict[int, Policy],
    seed: int,
    start_player: int | None = None,
    greedy: bool = True,
) -> Replay:
    """Play and record a complete match."""

    if start_player is None:
        start_player = seed % 4
    env = FourPlayerBriscolaEnv(seed=seed, start_player=start_player)
    rng = random.Random(seed + 30_000)
    replay = Replay(seed=seed, start_player=start_player, trump_suit=env.trump_suit)

    while not env.done:
        player_id = env.current_player
        obs = env.observe(player_id)
        action = policies_by_player[player_id].select_action(obs, rng, greedy=greedy)
        result = env.step(action)
        replay.actions.append(
            {
                "player": player_id,
                "card": action.id,
                "trick_completed": result.trick_completed,
                "trick_winner": result.trick_winner,
                "trick_points": result.trick_points,
                "scores": list(result.scores),
            }
        )

    replay.final_scores = tuple(env.scores)
    return replay


def save_replay(replay: Replay, path: str | Path) -> None:
    data = {
        "seed": replay.seed,
        "start_player": replay.start_player,
        "trump_suit": replay.trump_suit,
        "actions": replay.actions,
        "final_scores": list(replay.final_scores) if replay.final_scores else None,
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_replay(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

