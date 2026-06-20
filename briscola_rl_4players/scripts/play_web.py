#!/usr/bin/env python3
"""Local interactive web interface for four-player Briscola.

Run with:

    python3 scripts/play_web.py

Then open http://127.0.0.1:8000 in a browser. The human plays as player 0,
while partner and opponents are controlled by baseline policies or by a trained
learner checkpoint.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import random
import sys
import uuid
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_ROOT = PROJECT_ROOT / "web"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from briscola.baselines import GreedyTrickPolicy, RandomPolicy, SimpleHeuristicPolicy
from briscola.cards import Card, PlayedCard, card_from_id
from briscola.env import FourPlayerBriscolaEnv
from briscola.features import BriscolaFeatureExtractor
from briscola.policies import LinearSoftmaxPolicy, Policy, add_scaled_in_place
from briscola.reinforce import RewardConfig, terminal_reward
from briscola.rules import partner_of, team_of


@dataclass
class LearnerDecision:
    """One trainable learner action observed during an interactive game."""

    policy: LinearSoftmaxPolicy
    observation: Any
    action: Card
    team_id: int


@dataclass
class WebSession:
    """In-memory state for one browser game."""

    env: FourPlayerBriscolaEnv
    human_id: int
    policies_by_player: dict[int, Policy]
    greedy_bots: bool
    learn_after_game: bool
    learning_rate: float
    rng: random.Random
    events: list[str] = field(default_factory=list)
    last_completed_trick: list[PlayedCard] = field(default_factory=list)
    last_trick_winner: int | None = None
    last_trick_points: int = 0
    learner_decisions: list[LearnerDecision] = field(default_factory=list)
    learner_update_applied: bool = False
    learner_updated: bool = False
    learner_update_summary: str = ""


SESSIONS: dict[str, WebSession] = {}
DEFAULT_CHECKPOINT = PROJECT_ROOT / "experiments/results/checkpoint.json"
LEARNER_POLICIES: dict[Path, LinearSoftmaxPolicy] = {}
LEARNER_UPDATE_COUNTS: dict[Path, int] = {}
WEB_REWARD_CONFIG = RewardConfig(mode="combined_terminal", lambda_margin=0.2)


def create_policy(name: str, checkpoint_path: Path | None = None) -> Policy:
    if name == "random":
        return RandomPolicy()
    if name == "greedy":
        return GreedyTrickPolicy()
    if name == "heuristic":
        return SimpleHeuristicPolicy()
    if name == "learner":
        return load_learner_policy(checkpoint_path or DEFAULT_CHECKPOINT)
    raise ValueError(f"Unknown policy: {name}")


def load_learner_policy(path: Path) -> LinearSoftmaxPolicy:
    """Load the trained learner from a JSON checkpoint created by train.py."""

    path = path.resolve()
    if path in LEARNER_POLICIES:
        return LEARNER_POLICIES[path]

    if not path.exists():
        raise ValueError(
            f"Learner checkpoint not found: {path}. Train first or choose another path."
        )
    checkpoint = json.loads(path.read_text(encoding="utf-8"))
    feature_names = checkpoint.get("feature_names")
    learner = checkpoint.get("learner")
    if not isinstance(feature_names, list) or not isinstance(learner, dict):
        raise ValueError(f"Invalid learner checkpoint: {path}")
    theta = learner.get("theta")
    if not isinstance(theta, list):
        raise ValueError(f"Checkpoint has no learner theta: {path}")
    feature_extractor = BriscolaFeatureExtractor(feature_names=list(feature_names))
    policy = LinearSoftmaxPolicy(
        theta=[float(value) for value in theta],
        feature_extractor=feature_extractor,
        name=f"learner:{path.name}",
    )
    LEARNER_POLICIES[path] = policy
    LEARNER_UPDATE_COUNTS.setdefault(path, 0)
    return policy


def new_session(
    seed: int | None,
    p2_policy: str,
    partner_policy: str,
    p4_policy: str,
    checkpoint_path: Path | None,
    greedy_bots: bool,
    learn_after_game: bool,
    learning_rate: float,
) -> tuple[str, WebSession]:
    if seed is None:
        seed = random.randrange(1_000_000_000)
    human_id = 0
    env = FourPlayerBriscolaEnv(seed=seed, start_player=seed % 4)
    policies_by_player = {
        1: create_policy(p2_policy, checkpoint_path),
        2: create_policy(partner_policy, checkpoint_path),
        3: create_policy(p4_policy, checkpoint_path),
    }
    session = WebSession(
        env=env,
        human_id=human_id,
        policies_by_player=policies_by_player,
        greedy_bots=greedy_bots,
        learn_after_game=learn_after_game,
        learning_rate=learning_rate,
        rng=random.Random(seed + 40_000),
    )
    session.events.append(f"New game started with seed {seed}.")
    session.events.append(
        "Bots: P2={p2}, P3={p3}, P4={p4}.".format(
            p2=getattr(policies_by_player[1], "name", p2_policy),
            p3=getattr(policies_by_player[2], "name", partner_policy),
            p4=getattr(policies_by_player[3], "name", p4_policy),
        )
    )
    if learn_after_game:
        session.events.append(
            f"Online learner update enabled, learning rate {learning_rate:g}."
        )
    advance_to_human(session)
    session_id = uuid.uuid4().hex
    SESSIONS[session_id] = session
    return session_id, session


def advance_to_human(session: WebSession) -> None:
    """Let bots act until it is the human's turn or the match is over."""

    env = session.env
    while not env.done and env.current_player != session.human_id:
        player_id = env.current_player
        obs = env.observe(player_id)
        policy = session.policies_by_player[player_id]
        action = policy.select_action(obs, session.rng, greedy=session.greedy_bots)
        maybe_record_learner_decision(session, policy, obs, action, player_id)
        play_card(session, action)


def maybe_record_learner_decision(
    session: WebSession,
    policy: Policy,
    obs: Any,
    action: Card,
    player_id: int,
) -> None:
    """Store trainable learner actions for the optional post-game update."""

    if not session.learn_after_game:
        return
    if not isinstance(policy, LinearSoftmaxPolicy):
        return
    if not policy.name.startswith("learner:"):
        return
    session.learner_decisions.append(
        LearnerDecision(
            policy=policy,
            observation=obs,
            action=action,
            team_id=team_of(player_id),
        )
    )


def play_card(session: WebSession, card: Card) -> None:
    """Play one card and record public UI events."""

    env = session.env
    player_id = env.current_player
    trick_snapshot = list(env.current_trick) + [PlayedCard(player_id, card)]
    result = env.step(card)
    session.events.append(f"P{player_id + 1} played {card_label(card)}.")

    if result.trick_completed:
        session.last_completed_trick = trick_snapshot
        session.last_trick_winner = result.trick_winner
        session.last_trick_points = result.trick_points
        session.events.append(
            f"P{result.trick_winner + 1} won the trick for {result.trick_points} points."
        )
        if result.done:
            session.events.append(final_message(env))
            apply_online_learner_update(session)


def apply_online_learner_update(session: WebSession) -> None:
    """Apply one small REINFORCE update to learner bots after a finished game."""

    if session.learner_update_applied:
        return
    session.learner_update_applied = True

    if not session.learn_after_game:
        return
    if not session.learner_decisions:
        session.learner_update_summary = "No learner was used, so no update was applied."
        session.events.append(session.learner_update_summary)
        return

    policies = {id(decision.policy): decision.policy for decision in session.learner_decisions}
    decisions_by_policy: dict[int, list[LearnerDecision]] = {
        policy_id: [] for policy_id in policies
    }
    for decision in session.learner_decisions:
        decisions_by_policy[id(decision.policy)].append(decision)

    total_decisions = 0
    for policy_id, decisions in decisions_by_policy.items():
        policy = policies[policy_id]
        gradient = [0.0 for _ in policy.theta]

        for decision in decisions:
            reward = terminal_reward(
                scores=tuple(session.env.scores),
                learner_team=decision.team_id,
                reward_config=WEB_REWARD_CONFIG,
            )
            grad_log_prob = policy.grad_log_probability(
                decision.observation,
                decision.action,
            )
            add_scaled_in_place(gradient, grad_log_prob, reward / len(decisions))

        policy.apply_gradient(
            gradient,
            learning_rate=session.learning_rate,
            max_update_norm=5.0,
        )
        total_decisions += len(decisions)

    for checkpoint_path, policy in LEARNER_POLICIES.items():
        if any(decision.policy is policy for decision in session.learner_decisions):
            LEARNER_UPDATE_COUNTS[checkpoint_path] = LEARNER_UPDATE_COUNTS.get(checkpoint_path, 0) + 1

    session.learner_update_summary = (
        f"Learner updated from {total_decisions} decisions. "
        "Start a new game to use the updated parameters."
    )
    session.learner_updated = True
    session.events.append(session.learner_update_summary)


def final_message(env: FourPlayerBriscolaEnv) -> str:
    if env.scores[0] > env.scores[1]:
        return f"Team A wins {env.scores[0]}-{env.scores[1]}."
    if env.scores[1] > env.scores[0]:
        return f"Team B wins {env.scores[1]}-{env.scores[0]}."
    return "The game ends in a 60-60 draw."


def state_payload(session_id: str, session: WebSession) -> dict[str, Any]:
    env = session.env
    current_player = None if env.done else env.current_player
    human_turn = current_player == session.human_id
    partner_hand_visible = env.partner_hand_visible()
    partner_hand = env.hands[partner_of(session.human_id)] if partner_hand_visible else []

    return {
        "session_id": session_id,
        "done": env.done,
        "human_id": session.human_id,
        "human_turn": human_turn,
        "current_player": current_player,
        "players": player_payloads(session.human_id, session.policies_by_player),
        "scores": {"team_a": env.scores[0], "team_b": env.scores[1]},
        "trump_suit": env.trump_suit,
        "revealed_trump": card_payload(env.revealed_trump),
        "deck_count": len(env.deck),
        "trick_index": env.trick_index,
        "current_trick": [played_payload(play) for play in env.current_trick],
        "last_completed_trick": [played_payload(play) for play in session.last_completed_trick],
        "last_trick_winner": session.last_trick_winner,
        "last_trick_points": session.last_trick_points,
        "human_hand": [card_payload(card) for card in env.hands[session.human_id]],
        "partner_hand_visible": partner_hand_visible,
        "partner_hand": [card_payload(card) for card in partner_hand],
        "legal_action_ids": [card.id for card in env.legal_actions(session.human_id)]
        if human_turn
        else [],
        "events": session.events[-8:],
        "message": status_message(env, session.human_id),
        "winner": env.winner_team() if env.done else None,
        "greedy_bots": session.greedy_bots,
        "learn_after_game": session.learn_after_game,
        "learner_update_applied": session.learner_update_applied,
        "learner_updated": session.learner_updated,
        "learner_update_summary": session.learner_update_summary,
        "learner_update_counts": {
            str(path): count for path, count in LEARNER_UPDATE_COUNTS.items()
        },
        "policy_by_player": {
            str(player_id): getattr(policy, "name", "policy")
            for player_id, policy in session.policies_by_player.items()
        },
    }


def player_payloads(
    human_id: int,
    policies_by_player: dict[int, Policy],
) -> list[dict[str, Any]]:
    payloads = []
    for player_id in range(4):
        if player_id == human_id:
            role = "You"
        elif player_id == partner_of(human_id):
            role = "Partner"
        else:
            role = "Opponent"
        payloads.append(
            {
                "id": player_id,
                "label": f"P{player_id + 1}",
                "role": role,
                "team": "A" if team_of(player_id) == 0 else "B",
                "policy": "human"
                if player_id == human_id
                else getattr(policies_by_player[player_id], "name", "policy"),
            }
        )
    return payloads


def status_message(env: FourPlayerBriscolaEnv, human_id: int) -> str:
    if env.done:
        return final_message(env)
    if env.current_player == human_id:
        return "Your turn."
    return f"P{env.current_player + 1} is playing."


def played_payload(play: PlayedCard) -> dict[str, Any]:
    return {"player_id": play.player_id, "card": card_payload(play.card)}


def card_payload(card: Card | None) -> dict[str, Any] | None:
    if card is None:
        return None
    return {
        "id": card.id,
        "rank": card.rank,
        "suit": card.suit,
        "points": card.points,
        "strength": card.strength,
        "label": card_label(card),
        "rank_label": rank_label(card.rank),
        "suit_label": suit_label(card.suit),
    }


def card_label(card: Card) -> str:
    return f"{rank_label(card.rank)} {suit_label(card.suit)}"


def rank_label(rank: str) -> str:
    labels = {
        "ace": "A",
        "three": "3",
        "king": "R",
        "knight": "C",
        "jack": "F",
        "seven": "7",
        "six": "6",
        "five": "5",
        "four": "4",
        "two": "2",
    }
    return labels[rank]


def suit_label(suit: str) -> str:
    labels = {
        "cups": "🏆",
        "coins": "🪙",
        "clubs": "♣",
        "swords": "⚔",
    }
    return labels[suit]


class BriscolaRequestHandler(BaseHTTPRequestHandler):
    server_version = "BriscolaWeb/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the app terminal quiet while the browser polls the server.
        return

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/":
            path = "/index.html"
        self.serve_static(path)

    def do_POST(self) -> None:
        try:
            if self.path == "/api/new":
                self.handle_new()
            elif self.path == "/api/play":
                self.handle_play()
            else:
                self.send_json({"error": "Unknown endpoint"}, HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def handle_new(self) -> None:
        data = self.read_json()
        seed_value = data.get("seed")
        seed = int(seed_value) if seed_value not in (None, "") else None
        checkpoint_value = str(data.get("checkpoint", "")).strip()
        checkpoint_path = Path(checkpoint_value) if checkpoint_value else DEFAULT_CHECKPOINT
        if not checkpoint_path.is_absolute():
            checkpoint_path = PROJECT_ROOT / checkpoint_path
        p2_policy = str(data.get("p2_policy", "greedy"))
        partner_policy = str(data.get("partner_policy", "heuristic"))
        p4_policy = str(data.get("p4_policy", "greedy"))
        greedy_bots = bool(data.get("greedy_bots", False))
        learn_after_game = bool(data.get("learn_after_game", True))
        learning_rate = float(data.get("learning_rate", 0.005))
        if learning_rate <= 0:
            raise ValueError("Learning rate must be positive")
        session_id, session = new_session(
            seed=seed,
            p2_policy=p2_policy,
            partner_policy=partner_policy,
            p4_policy=p4_policy,
            checkpoint_path=checkpoint_path,
            greedy_bots=greedy_bots,
            learn_after_game=learn_after_game,
            learning_rate=learning_rate,
        )
        self.send_json(state_payload(session_id, session))

    def handle_play(self) -> None:
        data = self.read_json()
        session_id = str(data["session_id"])
        card_id = str(data["card_id"])
        session = SESSIONS[session_id]
        env = session.env

        if env.done:
            raise ValueError("Game is already finished")
        if env.current_player != session.human_id:
            raise ValueError("It is not the human player's turn")

        card = card_from_id(card_id)
        if card not in env.legal_actions(session.human_id):
            raise ValueError("Card is not legal in the current hand")

        play_card(session, card)
        advance_to_human(session)
        self.send_json(state_payload(session_id, session))

    def serve_static(self, url_path: str) -> None:
        requested = (WEB_ROOT / url_path.lstrip("/")).resolve()
        if not requested.is_file() or WEB_ROOT.resolve() not in requested.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        content_type = mimetypes.guess_type(str(requested))[0] or "application/octet-stream"
        body = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def send_json(
        self,
        payload: dict[str, Any],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local Briscola web UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    server = ThreadingHTTPServer((args.host, args.port), BriscolaRequestHandler)
    print(f"Briscola interface running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
