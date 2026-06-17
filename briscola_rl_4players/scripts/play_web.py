#!/usr/bin/env python3
"""Local interactive web interface for four-player Briscola.

Run with:

    python3 scripts/play_web.py

Then open http://127.0.0.1:8000 in a browser. The human plays as player 0,
while partner and opponents are controlled by baseline policies.
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
from briscola.policies import Policy
from briscola.rules import partner_of, team_of


@dataclass
class WebSession:
    """In-memory state for one browser game."""

    env: FourPlayerBriscolaEnv
    human_id: int
    partner_policy: Policy
    opponent_policy: Policy
    rng: random.Random
    events: list[str] = field(default_factory=list)
    last_completed_trick: list[PlayedCard] = field(default_factory=list)
    last_trick_winner: int | None = None
    last_trick_points: int = 0


SESSIONS: dict[str, WebSession] = {}


def create_policy(name: str) -> Policy:
    if name == "random":
        return RandomPolicy()
    if name == "greedy":
        return GreedyTrickPolicy()
    if name == "heuristic":
        return SimpleHeuristicPolicy()
    raise ValueError(f"Unknown policy: {name}")


def new_session(seed: int | None, partner: str, opponents: str) -> tuple[str, WebSession]:
    if seed is None:
        seed = random.randrange(1_000_000_000)
    human_id = 0
    env = FourPlayerBriscolaEnv(seed=seed, start_player=seed % 4)
    session = WebSession(
        env=env,
        human_id=human_id,
        partner_policy=create_policy(partner),
        opponent_policy=create_policy(opponents),
        rng=random.Random(seed + 40_000),
    )
    session.events.append(f"New game started with seed {seed}.")
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
        policy = (
            session.partner_policy
            if player_id == partner_of(session.human_id)
            else session.opponent_policy
        )
        action = policy.select_action(obs, session.rng, greedy=False)
        play_card(session, action)


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


def final_message(env: FourPlayerBriscolaEnv) -> str:
    if env.scores[0] > env.scores[1]:
        return f"Team A wins {env.scores[0]}-{env.scores[1]}."
    if env.scores[1] > env.scores[0]:
        return f"Team B wins {env.scores[1]}-{env.scores[0]}."
    return "The game ends in a 60-60 draw."


def state_payload(session_id: str, session: WebSession) -> dict[str, Any]:
    env = session.env
    human_obs = env.observe(session.human_id) if not env.done else None
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
        "players": player_payloads(session.human_id),
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
        "partner_policy": getattr(session.partner_policy, "name", "partner"),
        "opponent_policy": getattr(session.opponent_policy, "name", "opponent"),
    }


def player_payloads(human_id: int) -> list[dict[str, Any]]:
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
    return f"{rank_label(card.rank)} of {suit_label(card.suit)}"


def rank_label(rank: str) -> str:
    labels = {
        "ace": "Ace",
        "three": "Three",
        "king": "King",
        "knight": "Knight",
        "jack": "Jack",
        "seven": "Seven",
        "six": "Six",
        "five": "Five",
        "four": "Four",
        "two": "Two",
    }
    return labels[rank]


def suit_label(suit: str) -> str:
    labels = {
        "cups": "Cups",
        "coins": "Coins",
        "clubs": "Clubs",
        "swords": "Swords",
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
        partner = str(data.get("partner", "heuristic"))
        opponents = str(data.get("opponents", "greedy"))
        session_id, session = new_session(seed=seed, partner=partner, opponents=opponents)
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
