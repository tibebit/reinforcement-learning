const state = {
  sessionId: null,
  data: null,
};

const elements = {
  statusText: document.getElementById("statusText"),
  scoreA: document.getElementById("scoreA"),
  scoreB: document.getElementById("scoreB"),
  trickIndex: document.getElementById("trickIndex"),
  trumpSuit: document.getElementById("trumpSuit"),
  revealedTrump: document.getElementById("revealedTrump"),
  deckCount: document.getElementById("deckCount"),
  trickArea: document.getElementById("trickArea"),
  lastTrickArea: document.getElementById("lastTrickArea"),
  handArea: document.getElementById("handArea"),
  partnerHandArea: document.getElementById("partnerHandArea"),
  eventLog: document.getElementById("eventLog"),
  turnHint: document.getElementById("turnHint"),
  partnerHint: document.getElementById("partnerHint"),
  partnerPolicy: document.getElementById("partnerPolicy"),
  opponentPolicy: document.getElementById("opponentPolicy"),
  seedInput: document.getElementById("seedInput"),
  newGameButton: document.getElementById("newGameButton"),
};

elements.newGameButton.addEventListener("click", () => startNewGame());

async function startNewGame() {
  elements.newGameButton.disabled = true;
  const payload = {
    partner: elements.partnerPolicy.value,
    opponents: elements.opponentPolicy.value,
    seed: elements.seedInput.value,
  };

  const data = await postJson("/api/new", payload);
  state.sessionId = data.session_id;
  render(data);
  elements.newGameButton.disabled = false;
}

async function playCard(cardId) {
  if (!state.sessionId) {
    return;
  }

  disableHand(true);
  const data = await postJson("/api/play", {
    session_id: state.sessionId,
    card_id: cardId,
  });
  render(data);
}

async function postJson(url, payload) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

function render(data) {
  state.data = data;

  elements.statusText.textContent = data.message;
  elements.scoreA.textContent = data.scores.team_a;
  elements.scoreB.textContent = data.scores.team_b;
  elements.trickIndex.textContent = `${Math.min(data.trick_index + 1, 10)}/10`;
  elements.trumpSuit.textContent = labelSuit(data.trump_suit);
  elements.deckCount.textContent = data.deck_count;
  elements.revealedTrump.replaceWith(renderMiniCard(data.revealed_trump, "revealedTrump"));
  elements.revealedTrump = document.getElementById("revealedTrump");

  renderSeats(data);
  renderCurrentTrick(data.current_trick);
  renderLastTrick(data);
  renderHand(data);
  renderPartnerHand(data);
  renderEvents(data.events);
}

function renderSeats(data) {
  for (const player of data.players) {
    const seat = document.getElementById(`seat${player.id}`);
    seat.classList.toggle("active-turn", data.current_player === player.id);
    seat.querySelector("span").textContent = player.label;
    seat.querySelector("strong").textContent = player.role;
  }
}

function renderCurrentTrick(cards) {
  elements.trickArea.innerHTML = "";
  if (!cards.length) {
    const empty = document.createElement("div");
    empty.className = "empty-card mini-card";
    empty.textContent = "Waiting";
    elements.trickArea.appendChild(empty);
    return;
  }

  for (const played of cards) {
    const wrapper = document.createElement("div");
    wrapper.className = "played-card";
    const owner = document.createElement("div");
    owner.className = "owner";
    owner.textContent = `P${played.player_id + 1}`;
    wrapper.appendChild(owner);
    wrapper.appendChild(renderCard(played.card, false));
    elements.trickArea.appendChild(wrapper);
  }
}

function renderLastTrick(data) {
  elements.lastTrickArea.innerHTML = "";
  if (!data.last_completed_trick.length) {
    const empty = document.createElement("div");
    empty.className = "empty-card mini-card";
    empty.textContent = "None";
    elements.lastTrickArea.appendChild(empty);
    return;
  }

  for (const played of data.last_completed_trick) {
    const wrapper = document.createElement("div");
    wrapper.className = "played-card";
    const owner = document.createElement("div");
    owner.className = "owner";
    owner.style.color = "#66716d";
    owner.textContent = `P${played.player_id + 1}`;
    wrapper.appendChild(owner);
    wrapper.appendChild(renderMiniCard(played.card));
    elements.lastTrickArea.appendChild(wrapper);
  }
}

function renderHand(data) {
  elements.handArea.innerHTML = "";
  elements.turnHint.textContent = data.human_turn ? "Choose a card" : "";
  const legalIds = new Set(data.legal_action_ids);

  for (const card of data.human_hand) {
    const button = renderCard(card, true);
    button.disabled = !data.human_turn || !legalIds.has(card.id) || data.done;
    button.addEventListener("click", () => playCard(card.id));
    elements.handArea.appendChild(button);
  }

  if (!data.human_hand.length) {
    const empty = document.createElement("div");
    empty.className = "empty-card mini-card";
    empty.textContent = "Empty";
    elements.handArea.appendChild(empty);
  }
}

function renderPartnerHand(data) {
  elements.partnerHandArea.innerHTML = "";
  if (!data.partner_hand_visible) {
    elements.partnerHint.textContent = "Visible after the draw deck is empty";
    const hidden = document.createElement("div");
    hidden.className = "empty-card mini-card";
    hidden.textContent = "Hidden";
    elements.partnerHandArea.appendChild(hidden);
    return;
  }

  elements.partnerHint.textContent = "Final three tricks";
  if (!data.partner_hand.length) {
    const empty = document.createElement("div");
    empty.className = "empty-card mini-card";
    empty.textContent = "Empty";
    elements.partnerHandArea.appendChild(empty);
    return;
  }

  for (const card of data.partner_hand) {
    elements.partnerHandArea.appendChild(renderCard(card, false));
  }
}

function renderEvents(events) {
  elements.eventLog.innerHTML = "";
  for (const event of events.slice().reverse()) {
    const item = document.createElement("li");
    item.textContent = event;
    elements.eventLog.appendChild(item);
  }
}

function renderCard(card, asButton) {
  const node = document.createElement(asButton ? "button" : "div");
  node.className = `card suit-${card.suit}`;
  node.type = asButton ? "button" : undefined;
  node.setAttribute("aria-label", card.label);

  const rank = document.createElement("div");
  rank.className = "rank";
  rank.textContent = card.rank_label;

  const suit = document.createElement("div");
  suit.className = "suit";
  suit.textContent = card.suit_label;

  const points = document.createElement("div");
  points.className = "points";
  points.textContent = `${card.points} pts`;

  node.append(rank, suit, points);
  return node;
}

function renderMiniCard(card, id) {
  if (!card) {
    const empty = document.createElement("div");
    empty.className = "mini-card empty-card";
    empty.id = id || "";
    empty.textContent = "Hidden";
    return empty;
  }

  const node = document.createElement("div");
  node.className = `mini-card suit-${card.suit}`;
  node.id = id || "";

  const rank = document.createElement("div");
  rank.className = "rank";
  rank.textContent = card.rank_label;

  const suit = document.createElement("div");
  suit.className = "suit";
  suit.textContent = card.suit_label;

  const points = document.createElement("div");
  points.className = "points";
  points.textContent = `${card.points} pts`;

  node.append(rank, suit, points);
  return node;
}

function labelSuit(suit) {
  return {
    cups: "Cups",
    coins: "Coins",
    clubs: "Clubs",
    swords: "Swords",
  }[suit] || "-";
}

function disableHand(disabled) {
  for (const button of elements.handArea.querySelectorAll("button")) {
    button.disabled = disabled;
  }
}

startNewGame().catch((error) => {
  elements.statusText.textContent = error.message;
  elements.newGameButton.disabled = false;
});
