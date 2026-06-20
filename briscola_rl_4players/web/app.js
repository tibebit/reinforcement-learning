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
  nextButton: document.getElementById("nextButton"),
  collectButton: document.getElementById("collectButton"),
};

elements.newGameButton.addEventListener("click", () => startNewGame());
elements.nextButton.addEventListener("click", () => nextStep());
elements.collectButton.addEventListener("click", () => collectTrick());

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

async function nextStep() {
  if (!state.sessionId) {
    return;
  }
  const data = await postJson("/api/next", { session_id: state.sessionId });
  render(data);
}

async function collectTrick() {
  if (!state.sessionId) {
    return;
  }
  const data = await postJson("/api/collect", { session_id: state.sessionId });
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
  elements.nextButton.disabled = !data.can_next;
  elements.collectButton.disabled = !data.can_collect;

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
    elements.trickArea.appendChild(
      renderCard(played.card, false, `P${played.player_id + 1}`)
    );
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
    elements.lastTrickArea.appendChild(
      renderMiniCard(played.card, null, `P${played.player_id + 1}`)
    );
  }
}

function renderHand(data) {
  elements.handArea.innerHTML = "";
  if (data.pending_collect) {
    elements.turnHint.textContent = "Press Collect Trick";
  } else if (data.human_turn) {
    elements.turnHint.textContent = "Choose a card";
  } else {
    elements.turnHint.textContent = "Press Next";
  }
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

function renderCard(card, asButton, ownerLabel = null) {
  const node = document.createElement(asButton ? "button" : "div");
  node.className = `card suit-${card.suit}`;
  node.type = asButton ? "button" : undefined;
  node.setAttribute("aria-label", card.label);

  if (ownerLabel) {
    const owner = document.createElement("div");
    owner.className = "owner-badge owner-badge-card";
    owner.textContent = ownerLabel;
    node.appendChild(owner);
  }

  const rank = document.createElement("div");
  rank.className = "rank";
  rank.textContent = `${shortRank(card.rank)} ${suitEmoji(card.suit)}`;

  const points = document.createElement("div");
  points.className = "points";
  points.textContent = `${card.points} pts`;

  node.append(rank, points);
  return node;
}

function renderMiniCard(card, id, ownerLabel = null) {
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

  if (ownerLabel) {
    const owner = document.createElement("div");
    owner.className = "owner-badge owner-badge-card";
    owner.textContent = ownerLabel;
    node.appendChild(owner);
  }

  const rank = document.createElement("div");
  rank.className = "rank";
  rank.textContent = `${shortRank(card.rank)} ${suitEmoji(card.suit)}`;

  const points = document.createElement("div");
  points.className = "points";
  points.textContent = `${card.points} pts`;

  node.append(rank, points);
  return node;
}

function labelSuit(suit) {
  return {
    cups: "🏆 Cups",
    coins: "🪙 Coins",
    clubs: "🌳 Clubs",
    swords: "⚔️ Swords",
  }[suit] || "-";
}

function suitEmoji(suit) {
  return {
    cups: "🏆",
    coins: "🪙",
    clubs: "🌳",
    swords: "⚔️",
  }[suit] || "";
}

function shortRank(rank) {
  return {
    ace: "A",
    three: "3",
    king: "K",
    knight: "Q",
    jack: "J",
    seven: "7",
    six: "6",
    five: "5",
    four: "4",
    two: "2",
  }[rank] || "?";
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
