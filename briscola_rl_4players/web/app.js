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
  turnBanner: document.getElementById("turnBanner"),
  revealedTrump: document.getElementById("revealedTrump"),
  deckCount: document.getElementById("deckCount"),
  trickArea: document.getElementById("trickArea"),
  lastTrickArea: document.getElementById("lastTrickArea"),
  handArea: document.getElementById("handArea"),
  partnerHandArea: document.getElementById("partnerHandArea"),
  eventLog: document.getElementById("eventLog"),
  turnHint: document.getElementById("turnHint"),
  partnerHint: document.getElementById("partnerHint"),
  p2Policy: document.getElementById("p2Policy"),
  partnerPolicy: document.getElementById("partnerPolicy"),
  p4Policy: document.getElementById("p4Policy"),
  checkpointInput: document.getElementById("checkpointInput"),
  botMode: document.getElementById("botMode"),
  learnAfterGame: document.getElementById("learnAfterGame"),
  learningRateInput: document.getElementById("learningRateInput"),
  seedInput: document.getElementById("seedInput"),
  newGameButton: document.getElementById("newGameButton"),
};

elements.newGameButton.addEventListener("click", () => startNewGame());

async function startNewGame() {
  elements.newGameButton.disabled = true;
  const payload = {
    p2_policy: elements.p2Policy.value,
    partner_policy: elements.partnerPolicy.value,
    p4_policy: elements.p4Policy.value,
    checkpoint: elements.checkpointInput.value,
    greedy_bots: elements.botMode.value === "greedy",
    learn_after_game: elements.learnAfterGame.checked,
    learning_rate: Number(elements.learningRateInput.value || 0.005),
    seed: elements.seedInput.value,
  };

  try {
    const data = await postJson("/api/new", payload);
    state.sessionId = data.session_id;
    render(data);
  } catch (error) {
    elements.statusText.textContent = error.message;
  } finally {
    elements.newGameButton.disabled = false;
  }
}

async function playCard(cardId) {
  if (!state.sessionId) {
    return;
  }

  disableHand(true);
  try {
    const data = await postJson("/api/play", {
      session_id: state.sessionId,
      card_id: cardId,
    });
    render(data);
  } catch (error) {
    elements.statusText.textContent = error.message;
    disableHand(false);
  }
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
  elements.turnBanner.textContent = turnBannerText(data);
  elements.turnBanner.classList.toggle("done", data.done);
  elements.turnBanner.classList.toggle("waiting", !data.done && !data.human_turn);
  elements.revealedTrump.replaceWith(renderMiniCard(data.revealed_trump, "revealedTrump"));
  elements.revealedTrump = document.getElementById("revealedTrump");

  renderSeats(data);
  renderCurrentTrick(data.current_trick);
  renderLastTrick(data);
  renderHand(data);
  renderPartnerHand(data);
  renderEvents(data.events);
  elements.newGameButton.textContent =
    data.done && data.learner_updated
      ? "New Game with Updated Learner"
      : "New Game";
}

function renderSeats(data) {
  for (const player of data.players) {
    const seat = document.getElementById(`seat${player.id}`);
    seat.classList.toggle("active-turn", data.current_player === player.id);
    seat.querySelector("span").textContent = player.label;
    seat.querySelector("strong").textContent = player.role;
    const policy = seat.querySelector("small");
    if (policy) {
      policy.textContent = shortPolicyName(player.policy);
      policy.title = player.policy;
    }
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
    cups: "🏆",
    coins: "🪙",
    clubs: "♣",
    swords: "⚔",
  }[suit] || "-";
}

function turnBannerText(data) {
  if (data.done) {
    return data.message;
  }
  if (data.human_turn) {
    return "Your turn: choose a card";
  }
  const player = data.players.find((entry) => entry.id === data.current_player);
  if (!player) {
    return data.message;
  }
  return `${player.label} ${player.role} is playing (${shortPolicyName(player.policy)})`;
}

function shortPolicyName(name) {
  if (!name) {
    return "";
  }
  if (name.startsWith("learner:")) {
    return name;
  }
  return name.replace("_", " ");
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
