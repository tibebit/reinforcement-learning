# Learning Cooperative Team Strategy in Four-Player Briscola with Reinforcement Learning and Self-Play

This project studies a Reinforcement Learning agent for **four-player Briscola**.
The game is played by four players divided into two teams:

\[
Team A = \{P_1, P_3\}
\]

\[
Team B = \{P_2, P_4\}
\]

Players in the same team share the final objective, but each player only sees
their own hand and public information. The goal is to learn a policy that can
play legal and meaningful games, cooperate implicitly with a partner, use public
memory, and improve through self-play against a population of frozen historical
policies.

The project does **not** aim to solve Briscola mathematically or compute a Nash
equilibrium. The objective is to build and evaluate a robust policy that performs
better than random, greedy, and simple heuristic baselines.

## Quick Setup

Recommended Python version:

```bash
python 3.11+
```

The project has been tested with Python 3.12. Minimal runtime and test
dependencies are:

```text
numpy
pytest
pyyaml
```

Optional notebook dependencies are:

```text
jupyter
matplotlib
pandas
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Install minimal dependencies:

```bash
pip install numpy pytest pyyaml
```

Install optional notebook dependencies:

```bash
pip install jupyter matplotlib pandas
```

Run all tests from this folder:

```bash
PYTHONPATH=. pytest
```

Expected result:

```text
16 passed
```

Run a training job:

```bash
PYTHONPATH=. python scripts/train.py --config configs/train_pool_selfplay.yaml
```

Other training configs:

```text
configs/train_random.yaml
configs/train_latest_only.yaml
configs/train_pool_selfplay.yaml
```

Run evaluation:

```bash
PYTHONPATH=. python scripts/evaluate.py --config configs/eval.yaml
```

Start the local playable web interface:

```bash
PYTHONPATH=. python scripts/play_web.py
```

Then open:

```text
http://127.0.0.1:8000
```

Generated files such as checkpoints, logs, JSON results, plots, caches, and
virtual environments should not be committed.

## Project Structure

```text
briscola/
  Core environment, cards, rules, observations, policies, features, REINFORCE.

configs/
  YAML configs for training and evaluation.

scripts/
  CLI entry points for training, evaluation, and web play.

tests/
  Unit tests for rules, environment, observations, policies, and training.

notebooks/
  Analysis notebook.

web/
  Static frontend for the playable demo.

experiments/
  Output folders for logs, checkpoints, plots, and evaluation artifacts.
```

## Core Idea

All players use the same **policy architecture**, but not the same live
parameters during training.

```text
architecture: same for everyone
learner parameters: current theta
partner parameters: frozen theta_partner^- sampled from the pool
opponent parameters: frozen theta_opponent^- sampled from the pool
```

This means the model always chooses actions with the same kind of function, but
the learner, partner, and opponents can have different weights.

The main training setup is:

```text
one active learner: current policy theta
partner: frozen historical snapshot sampled from the pool
opponents: frozen historical snapshot(s) sampled from the pool
only the learner's parameters are updated
```

This prevents the agent from learning only how to cooperate with an identical
copy of itself or exploit one fixed opponent style.

## Game Variant

The project uses standard four-player Briscola with drawing.

Rules:

- 40-card Italian deck.
- 4 suits.
- 4 players.
- 2 teams: players 1 and 3 versus players 2 and 4.
- Each player receives 3 cards at the beginning.
- One card is revealed and determines the trump suit, called `briscola`.
- The revealed trump card remains part of the draw deck and is drawn last.
- No obligation to follow suit.
- Each trick contains 4 cards.
- The strongest card wins the trick.
- The winner of the trick leads the next trick.
- After each trick, if the deck is not empty, players draw one card each.
- Drawing starts from the trick winner and continues in table order.
- Each player plays exactly 10 cards.
- The match ends after 10 tricks.
- Total points are 120.
- The team with more than 60 points wins.
- A 60-60 result is a draw.

Card values:

| Card | Points |
|---|---:|
| Ace | 11 |
| Three | 10 |
| King | 4 |
| Knight | 3 |
| Jack | 2 |
| Seven | 0 |
| Six | 0 |
| Five | 0 |
| Four | 0 |
| Two | 0 |

Trick strength order, from strongest to weakest:

\[
Ace > Three > King > Knight > Jack > Seven > Six > Five > Four > Two
\]

Trick resolution:

1. If at least one trump card is played, the strongest trump wins.
2. If no trump card is played, the strongest card of the leading suit wins.
3. Cards from other suits cannot win unless they are trump cards.

## Mathematical Formulation

Four-player Briscola is not a standard single-agent MDP. It is a:

\[
\text{partially observable multi-agent stochastic game}
\]

The game can be written as:

\[
\mathcal{G} =
\left(
\mathcal{I},
\mathcal{S},
\{\mathcal{A}_i\},
P,
\{\mathcal{O}_i\},
Z,
\{R_i\},
\gamma
\right)
\]

where:

- \(\mathcal{I}=\{1,2,3,4\}\) is the set of players.
- \(\mathcal{S}\) is the full internal game state.
- \(\mathcal{A}_i\) is the action space of player \(i\).
- \(P\) is the transition function.
- \(\mathcal{O}_i\) is the observation space of player \(i\).
- \(Z\) maps states to player observations.
- \(R_i\) is the reward of player \(i\).
- \(\gamma=1\), because each match is finite and episodic.

The teams are:

\[
A = \{P_1, P_3\}
\]

\[
B = \{P_2, P_4\}
\]

Team rewards satisfy:

\[
R_1 = R_3
\]

\[
R_2 = R_4
\]

\[
R_A = -R_B
\]

The game is cooperative inside each team and competitive between teams.

## Internal State

The full state is known only by the environment:

\[
S_t =
\left(
H_t^1,H_t^2,H_t^3,H_t^4,
D_t,
\tau,
B_t,
C_t,
P_t^A,
P_t^B,
\ell_t,
i_t,
U_t
\right)
\]

where:

- \(H_t^i\) is the hand of player \(i\).
- \(D_t\) is the remaining ordered draw deck.
- \(\tau\) is the trump suit.
- \(B_t\) is the revealed trump card, while still visible.
- \(C_t\) is the current trick.
- \(P_t^A\) and \(P_t^B\) are the team scores.
- \(\ell_t\) is the current trick leader.
- \(i_t\) is the player who must act.
- \(U_t\) is the public history of played cards.

The full state is Markovian, but it is not fully observable by any player.

## Player Observation

Each player receives a legal observation:

- own hand;
- partner hand only during the final phase, after the draw deck is empty;
- trump suit;
- revealed trump card, until it is drawn;
- current trick;
- cards already played;
- who played each public card;
- who is the partner;
- who are the opponents;
- team score;
- opponent team score;
- current trick leader;
- current position in the trick;
- number of cards left in the draw deck;
- game phase.

The player does **not** observe:

- partner's hand before the final phase;
- opponents' hands;
- order of the draw deck;
- future draws;
- hidden random seed;
- internal parameters used by other policies.

The observation is therefore not equal to the real state:

\[
O_t^i \neq S_t
\]

This is why the game must be treated as partially observable.

## Information State Approximation

The theoretically ideal solution would maintain a belief state:

\[
b_t(s) = P(S_t=s \mid \text{public history and private hand})
\]

This belief would represent probabilities over:

- partner's possible cards;
- opponents' possible cards;
- remaining deck order;
- future draws.

The project does not implement an exact Bayesian belief state. Instead, it uses
a compressed public-memory representation:

\[
M_t =
\left(
H_t,
\tau,
B_t,
C_t,
U_t,
P_t^{team},
P_t^{opp},
N_t,
\ell_t,
pos_t,
\phi_t
\right)
\]

where:

- \(H_t\) is the current player's hand.
- \(\tau\) is the trump suit.
- \(B_t\) is the revealed trump card, if still visible.
- \(C_t\) is the current trick.
- \(U_t\) is the public played-card history.
- \(P_t^{team}\) is the current player's team score.
- \(P_t^{opp}\) is the opposing team score.
- \(N_t\) is the number of cards left in the deck.
- \(\ell_t\) is the trick leader.
- \(pos_t\) is whether the player acts first, second, third, or fourth.
- \(\phi_t\) is a game-phase indicator.

This memory is legal because it is built only from information available to the
player.

## Policy Architecture

The project uses a shared **linear softmax policy with action masking**.

For each legal card \(a\), compute a feature vector:

\[
\phi(M_t,a) \in \mathbb{R}^d
\]

The preference score of action \(a\) is:

\[
h_\theta(M_t,a)=\theta^\top \phi(M_t,a)
\]

The policy is a softmax over legal cards only:

\[
\pi_\theta(a|M_t)=
\frac{\exp(h_\theta(M_t,a))}
{\sum_{a'\in H_t}\exp(h_\theta(M_t,a'))}
\]

Illegal cards receive probability zero:

\[
\pi_\theta(a|M_t)=0
\quad \text{if } a \notin H_t
\]

The same architecture is used by all players. The observation is always
canonical:

```text
self
partner
left_opponent
right_opponent
team_score
opponent_score
current_trick
played_cards
deck_count
game_phase
```

This allows the same policy function to be used in all seats.

## Parameters

The learned parameter vector is:

\[
\theta \in \mathbb{R}^d
\]

Each component of \(\theta\) is a learned weight associated with one feature.

Example:

```text
theta_trump
theta_card_points
theta_can_win_trick
theta_partner_winning
theta_opponent_winning
theta_final_phase
```

Large positive weights increase the probability of choosing cards with that
feature. Negative weights reduce it.

During training:

```text
learner: theta, updated
partner: theta_partner^-, frozen
opponents: theta_opponent^-, frozen
```

Only the learner's \(\theta\) is updated.

## Feature Set

Features are computed for every candidate action \(a\).

Basic card features:

- card point value;
- normalized point value;
- card rank;
- normalized rank strength;
- is trump;
- is Ace;
- is Three;
- is King, Knight, or Jack;
- is zero-point card;
- is high card;
- is low card.

Trick-state features:

- position in trick: first, second, third, fourth;
- number of cards already in the trick;
- current trick points;
- current winning card;
- current winning player is partner;
- current winning player is opponent;
- candidate card would currently win the trick;
- candidate card would beat partner's winning card;
- candidate card would beat opponent's winning card;
- number of players still to act;
- number of opponents still to act;
- partner still to act;
- opponent still to act.

Memory features:

- number of cards already played;
- number of trump cards already played;
- number of trump cards not yet observed;
- number of Aces already played;
- number of Threes already played;
- number of high cards already played;
- stronger cards of the same suit not yet observed;
- stronger trump cards not yet observed;
- card already known to be safe, if applicable.

Score and phase features:

- team score;
- opponent score;
- score difference;
- team is ahead;
- team is behind;
- early phase;
- middle phase;
- late phase;
- deck is empty;
- final tricks.

Interaction features:

- `trump * trick_points`;
- `trump * opponent_winning`;
- `trump * partner_winning`;
- `high_card * final_phase`;
- `partner_winning * card_points`;
- `opponent_winning * can_win_trick`;
- `can_win_trick * trick_points`;
- `score_advantage * final_phase`;
- `score_deficit * final_phase`;
- `players_after_me * risky_card`;
- `opponents_after_me * high_card`;
- `partner_still_to_act * card_points`;
- `deck_empty * high_card`;
- `deck_empty * trump`.

Feature normalization is recommended. For example:

```text
card_points_normalized = card_points / 11
trick_points_normalized = trick_points / 22
team_score_normalized = team_score / 120
score_difference_normalized = (team_score - opponent_score) / 120
deck_count_normalized = deck_count / 28
```

## Reward Definition

The main reward is a team reward.

Let:

\[
\Delta_T = P_T^{team} - P_T^{opp}
\]

### Terminal Win Reward

\[
R_T^{win} =
\begin{cases}
+1, & P_T^{team} > 60 \\
0, & P_T^{team} = 60 \\
-1, & P_T^{team} < 60
\end{cases}
\]

This directly optimizes match outcome but gives a sparse signal.

### Dense Trick-Point Reward

For each completed trick:

\[
R_{t+1}^{points} =
\begin{cases}
+v(C_t), & \text{if learner team wins the trick} \\
-v(C_t), & \text{otherwise}
\end{cases}
\]

where \(v(C_t)\) is the point value of the four-card trick.

With \(\gamma=1\), the sum of trick rewards equals the final point
difference:

\[
\sum_t R_{t+1}^{points} = P_T^{team} - P_T^{opp}
\]

### Recommended Combined Reward

The recommended first reward is:

\[
R_T^{combined}
=
\operatorname{sign}(P_T^{team}-P_T^{opp})
+
\lambda
\frac{P_T^{team}-P_T^{opp}}{120}
\]

with:

\[
\lambda = 0.2
\]

This makes winning the primary objective while preserving information about
the final score margin.

Default setting:

```text
reward_mode = combined_terminal
lambda_margin = 0.2
gamma = 1.0
```

Reward ablations should compare:

```text
terminal_win
trick_point_dense
combined_terminal
```

## Learning Algorithm

The main learning algorithm is **REINFORCE with baseline**.

For each learner decision:

\[
\theta
\leftarrow
\theta
+
\alpha
(G_t-b_t)
\nabla_\theta
\log \pi_\theta(a_t|M_t)
\]

where:

- \(\alpha\) is the learning rate.
- \(G_t\) is the reward-to-go.
- \(b_t\) is a baseline independent of the action.
- \(\pi_\theta(a_t|M_t)\) is the probability assigned to the chosen card.

For a linear softmax policy:

\[
\nabla_\theta \log \pi_\theta(a_t|M_t)
=
\phi(M_t,a_t)
-
\sum_{a'\in H_t}
\pi_\theta(a'|M_t)\phi(M_t,a')
\]

This gives a simple explicit update:

\[
\theta
\leftarrow
\theta
+
\alpha
(G_t-b_t)
\left[
\phi(M_t,a_t)
-
\sum_{a'\in H_t}
\pi_\theta(a'|M_t)\phi(M_t,a')
\right]
\]

## Multi-Agent Training Objective

The main robust training setup updates one active learner per episode or batch.

At the beginning of an episode:

```text
sample learner seat uniformly
assign current theta to the learner
sample partner theta_partner^- from snapshot pool
sample opponent theta_opponent^- from snapshot pool
freeze partner and opponents
play a full match
update only theta using learner actions
```

The gradient estimate is:

\[
\nabla J(\theta)
\approx
\sum_t
(G_t-b_t)
\nabla_\theta
\log \pi_\theta(a_t^{learner}|M_t^{learner})
\]

This is more robust than always pairing the learner with an identical copy of
itself, because the learner must adapt to different partner behaviors.

An optional team-learning variant updates both players of the learner team:

\[
\nabla J(\theta)
\approx
\sum_{i \in TeamLearner}
\sum_t
(G_t-b_t)
\nabla_\theta
\log \pi_\theta(a_t^i|M_t^i)
\]

This variant is useful as an ablation, but the default project uses a
pool-sampled partner.

## Baseline For Variance Reduction

REINFORCE can have high variance. A baseline reduces variance without changing
the expected policy gradient, as long as it does not depend on the selected
action.

Recommended baseline:

\[
b = \frac{1}{B}\sum_{j=1}^{B}G^{(j)}
\]

where \(B\) is the number of episodes in the batch.

Other possible baselines:

- moving average of recent returns;
- leave-one-out batch baseline;
- separate baseline by game phase;
- learned value baseline as an extension.

The minimal implementation should use the batch mean return.

## Self-Play With Historical Pool

The project uses population-based self-play.

The pool at training cycle \(k\) is:

\[
\mathcal{P}_k =
\{
\pi_{\theta^{(0)}},
\pi_{\theta^{(1)}},
\ldots,
\pi_{\theta^{(m)}}
\}
\]

Each pool member is a frozen snapshot of the same policy architecture.

Default sampling:

\[
\theta^- \sim Uniform(\mathcal{P}_k)
\]

During training:

1. Sample the active learner seat.
2. Use current parameters \(\theta_k\) for the learner.
3. Sample partner snapshot \(\theta_p^-\) from the pool.
4. Sample opponent snapshot \(\theta_o^-\) from the pool.
5. Freeze all non-learner policies.
6. Generate a batch of complete matches.
7. Update only \(\theta_k\).
8. Every \(K\) updates, save \(\theta_k\) into the pool.

Example seating:

```text
P1: learner theta
P2: opponent theta_o^-
P3: partner theta_p^-
P4: opponent theta_o^-
```

Another possible episode:

```text
P1: opponent theta_o^-
P2: learner theta
P3: opponent theta_o^-
P4: partner theta_p^-
```

The learner seat is sampled uniformly so the agent does not specialize in one
fixed table position.

## Pool Management

The pool should contain:

- initial random policy;
- early snapshots;
- intermediate snapshots;
- recent snapshots;
- best evaluated snapshot.

Recommended pool parameters:

```text
max_pool_size = 20
snapshot_interval = 50 updates
pool_best_count = 4
pool_recent_count = 10
pool_eval_games = optional
sampling = uniform
```

If the pool exceeds the maximum size, keep:

- the initial snapshot;
- the best evaluated snapshots, when evaluation scores are available;
- a few recent snapshots;
- a time-spaced subset of older snapshots.

This avoids both overfitting to the latest policy and uncontrolled pool growth.
If `pool_eval_games` is zero, no external score is assigned and the pool still
keeps diversity through recent and time-spaced historical snapshots.

## Hyperparameters

Recommended initial values:

| Parameter | Value |
|---|---:|
| policy type | linear softmax |
| learning rate \(\alpha\) | 0.01 |
| backup learning rate | 0.001 |
| discount \(\gamma\) | 1.0 |
| batch size | 500 games |
| training updates | 2,000 to 10,000 |
| snapshot interval | 50 updates |
| max pool size | 20 |
| reward margin weight \(\lambda\) | 0.2 |
| baseline | batch mean return |
| exploration | stochastic softmax |
| evaluation policy | greedy and stochastic |
| learner seat sampling | uniform |
| partner sampling | uniform from pool |
| opponent sampling | uniform from pool |
| random seeds | fixed train/eval split |

If training is unstable:

- reduce learning rate from `0.01` to `0.001`;
- increase batch size;
- normalize returns;
- clip very large updates;
- check feature scaling;
- verify that hidden information is not leaked.

## Algorithms To Implement

### 1. Game Engine

The engine must implement:

- deck creation;
- deterministic shuffling from seed;
- initial deal of 3 cards to each player;
- trump reveal;
- partner-hand visibility during the final three tricks;
- legal action generation;
- four-card trick resolution;
- team scoring;
- drawing after each trick;
- final card draw, including the revealed trump card;
- terminal condition;
- replay logging.

Required invariants:

```text
no duplicate cards
each player plays exactly 10 cards
total final points = 120
exactly 10 tricks are played
all 40 cards are used
legal actions are always cards in hand
same seed gives same game
```

### 2. Observation Builder

The observation builder must expose only legal information:

```text
observe(player_id) -> PlayerObservation
```

It must not expose:

```text
partner hand
opponent hands
deck order
future draws
policy parameters of other players
```

### 3. Feature Extractor

The feature extractor computes:

```text
phi(observation, candidate_card)
```

for every card in the player's hand.

### 4. Linear Softmax Policy

The policy must support:

- action probabilities;
- stochastic sampling;
- greedy selection;
- action masking;
- log-probability computation;
- gradient computation.

### 5. REINFORCE Trainer

The trainer must:

- collect trajectories;
- store observations, actions, rewards, and log probabilities;
- compute reward-to-go;
- subtract baseline;
- update only learner parameters;
- log training statistics.

### 6. Snapshot Pool

The pool must:

- store frozen parameter vectors;
- sample partner snapshots;
- sample opponent snapshots;
- keep a maximum number of snapshots;
- preserve important checkpoints.

### 7. Evaluation Suite

The evaluation suite must:

- run matches without updates;
- use fixed test seeds;
- evaluate both stochastic and greedy policies;
- compare against baselines;
- compute confidence intervals;
- build matchup matrices.

## Baseline Policies

### Random Policy

Chooses uniformly among legal cards:

\[
\pi_{random}(a|H)=\frac{1}{|H|}
\]

### Greedy Trick Policy

Chooses the action that maximizes immediate trick result.

Example behavior:

- if it can win a trick with many points, it tries to win;
- if it cannot win, it throws a low-value card;
- it ignores future consequences.

### Simple Heuristic Policy

A simple documented heuristic can be used for evaluation.

Example:

- if partner is winning, play a low card unless it is safe to load points;
- if an opponent is winning and the trick has many points, try to win;
- avoid spending high trump on low-value tricks;
- in the final phase, prefer winning valuable tricks.

The heuristic should not be used for the main training unless explicitly
declared as curriculum.

### Historical Snapshot Policies

Historical snapshots are used both for training and evaluation.

The final model should be tested against:

- initial policy;
- early snapshots;
- intermediate snapshots;
- recent snapshots;
- best previous snapshot.

## Experiments

### Experiment 1: Latest-Only Versus Historical Pool

Compare:

```text
self-play against latest snapshot only
self-play against uniform historical pool
```

Hypothesis:

```text
The historical pool produces a policy that is more robust and less specialized.
```

Metrics:

- win rate;
- draw rate;
- loss rate;
- average point difference;
- worst-case result against test opponents;
- stability across seeds.

### Experiment 2: Partner Sampling Ablation

Compare:

```text
partner = current theta
partner = frozen snapshot from pool
partner = random/heuristic policy
```

Hypothesis:

```text
Pool-sampled partners improve generalization to non-identical partners.
```

### Experiment 3: Memory Ablation

Compare:

```text
no played-card memory
public played-card memory
public memory plus aggregate remaining-card features
```

Hypothesis:

```text
Memory becomes more useful in late-game situations and after the deck is empty.
```

### Experiment 4: Reward Ablation

Compare:

```text
terminal_win
trick_point_dense
combined_terminal
```

Hypothesis:

```text
The combined reward provides a better compromise between outcome relevance and learning stability.
```

## Evaluation Protocol

Official evaluations must use:

- frozen policies;
- no parameter updates;
- seeds not used for training;
- both team assignments;
- multiple learner seats;
- repeated paired deals;
- confidence intervals.

For paired deals:

1. Run policy A as Team A and policy B as Team B with seed \(s\).
2. Run policy B as Team A and policy A as Team B with the same seed \(s\).
3. Average the two results.

Main metrics:

- win rate;
- draw rate;
- loss rate;
- mean point difference;
- standard error;
- confidence interval;
- worst-case score against test pool;
- performance against random;
- performance against greedy;
- performance against heuristic.

The matchup matrix is:

\[
W_{ij}
=
\text{average result of policy } i
\text{ against policy } j
\]

This helps detect:

- real improvement;
- cyclic strategies;
- overfitting;
- regressions;
- brittle policies.

## Diagnostics

Diagnostics describe the learned behavior. They are not used as reward.

Track:

- trump usage rate;
- points won when using trump;
- points lost after playing high-value cards;
- frequency of loading points when partner is winning;
- frequency of beating partner unnecessarily;
- frequency of cutting with trump;
- behavior before and after deck exhaustion;
- performance by trick number;
- performance when ahead or behind;
- action probabilities in selected example states;
- largest learned feature weights.

## Theory Connected To The Project

### MDP Theory

The full internal state is Markovian. This connects to:

- states;
- actions;
- transitions;
- reward;
- policy;
- value functions;
- Bellman equations.

However, players do not observe the full state, so an MDP is not sufficient for
the agent's decision problem.

### POMDP Theory

Each player observes only a partial view of the game. Hidden information
includes the partner hand before the final phase, opponent hands, and deck
order. When the deck is empty and the last three tricks begin, the partner hand
becomes legal information for the player.

The project therefore uses the POMDP idea of decision-making from observations
and memory.

### Bayesian POMDP

The exact theoretical solution would maintain a belief distribution over hidden
states. The project does not compute this exactly, but uses a compressed
information state based on public history and private hand.

### Monte Carlo Reinforcement Learning

Episodes are complete matches. Returns can be computed after observing the
final result.

### Policy Gradient

The policy is optimized directly through REINFORCE:

\[
\nabla J(\theta)
=
\mathbb{E}
\left[
G_t
\nabla_\theta
\log \pi_\theta(a_t|M_t)
\right]
\]

### Stochastic Approximation

Each batch gives a noisy estimate of the true gradient. The update process is a
stochastic approximation procedure controlled by learning rate, batch size, and
variance reduction.

### TD Learning And Eligibility Traces

TD methods are not part of the core implementation. They can be used as an
extension to learn a value baseline, but the minimal project remains
REINFORCE with a simple baseline.

### Multi-Agent Reinforcement Learning

The project is multi-agent because:

- four policies act in the same environment;
- teammates share reward;
- opponents have opposite reward;
- each player acts from a different observation;
- non-learner policies are frozen during updates.

## Proposed Repository Structure

```text
briscola_rl_4players/
  README.md
  briscola/
    cards.py
    rules.py
    env.py
    observation.py
    features.py
    policies.py
    baselines.py
    reinforce.py
    self_play.py
    evaluation.py
    replay.py
  tests/
    test_cards.py
    test_rules.py
    test_env.py
    test_observation.py
    test_features.py
    test_policy.py
    test_training.py
  configs/
    train_random.yaml
    train_pool_selfplay.yaml
    train_latest_only.yaml
    eval.yaml
  experiments/
    results/
    plots/
  notebooks/
    analysis.ipynb
```

## Running Training And Evaluation

Run tests first:

```bash
cd /Users/anna/Documents/RL/briscola_rl_4players
python3 -m unittest discover -s tests
```

Run a small training job:

```bash
python3 scripts/train.py \
  --updates 100 \
  --batch-size 100 \
  --learning-rate 0.01 \
  --snapshot-interval 10 \
  --max-pool-size 20 \
  --pool-best-count 4 \
  --pool-recent-count 10 \
  --pool-eval-games 20 \
  --output experiments/results/checkpoint.json \
  --log experiments/results/train_log.jsonl
```

Run a quick evaluation against random:

```bash
python3 scripts/evaluate.py \
  --checkpoint experiments/results/checkpoint.json \
  --opponent random \
  --games 200
```

Evaluate against the greedy or heuristic baselines:

```bash
python3 scripts/evaluate.py --checkpoint experiments/results/checkpoint.json --opponent greedy --games 200
python3 scripts/evaluate.py --checkpoint experiments/results/checkpoint.json --opponent heuristic --games 200
```

Build a small matchup matrix:

```bash
python3 scripts/evaluate.py \
  --checkpoint experiments/results/checkpoint.json \
  --matrix \
  --games 100 \
  --max-snapshots 5 \
  --output experiments/results/matchup_matrix.json
```

The training script saves:

- final learner parameters;
- frozen historical snapshots in the pool;
- feature names;
- training configuration;
- training logs with return, score difference, gradient norm, and pool size.

## Running The Interactive Interface

Start the local web interface:

```bash
cd /Users/anna/Documents/RL/briscola_rl_4players
python3 scripts/play_web.py
```

Then open:

```text
http://127.0.0.1:8000
```

The human plays as `P1` on Team A. `P3` is the partner, while `P2` and `P4`
are opponents. The interface supports random, greedy, heuristic, and trained
learner-checkpoint bot policies.

The UI lets you choose:

- policy for `P2`, the first opponent;
- policy for `P3`, the partner;
- policy for `P4`, the second opponent;
- learner checkpoint path, by default `experiments/results/checkpoint.json`;
- stochastic or greedy bot action selection.
- whether learner bots should be updated after each completed human game;
- the small web-interface learning rate used for that online update.

To play with the learner, set `P3 Partner` to `Learner checkpoint`. To play
against the learner, set `P2 Opponent`, `P4 Opponent`, or both to
`Learner checkpoint`.

After each human move, the server automatically advances all bot turns until
the next human decision or the end of the match. The table highlights the
current player and shows each bot's policy under its seat label.

If `Learn after game` is enabled and at least one bot is using `Learner
checkpoint`, the interface applies a small REINFORCE update to the learner at
the end of the match. The update is kept in server memory, so pressing `New Game
with Updated Learner` starts another match with the updated parameters. This
interactive update is meant for demonstration and inspection; official
evaluation should still use frozen checkpoints with no online updates.

## Development Phases

### Phase 0: Rule Specification

Freeze the exact rules:

- card representation;
- rank order;
- point values;
- trump reveal;
- drawing order;
- trick resolution;
- terminal condition;
- tie handling at 60-60.

### Phase 1: Game Engine

Implement and test the environment.

Required tests:

- no duplicate cards;
- all cards are used;
- each player plays 10 cards;
- final score sums to 120;
- trick winner is correct;
- draw order is correct;
- seed reproducibility.

### Phase 2: State And Observation Separation

Implement:

```text
internal state
legal player observation
```

Add tests against information leakage.

### Phase 3: Baselines

Implement:

- random;
- greedy trick;
- simple heuristic.

Produce the first baseline matchup matrix.

### Phase 4: Feature-Based Policy

Implement:

- feature extractor;
- linear preference function;
- softmax action distribution;
- action mask;
- stochastic and greedy action selection.

### Phase 5: REINFORCE Against Frozen Baselines

Train current policy against frozen random or heuristic opponents.

Check:

- average return improves;
- trained model beats random on unseen seeds;
- gradients are finite and stable.

### Phase 6: Self-Play With Snapshot Pool

Implement:

- snapshot saving;
- pool sampling;
- partner from pool;
- opponents from pool;
- learner seat randomization;
- update only learner parameters.

### Phase 7: Experiments

Run:

- latest-only versus historical pool;
- current partner versus pool-sampled partner;
- memory versus no-memory;
- reward ablations;
- multiple seeds.

### Phase 8: Interpretation

Analyze:

- feature weights;
- trump usage;
- partner cooperation behavior;
- late-game decisions;
- example hands;
- matchup matrix.

### Phase 9: Human Interface

Optional final extension:

- human versus agent;
- human plus agent team;
- replay saving;
- show action probabilities after each game.

## Success Criteria

Minimum success:

- complete legal four-player games;
- no hidden information leakage;
- each player plays exactly 10 cards;
- final score always sums to 120;
- REINFORCE improves against random;
- final policy beats random on unseen seeds;
- evaluation includes confidence intervals.

Full success:

- historical pool beats latest-only;
- pool-sampled partner improves generalization;
- public memory beats no-memory;
- policy competes with greedy or heuristic baselines;
- learned behavior shows plausible team cooperation;
- matchup matrix does not show severe regressions.

Valid negative result:

- pool does not improve robustness;
- linear policy does not beat heuristic;
- partner sampling makes learning too noisy;
- memory provides limited advantage;
- REINFORCE variance is too high.

In that case, the project remains valid if the failure is analyzed through:

- partial observability;
- limited feature representation;
- credit assignment;
- stochastic approximation noise;
- non-stationarity of self-play;
- difficulty of implicit cooperation.

## What The Project Does Not Claim

The project does not claim:

- to solve Briscola;
- to find the optimal policy;
- to reach Nash equilibrium;
- to compute an exact belief state;
- to model human psychology;
- to guarantee victory against every opponent;
- to prove that self-play always improves strategy.

## Final Project Claim

A suitable final claim is:

> We developed a four-player Briscola reinforcement learning agent using a
> shared linear softmax policy architecture, public-memory features, team
> rewards, REINFORCE updates, and population-based self-play. During training,
> the learner used current parameters, while the partner and opponents were
> sampled as frozen historical snapshots from a policy pool. The system was
> evaluated against random, greedy, heuristic, and historical policies to
> measure robustness, memory usage, and cooperative behavior under partial
> observability.
