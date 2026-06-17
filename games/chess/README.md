# Neural Chess

A chess engine where the position evaluation is a small **neural network** I trained,
plugged into a classic **alpha-beta search** for actually choosing moves. I built it to
understand how engines like AlphaZero work at a basic level — "learn to score positions,
then search" — and to have something I could test against real chess algorithms.

> _Why I made this: [add a sentence in your own words — e.g. "I'd used chess engines for
> years and wanted to know what's actually inside one" / "I wanted a hands-on way to learn
> PyTorch"]._ It's a learning project, not a Stockfish rival — see [Honesty about scope](#honesty-about-scope).

Stack: **PyTorch** for the network, [python-chess](https://python-chess.readthedocs.io/)
for the rules and move generation (I didn't want to re-implement castling and en passant).

---

## The core idea

I split the problem into two halves that I could reason about separately:

- **The network** answers one narrow question: *"how good is this position for the side to
  move?"* — one number from `-1` (losing) to `+1` (winning). It does **not** pick moves.
- **The search** ([`chess_ai.py`](chess_ai.py)) does the move-picking with **alpha-beta
  (negamax)**: look a few moves ahead, assume both sides play their best, and choose the
  move that leads to the best score the network can promise. Captures are tried first so
  the pruning actually kicks in, and mate/stalemate are scored directly so the net never
  has to learn that checkmate is good.

| File | What it does |
|---|---|
| [`encode.py`](encode.py) | turns a board into `17×8×8` number planes the net can read |
| [`model.py`](model.py) | the network itself (a small residual CNN) + saving/loading |
| [`chess_ai.py`](chess_ai.py) | the alpha-beta search that uses the net to score leaves |
| [`heuristics.py`](heuristics.py) | a classic material + piece-square score (no ML) |
| [`opponents.py`](opponents.py) | the bots I play against: random / greedy / minimax / Stockfish |
| [`selfplay.py`](selfplay.py) | plays games and records `(position, who won)` for training |
| [`train.py`](train.py) | the actual training |
| [`arena.py`](arena.py) | runs a match vs an opponent, prints W/D/L + an Elo estimate |
| [`play.py`](play.py) | watch a game in the terminal |

## How I trained it

The thing that surprised me most: you can't just throw a random network into self-play. With
random weights every evaluation is noise, so "self-play" is two random players flailing —
there's nothing to learn from. I got stuck here for a while. The fix was two stages:

1. **Bootstrap (supervised).** First I train the net to *imitate* a classic
   material + piece-square heuristic ([`heuristics.py`](heuristics.py)) over thousands of
   random positions. This is plain supervised learning — predict a known number — and after
   it the net already plays sensible, if shallow, chess. This is what gives self-play
   something to build on.

2. **Self-play (reinforcement-ish).** Now the net plays games (against itself with some
   random opening moves for variety, and against the baseline bots), and I train it to
   predict *the eventual result of the game* from each position it saw. Positions that led
   to wins get nudged toward `+1`, losses toward `-1`. It's outcome regression / TD(1) —
   the simplest version of "learn from your own games."

```bash
pip install -r requirements.txt

python3 train.py --bootstrap-only            # fast: a playable net in ~1 min
python3 train.py                             # bootstrap + a few self-play rounds
python3 train.py --iterations 10 --games 60  # longer = stronger
python3 train.py --resume                    # keep training the saved net
```

The trained net is saved to `model.pt`; `arena.py` and `play.py` load it automatically.

## Playing it against other algorithms

This was the fun part — actually measuring whether it learned anything.

```bash
python3 arena.py --opponent random --games 40
python3 arena.py --opponent material         # greedy: grabs the best capture
python3 arena.py --opponent minimax --depth 3   # classic search, no neural net
python3 arena.py --opponent stockfish        # if you have Stockfish installed
```

Where my current net lands (depth-2 search, bootstrapped):

| Opponent | Result |
|---|---|
| random | never loses (≈71%, lots of draws it doesn't convert) |
| greedy material | roughly even / slightly ahead (≈54%) |

Colors alternate each game and it reports an estimated Elo gap.

## Watch it play

```bash
python3 play.py --opponent minimax
python3 play.py --self-play --delay 0.3
```

Renders the board with Unicode pieces, an eval bar (the net's read of the position), and
live search stats.

## Things I'd still fix / learned the hard way

- **Side-to-move perspective.** Every position is encoded from the mover's point of view
  (the board gets mirrored on Black's turn) so one value head works for both colors. Getting
  the mirroring consistent between encoding, search, and the eval bar was the buggiest part.
- **It draws won games.** At depth 1 it doesn't see far enough to convert winning endgames,
  so a lot of games time out as draws. More search depth fixes most of it.
- **Speed.** Evaluating one position at a time through the net during search is slow on CPU.
  Batching the leaves (or caching) is the obvious next step.

## Honesty about scope

This is a **learning project**. The goal was a correct engine that demonstrably *learns* —
beats a random player, holds its own against a greedy one — not something that competes with
real engines. Getting to a strong rating would need a lot more self-play and probably MCTS
with a policy head (the actual AlphaZero recipe), which I'd like to try next.
