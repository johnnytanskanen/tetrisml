# Tetris AI — a self-learning agent

A Tetris-playing agent that runs in your terminal and **learns how to play from scratch**.
It doesn't use any hard-coded strategy: a genetic algorithm discovers — through thousands
of self-played games — *what makes a board good*, and the agent then searches for the move
that leads there.

![Tetris AI in the terminal](screenshots/terminal.png)

Pure Python, zero dependencies (just the standard library).

---

## Two layers of intelligence

It's worth being precise about what's "AI" and what's "machine learning" here, because they're
different things working together:

**1. The player — search-based AI.**
For every piece, the agent enumerates all rotations and columns for the current piece **and**
the next one (a 2-ply lookahead), simulates each landing, and scores the resulting board with a
weighted sum of features. It plays the move with the best score. ~2,700 boards per move.

**2. The learner — machine learning (`train.py`).**
Those feature weights are *not* hand-written. A **genetic algorithm** learns them by self-play:
a population of random weight vectors plays headless games, the ones that clear the most lines
survive and breed, and the population improves generation after generation. No gradients, no
training labels, no human strategy — just selection pressure. This is the part that actually
*learns*.

---

## How the learning works

```
genome  =  one weight per board feature   (holes, bumpiness, height, …)
fitness =  lines cleared in self-play before topping out
```

Each generation:

1. every genome plays several headless games on identical piece sequences (fair contest),
2. the top genomes are kept (elitism) and used as parents,
3. children are bred by **tournament selection + uniform crossover + annealed Gaussian mutation**,
4. weight vectors are renormalised (the scale is arbitrary; only direction matters).

A real run — watch the population **mean** climb as bad strategies die off:

```
gen  0   best 137.8   mean  92.5   *saved
gen  9   best 139.0   mean 118.5   *saved
gen 24   best 138.8   mean 137.5

benchmark (1000 pieces, unseen seeds):
  hand-tuned baseline   397.8 lines
  learned from scratch  398.0 lines
```

The learned agent matches an expert hand-tuned baseline (~398 lines, **zero top-outs over
3,000 pieces**) — having rediscovered competent play on its own, and landing on a genuinely
different strategy (it leans heavily on column transitions, and even found a *positive*
bumpiness weight). It ties rather than beats the baseline because both are good enough to
essentially never lose within the benchmark; longer games with a higher piece cap give the
search more room to separate strong vectors.

### Feature set the agent learns to weigh

| Feature | What it measures |
|---|---|
| Lines cleared | rows completed (the reward) |
| Holes | empty cells trapped under blocks |
| Aggregate / max height | how tall the stack is |
| Bumpiness | how jagged the surface is |
| Wells | deep single-column gaps |
| Row / column transitions | filled↔empty flips (fragmentation) |
| Landing height | how high the piece came to rest |

---

## It learns while you watch

Launching the game starts a trainer in the background. The agent **hot-reloads** the improving
weights every few seconds, so you can watch it get smarter in real time — the header shows
`learning · gen N`. It uses half your CPU cores to stay responsive, stops when you quit, and is
disabled with `TETRIS_NO_TRAIN=1`.

The terminal UI also visualises the agent's reasoning live: the weighted contribution of each
feature to the chosen move, the spread of candidate-placement scores, board-health trends
(holes / height / bumpiness over time), the line-clear rate, and a column-height profile.

---

## Run it

```bash
python3 tetris_ai.py        # watch the agent play (needs an ~80×30 terminal)
```

Train explicitly (writes `weights.json`, which the game loads automatically):

```bash
python3 train.py                                   # ~1–2 min on 8 cores
python3 train.py --generations 40 --population 60 --pieces 500   # a deeper run
python3 train.py --resume                          # keep improving the current weights
```

## Controls

| Key | Action |
|---|---|
| `+` / `-` | speed the agent up / down |
| `Space` | reset speed |
| `p` | pause / resume |
| `r` | restart |
| `q` | quit |

## Files

```
tetris_ai.py    game engine + the search-based player + the curses UI
train.py        genetic algorithm — learns the weights via self-play  ← the ML
autotrain.py    runs training in the background while you play
weights.json    the learned weights (committed; regenerate with train.py)
```

## Requirements

Python 3.7+. No third-party packages.

## License

MIT
