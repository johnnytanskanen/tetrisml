# ML Games

Game-playing AIs I've built to teach myself machine learning. Each one uses the same idea —
*learn how to score a position, then search for the best move* — but with a different
learning method, so I could see how the approaches compare on problems I actually understand.

> Started as just a Tetris bot; I'm turning it into a small collection. (Planning to rename
> the repo to **`mlgames`**.)

## What's here

| Game | How it picks moves | How it learns | Stack |
|---|---|---|---|
| [**Chess**](games/chess) | a neural network scores positions; alpha-beta search picks moves | trains on its own games (after a supervised bootstrap) | PyTorch + python-chess |
| [**Tetris**](games/tetris) | a weighted score over board features; 2-ply search picks moves | a genetic algorithm evolves the weights via self-play | pure Python |

The two sit at opposite ends of the ML spectrum on purpose: Tetris uses **no gradients** —
just evolution over a handful of numbers — while Chess uses a **gradient-trained neural
net**. Building both is what made the trade-offs click for me.

## Quick start

```bash
# Chess — neural net vs classic chess algorithms
cd games/chess
pip install -r requirements.txt
python3 train.py --bootstrap-only      # get a playable net fast
python3 arena.py --opponent minimax    # see how it does against a classic engine
python3 play.py  --opponent material   # watch a game

# Tetris — genetic-algorithm-tuned search (no dependencies)
cd games/tetris
python3 terminal/tetris_ai.py          # watch it play, with live analytics
python3 terminal/train.py              # evolve the weights yourself
```

Each game has its own README with the details and what I learned building it:
[chess](games/chess/README.md) · [tetris](games/tetris/README.md).

## Layout

```
games/
  chess/      # PyTorch value net + alpha-beta search; trains by self-play
  tetris/     # genetic-algorithm-tuned heuristic search; curses terminal UI
```

## License

MIT
