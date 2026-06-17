#!/usr/bin/env python3
"""Generate training data by playing games.

Each game produces a list of (position, z) pairs, where z is the final result seen
from the side to move at that position (+1 won, -1 lost, 0 drew). Regressing the
value net onto z is the "learn from your own games" signal (outcome regression /
TD(1)) used by train.py.

To keep the data diverse we mix:
  * neural self-play with a temperature on the opening moves (exploration), and
  * games against the baseline opponents (random / material / minimax).
"""
import random

import numpy as np

from game import play_game, labelled_positions
from chess_ai import ChessAI
from opponents import RandomOpponent, MaterialOpponent, MinimaxOpponent


class TemperaturePlayer:
    """Wrap a ChessAI so its first `opening_plies` moves are sampled (exploration)."""

    def __init__(self, ai, temperature=0.6, opening_plies=12, depth=1, rng=None):
        self.ai = ai
        self.temperature = temperature
        self.opening_plies = opening_plies
        self.depth = depth
        self.rng = rng or random.Random()

    def select_move(self, board):
        temp = self.temperature if board.ply() < self.opening_plies else 0.0
        return self.ai.select_move(board, depth=self.depth, temperature=temp, rng=self.rng)


def generate_dataset(ai, games=40, depth=1, max_plies=160, rng=None, verbose=False):
    """Play `games` games with the current net and return (X, y) numpy arrays.

    Roughly half are neural self-play; the rest are vs. baselines so the net also
    learns from positions a pure self-play loop might never visit.
    """
    rng = rng or random.Random()
    samples = []
    baselines = [RandomOpponent(rng), MaterialOpponent(rng), MinimaxOpponent(2, rng)]

    for g in range(games):
        explorer = TemperaturePlayer(ai, temperature=0.7, depth=depth, rng=rng)
        if g % 2 == 0:
            white, black, tag = explorer, explorer, 'self-play'
        else:
            opp = baselines[(g // 2) % len(baselines)]
            # alternate which color the net takes
            if (g // 2) % 2 == 0:
                white, black, tag = explorer, opp, f'vs {opp.name}'
            else:
                white, black, tag = opp, explorer, f'vs {opp.name}'

        _board, result, history = play_game(white, black, max_plies=max_plies,
                                             record=True)
        pairs = labelled_positions(history, result)
        samples.extend(pairs)
        if verbose:
            print(f"  game {g + 1:>3}/{games}  {tag:<14} "
                  f"plies={len(history):>3}  result={result:+.0f}  "
                  f"(total samples {len(samples)})")

    if not samples:
        return np.empty((0, 17, 8, 8), np.float32), np.empty((0,), np.float32)
    X = np.stack([p for p, _ in samples]).astype(np.float32)
    y = np.array([z for _, z in samples], dtype=np.float32)
    return X, y
