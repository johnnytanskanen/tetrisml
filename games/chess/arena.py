#!/usr/bin/env python3
"""Play the neural net against an existing algorithm and report the score.

This is the headline deliverable: the learned bot vs. a real opponent (random,
greedy material, classic minimax, or Stockfish). Colors alternate each game for
fairness; we print wins / draws / losses and a rough Elo difference.

    python3 arena.py --opponent minimax --games 40
    python3 arena.py --opponent material --depth 3
    python3 arena.py --opponent stockfish --games 20      # needs Stockfish on PATH
"""
import argparse
import math
import random

from game import play_game
from chess_ai import ChessAI
from opponents import make_opponent
import model as model_mod


def elo_delta(score, n):
    """Elo difference implied by a match score fraction in (0,1)."""
    if score <= 0:
        return -800.0
    if score >= 1:
        return 800.0
    return -400.0 * math.log10(1.0 / score - 1.0)


def play_match(bot, opponent, games=40, max_plies=200, verbose=True):
    """Alternate colors. Returns (wins, draws, losses) from the bot's perspective."""
    import chess
    w = d = l = 0
    for g in range(games):
        bot_white = (g % 2 == 0)
        white, black = (bot, opponent) if bot_white else (opponent, bot)
        _board, res, _ = play_game(white, black, max_plies=max_plies)
        bot_res = res if bot_white else -res          # to bot's perspective
        if bot_res > 0:
            w += 1; tag = 'W'
        elif bot_res < 0:
            l += 1; tag = 'L'
        else:
            d += 1; tag = 'D'
        if verbose:
            color = 'white' if bot_white else 'black'
            print(f"  game {g + 1:>3}/{games}  bot={color:<5}  {tag}   "
                  f"(W{w} D{d} L{l})")
    return w, d, l


def report(name, w, d, l):
    n = w + d + l
    score = (w + 0.5 * d) / n if n else 0.0
    print(f"\nNeural bot vs {name}:  {w}W / {d}D / {l}L  "
          f"over {n} games")
    print(f"  score {score * 100:.1f}%   est. Elo diff {elo_delta(score, n):+.0f}")
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--opponent', default='minimax',
                    help='random | material | minimax | minimax3 | stockfish')
    ap.add_argument('--games', type=int, default=40)
    ap.add_argument('--depth', type=int, default=2, help='neural bot search depth')
    ap.add_argument('--max-plies', type=int, default=200)
    ap.add_argument('--model', default=None, help='path to model.pt (default: ./model.pt)')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--quiet', action='store_true')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = model_mod.pick_device(args.device)
    model, meta, trained = model_mod.load_or_new(
        args.model or model_mod.DEFAULT_PATH, device)
    if not trained:
        print("! no model.pt found — using an UNTRAINED net (run train.py first)\n")
    elif meta:
        print(f"loaded model.pt  ({meta})\n")

    bot = ChessAI(model=model, device=device, depth=args.depth)
    opponent = make_opponent(args.opponent, rng=rng)

    print(f"Arena: neural(depth {args.depth}) vs {opponent.name}, "
          f"{args.games} games\n")
    w, d, l = play_match(bot, opponent, args.games, args.max_plies,
                         verbose=not args.quiet)
    report(opponent.name, w, d, l)

    if hasattr(opponent, 'close'):
        opponent.close()


if __name__ == '__main__':
    main()
