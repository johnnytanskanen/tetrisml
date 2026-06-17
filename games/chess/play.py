#!/usr/bin/env python3
"""Watch the neural net play a full game in the terminal.

Renders the board with Unicode pieces, an evaluation bar (the net's read of the
position, from the mover's side), the last move, and live search stats.

    python3 play.py --opponent minimax
    python3 play.py --opponent material --delay 0.4 --depth 2
    python3 play.py --self-play                       # the net against itself
"""
import argparse
import random
import sys
import time

import chess

from game import play_game
from chess_ai import ChessAI
from opponents import make_opponent
import model as model_mod

RESET, DIM, BOLD = '\033[0m', '\033[2m', '\033[1m'
GREEN, RED, CYAN, YELLOW = '\033[32m', '\033[31m', '\033[36m', '\033[33m'


def eval_bar(value, width=24):
    """value in ~[-1, 1] from White's perspective → a centered bar."""
    value = max(-1.0, min(1.0, value))
    half = width // 2
    filled = int(round(abs(value) * half))
    if value >= 0:
        bar = ' ' * (half - filled) + GREEN + '█' * filled + RESET + '│' + ' ' * half
    else:
        bar = ' ' * half + '│' + RED + '█' * filled + RESET + ' ' * (half - filled)
    return f"[{bar}] {value:+.2f} (white)"


def render(board, last_move, stats, white_name, black_name, ply):
    sys.stdout.write('\033[H\033[2J')  # home + clear
    print(f"{BOLD}NEURAL CHESS{RESET}   {CYAN}{white_name}{RESET} (white)  "
          f"vs  {YELLOW}{black_name}{RESET} (black)\n")

    # board.unicode() draws from White's view; pieces as figurines
    grid = board.unicode(borders=True, empty_square='·')
    print(grid)

    # eval is reported from the side-to-move that just moved; convert to White view
    ev = stats.get('eval', 0.0)
    mover_was_white = (board.turn == chess.BLACK)  # side to move flipped after push
    white_eval = ev if mover_was_white else -ev
    print('\n' + eval_bar(white_eval))

    mv = last_move.uci() if last_move else '—'
    print(f"{DIM}ply {ply:>3}   last {mv:<6}   "
          f"nodes {stats.get('nodes', 0):>6}   depth {stats.get('depth', '-')}"
          f"   moves {stats.get('n_moves', '-')}{RESET}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--opponent', default='minimax',
                    help='random | material | minimax | minimax3 | stockfish')
    ap.add_argument('--self-play', action='store_true', help='net vs itself')
    ap.add_argument('--bot-white', action='store_true', help='force the net to play white')
    ap.add_argument('--depth', type=int, default=2)
    ap.add_argument('--delay', type=float, default=0.5, help='seconds between moves')
    ap.add_argument('--max-plies', type=int, default=200)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--model', default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    device = model_mod.pick_device(args.device)
    model, meta, trained = model_mod.load_or_new(
        args.model or model_mod.DEFAULT_PATH, device)
    if not trained:
        print("! no model.pt — using an UNTRAINED net (run train.py first)")
        time.sleep(1.0)

    bot = ChessAI(model=model, device=device, depth=args.depth)

    if args.self_play:
        white, black, wn, bn = bot, bot, 'neural', 'neural'
    else:
        opp = make_opponent(args.opponent, rng=rng)
        bot_white = args.bot_white or bool(rng.getrandbits(1))
        if bot_white:
            white, black, wn, bn = bot, opp, 'neural', opp.name
        else:
            white, black, wn, bn = opp, bot, opp.name, 'neural'

    state = {'ply': 0}

    def on_move(board, move, stats):
        state['ply'] += 1
        render(board, move, stats, wn, bn, state['ply'])
        time.sleep(args.delay)

    board, result, _ = play_game(white, black, max_plies=args.max_plies,
                                 on_move=on_move)

    outcome = board.outcome(claim_draw=True)
    if outcome and outcome.winner is not None:
        winner = wn if outcome.winner == chess.WHITE else bn
        print(f"\n{BOLD}{GREEN}{winner} wins{RESET} "
              f"({outcome.termination.name.lower()})")
    elif result > 0:
        print(f"\n{BOLD}{wn} ahead on material at the cap{RESET}")
    elif result < 0:
        print(f"\n{BOLD}{bn} ahead on material at the cap{RESET}")
    else:
        print(f"\n{BOLD}draw{RESET}")

    for p in (white, black):
        if hasattr(p, 'close'):
            p.close()


if __name__ == '__main__':
    main()
