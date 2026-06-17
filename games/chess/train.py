#!/usr/bin/env python3
"""Train the chess value network — this is the "machine learning" of the project.

Two phases:

1. **Bootstrap** (supervised). Before any self-play, regress the net onto the
   classic material+piece-square heuristic (heuristics.py) over random positions.
   This gives the net sensible opening values so it starts out playing real chess
   instead of random moves — which in turn makes self-play data meaningful.

2. **Self-play refinement** (reinforcement). Repeatedly: the current net plays
   games (selfplay.py), and the net is trained to predict the eventual game outcome
   from each position (outcome regression / TD(1)). After each round we checkpoint
   model.pt and play a short arena vs a fixed baseline to show progress.

    python3 train.py                         # bootstrap + a few self-play rounds
    python3 train.py --iterations 10 --games 60 --depth 1
    python3 train.py --resume                # continue from model.pt
    python3 train.py --bootstrap-only        # just phase 1 (fast, gets a playable net)
"""
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import chess

from encode import board_to_planes
from heuristics import normalized_value
from model import ValueNet, save, load_or_new, pick_device, DEFAULT_PATH
from chess_ai import ChessAI
from selfplay import generate_dataset
from arena import play_match, elo_delta
from opponents import make_opponent


# ---- random positions for the bootstrap phase ----------------------------------

def random_positions(n, rng):
    """Sample positions along random games (varied depth) and label by the heuristic."""
    X, y = [], []
    while len(X) < n:
        board = chess.Board()
        steps = rng.randint(0, 40)
        for _ in range(steps):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over():
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        X.append(board_to_planes(board))
        y.append(normalized_value(board))
    return (np.stack(X).astype(np.float32),
            np.array(y, dtype=np.float32))


# ---- generic supervised fit ----------------------------------------------------

def fit(model, device, X, y, epochs=4, batch=256, lr=1e-3, log_prefix=''):
    if len(X) == 0:
        return float('nan')
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)
    n = len(X)
    last = float('nan')
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = Xt[idx].to(device)
            yb = yt[idx].to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        last = total / n
        if log_prefix:
            print(f"  {log_prefix} epoch {ep + 1}/{epochs}  mse {last:.4f}")
    model.eval()
    return last


# ---- phases --------------------------------------------------------------------

def bootstrap(model, device, n_positions, epochs, rng):
    print(f"[bootstrap] regressing net onto material+PST heuristic "
          f"({n_positions} positions)")
    X, y = random_positions(n_positions, rng)
    mse = fit(model, device, X, y, epochs=epochs, lr=1e-3, log_prefix='bootstrap')
    print(f"[bootstrap] done, final mse {mse:.4f}\n")
    return mse


def selfplay_round(model, device, ai, games, depth, epochs, rng, verbose):
    X, y = generate_dataset(ai, games=games, depth=depth, rng=rng, verbose=verbose)
    print(f"  collected {len(X)} positions from {games} games")
    mse = fit(model, device, X, y, epochs=epochs, lr=5e-4, log_prefix='train')
    return mse, len(X)


def quick_gate(model, device, depth, opponent_kind, games, rng):
    """Short arena vs a fixed baseline → score fraction, to track progress."""
    bot = ChessAI(model=model, device=device, depth=depth)
    opp = make_opponent(opponent_kind, rng=rng)
    w, d, l = play_match(bot, opp, games=games, verbose=False)
    n = w + d + l
    score = (w + 0.5 * d) / n if n else 0.0
    if hasattr(opp, 'close'):
        opp.close()
    return score, (w, d, l)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--iterations', type=int, default=4, help='self-play rounds')
    ap.add_argument('--games', type=int, default=40, help='games per self-play round')
    ap.add_argument('--depth', type=int, default=1, help='search depth during self-play')
    ap.add_argument('--epochs', type=int, default=3, help='train epochs per round')
    ap.add_argument('--bootstrap-positions', type=int, default=6000)
    ap.add_argument('--bootstrap-epochs', type=int, default=5)
    ap.add_argument('--bootstrap-only', action='store_true')
    ap.add_argument('--gate-opponent', default='minimax', help='baseline for the progress arena')
    ap.add_argument('--gate-games', type=int, default=20)
    ap.add_argument('--resume', action='store_true', help='continue from model.pt')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=DEFAULT_PATH)
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    print(f"device: {device}\n")

    if args.resume:
        model, meta, trained = load_or_new(args.out, device)
        print("resumed from model.pt" if trained else "no model.pt — starting fresh")
        games_trained = meta.get('games_trained', 0)
    else:
        model = ValueNet().to(device)
        games_trained = 0

    start = time.time()

    # phase 1: bootstrap (skip on resume — already playable)
    if not args.resume:
        boot_mse = bootstrap(model, device, args.bootstrap_positions,
                             args.bootstrap_epochs, rng)
        save(model, args.out, meta={'method': 'bootstrap (heuristic regression)',
                                    'games_trained': 0, 'bootstrap_mse': round(boot_mse, 4)})
        print(f"saved {args.out} (bootstrapped)\n")
    else:
        boot_mse = float('nan')

    if args.bootstrap_only:
        print(f"bootstrap-only done in {time.time() - start:.0f}s")
        return

    # baseline gate before refinement
    base_score, base_wdl = quick_gate(model, device, max(args.depth, 2),
                                      args.gate_opponent, args.gate_games, rng)
    print(f"gate vs {args.gate_opponent}: {base_score * 100:.0f}% "
          f"{base_wdl}  (pre-self-play)\n")

    # phase 2: self-play refinement
    ai = ChessAI(model=model, device=device, depth=args.depth)
    best_score = base_score
    for it in range(args.iterations):
        print(f"=== self-play round {it + 1}/{args.iterations} ===")
        t0 = time.time()
        mse, n = selfplay_round(model, device, ai, args.games, args.depth,
                                args.epochs, rng, args.verbose)
        games_trained += args.games
        score, wdl = quick_gate(model, device, max(args.depth, 2),
                                args.gate_opponent, args.gate_games, rng)
        marker = ''
        if score >= best_score:
            best_score = score
            marker = ' *best'
        save(model, args.out, meta={
            'method': 'self-play outcome regression (TD1) + heuristic bootstrap',
            'games_trained': games_trained,
            'train_mse': round(mse, 4),
            f'gate_vs_{args.gate_opponent}': round(score, 3),
            'elo_vs_gate': round(elo_delta(score, args.gate_games), 0),
        })
        print(f"round {it + 1}: mse {mse:.4f}  gate {score * 100:.0f}% {wdl}  "
              f"({time.time() - t0:.0f}s){marker}\n")

    print(f"training done in {time.time() - start:.0f}s. "
          f"gate vs {args.gate_opponent}: {base_score * 100:.0f}% → {best_score * 100:.0f}%")
    print(f"saved to {args.out}")


if __name__ == '__main__':
    main()
