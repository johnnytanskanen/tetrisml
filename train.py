#!/usr/bin/env python3
"""
Learn the Tetris AI's evaluation weights with a genetic algorithm.

The game AI scores each candidate placement as a weighted sum of board
features (holes, bumpiness, aggregate height, ...). Those weights are what
make the AI good or bad. Instead of hand-tuning them, this script *learns*
them: it evolves a population of weight vectors, scoring each by how many
lines it clears in headless self-play, and breeds the best ones.

This is the actual "machine learning" part of the project. It's a black-box
evolutionary search (no gradients, no labels) — the standard, well-suited
method for optimizing Tetris heuristics.

Usage:
    python3 train.py                      # sensible defaults (~1-2 min on 8 cores)
    python3 train.py --generations 40 --population 60 --pieces 500
    python3 train.py --resume             # seed the population from weights.json

The best weight vector is written to weights.json, which the game
(tetris_ai.py) and both web UIs load automatically.
"""
import argparse
import json
import math
import os
import random
import time
from multiprocessing import Pool, cpu_count

import tetris_ai
from tetris_ai import WEIGHT_KEYS, DEFAULT_WEIGHTS, simulate

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, 'weights.json')


# ----- genome helpers -----------------------------------------------------

def random_genome(rng):
    # lines_cleared/perfect_clear lean positive, the rest lean negative —
    # just a head start; evolution is free to flip any sign.
    g = {}
    for k in WEIGHT_KEYS:
        if k in ('lines_cleared', 'perfect_clear'):
            g[k] = abs(rng.gauss(0, 1))
        else:
            g[k] = -abs(rng.gauss(0, 1))
    return normalize(g)


def normalize(g):
    norm = math.sqrt(sum(v * v for v in g.values())) or 1.0
    return {k: v / norm for k, v in g.items()}


def crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in WEIGHT_KEYS}


def mutate(g, rng, rate, sigma):
    out = dict(g)
    for k in WEIGHT_KEYS:
        if rng.random() < rate:
            out[k] += rng.gauss(0, sigma)
    return normalize(out)


# ----- fitness (runs in worker processes) ---------------------------------

def _eval(args):
    """Score one genome: mean lines cleared across a fixed set of seeds."""
    genome, seeds, pieces = args
    total_lines = 0
    total_pieces = 0
    for s in seeds:
        r = simulate(genome, seed=s, max_pieces=pieces, lookahead=False)
        total_lines += r['lines']
        total_pieces += r['pieces']
    n = len(seeds)
    # fitness is mean lines; mean pieces is a tiebreaker for early generations
    return total_lines / n + (total_pieces / n) * 1e-4


def evaluate_population(pop, seeds, pieces, pool):
    args = [(g, seeds, pieces) for g in pop]
    if pool is not None:
        return pool.map(_eval, args)
    return [_eval(a) for a in args]


# ----- main loop -----------------------------------------------------------

def train(generations, population, seeds_per_eval, pieces, workers, resume, seed):
    rng = random.Random(seed)

    pop = [random_genome(rng) for _ in range(population)]
    if resume:
        learned = tetris_ai.load_weights()
        if learned:
            pop[0] = normalize({k: learned.get(k, DEFAULT_WEIGHTS[k]) for k in WEIGHT_KEYS})
            print("Resuming from weights.json")
    # always include the hand-tuned baseline as one competitor
    pop[-1] = normalize(dict(DEFAULT_WEIGHTS))

    pool = Pool(workers) if workers > 1 else None
    best_genome, best_fitness = None, -1.0
    elite_count = max(2, population // 10)

    print(f"GA: pop={population} gens={generations} seeds/eval={seeds_per_eval} "
          f"pieces={pieces} workers={workers}\n")

    try:
        for gen in range(generations):
            # same piece sequences for every genome this generation → fair contest
            seeds = [gen * 1009 + i for i in range(seeds_per_eval)]
            t0 = time.time()
            fits = evaluate_population(pop, seeds, pieces, pool)
            ranked = sorted(zip(fits, pop), key=lambda p: p[0], reverse=True)

            gen_best_fit, gen_best_genome = ranked[0]
            if gen_best_fit > best_fitness:
                best_fitness = gen_best_fit
                best_genome = gen_best_genome
                save_weights(best_genome, best_fitness, gen, pieces)
                marker = " *saved"
            else:
                marker = ""

            mean_fit = sum(fits) / len(fits)
            dt = time.time() - t0
            print(f"gen {gen:2d}  best {gen_best_fit:7.1f}  mean {mean_fit:7.1f}  "
                  f"({dt:.1f}s){marker}")

            # breed next generation
            elites = [g for _, g in ranked[:elite_count]]
            sigma = 0.18 * (1 - gen / generations) + 0.03  # anneal mutation
            new_pop = list(elites)
            while len(new_pop) < population:
                pa = tournament(ranked, rng)
                pb = tournament(ranked, rng)
                child = mutate(crossover(pa, pb, rng), rng, rate=0.25, sigma=sigma)
                new_pop.append(child)
            pop = new_pop
    finally:
        if pool is not None:
            pool.close()
            pool.join()

    return best_genome, best_fitness


def tournament(ranked, rng, k=3):
    contenders = [rng.choice(ranked) for _ in range(k)]
    return max(contenders, key=lambda p: p[0])[1]


def save_weights(genome, fitness, gen, pieces):
    # rescale so lines_cleared ~ 1 for human-readable weights (sign preserved)
    ref = abs(genome.get('lines_cleared', 0)) or 1.0
    scaled = {k: round(v / ref, 4) for k, v in genome.items()}
    payload = {
        'weights': scaled,
        'meta': {
            'method': 'genetic algorithm (self-play)',
            'fitness_mean_lines': round(fitness, 1),
            'generation': gen,
            'eval_pieces': pieces,
        },
    }
    # atomic write: the game reads weights.json concurrently
    tmp = OUT_PATH + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, OUT_PATH)


def benchmark(weights, label, seeds=(101, 202, 303, 404, 505), pieces=1000):
    total = 0
    for s in seeds:
        r = simulate(weights, seed=s, max_pieces=pieces, lookahead=False)
        total += r['lines']
    print(f"  {label:18s} {total / len(seeds):7.1f} lines / {pieces} pieces "
          f"(avg over {len(seeds)} games)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--generations', type=int, default=15)
    ap.add_argument('--population', type=int, default=24)
    ap.add_argument('--seeds', type=int, default=4, help='games per fitness eval')
    ap.add_argument('--pieces', type=int, default=200, help='max pieces per game')
    ap.add_argument('--workers', type=int, default=cpu_count())
    ap.add_argument('--resume', action='store_true', help='seed from weights.json')
    ap.add_argument('--seed', type=int, default=42, help='RNG seed for the GA')
    args = ap.parse_args()

    start = time.time()
    best, fit = train(args.generations, args.population, args.seeds,
                      args.pieces, args.workers, args.resume, args.seed)
    print(f"\nTraining done in {time.time() - start:.0f}s. "
          f"Best fitness: {fit:.1f} mean lines.")
    print(f"Saved to {OUT_PATH}\n")

    print("Benchmark (longer games, unseen seeds):")
    benchmark(DEFAULT_WEIGHTS, 'hand-tuned')
    benchmark(best, 'learned')


if __name__ == '__main__':
    main()
