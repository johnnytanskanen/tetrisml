#!/usr/bin/env python3
import curses
import random
import time
import math
import os

BOARD_WIDTH = 10
BOARD_HEIGHT = 20
EMPTY = 0

PIECES = {
    'I': [[1,1,1,1]],
    'O': [[1,1],[1,1]],
    'T': [[0,1,0],[1,1,1]],
    'S': [[0,1,1],[1,1,0]],
    'Z': [[1,1,0],[0,1,1]],
    'L': [[1,0],[1,0],[1,1]],
    'J': [[0,1],[0,1],[1,1]],
}

PIECE_COLORS = {'I': 1, 'O': 2, 'T': 3, 'S': 4, 'Z': 5, 'L': 6, 'J': 7}
PIECE_NAMES = list(PIECES.keys())

SPARK_CHARS = " ▁▂▃▄▅▆▇█"
BAR_FULL = "█"
BAR_CHARS = " ▏▎▍▌▋▊▉█"


def rotate_cw(shape):
    return [list(row) for row in zip(*shape[::-1])]


def get_rotations(shape):
    rotations = [shape]
    current = shape
    for _ in range(3):
        current = rotate_cw(current)
        if current not in rotations:
            rotations.append(current)
    return rotations


class Board:
    def __init__(self):
        self.grid = [[EMPTY] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]
        self.color_grid = [[0] * BOARD_WIDTH for _ in range(BOARD_HEIGHT)]

    def copy(self):
        b = Board()
        b.grid = [row[:] for row in self.grid]
        b.color_grid = [row[:] for row in self.color_grid]
        return b

    def valid_position(self, shape, offset_x, offset_y):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    bx = offset_x + x
                    by = offset_y + y
                    if bx < 0 or bx >= BOARD_WIDTH or by >= BOARD_HEIGHT:
                        return False
                    if by >= 0 and self.grid[by][bx]:
                        return False
        return True

    def place_piece(self, shape, offset_x, offset_y, color):
        for y, row in enumerate(shape):
            for x, cell in enumerate(row):
                if cell:
                    by = offset_y + y
                    bx = offset_x + x
                    if 0 <= by < BOARD_HEIGHT and 0 <= bx < BOARD_WIDTH:
                        self.grid[by][bx] = 1
                        self.color_grid[by][bx] = color

    def clear_lines(self):
        lines_cleared = 0
        new_grid = []
        new_colors = []
        for y in range(BOARD_HEIGHT):
            if all(self.grid[y]):
                lines_cleared += 1
            else:
                new_grid.append(self.grid[y])
                new_colors.append(self.color_grid[y])
        for _ in range(lines_cleared):
            new_grid.insert(0, [EMPTY] * BOARD_WIDTH)
            new_colors.insert(0, [0] * BOARD_WIDTH)
        self.grid = new_grid
        self.color_grid = new_colors
        return lines_cleared

    def get_heights(self):
        heights = [0] * BOARD_WIDTH
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if self.grid[y][x]:
                    heights[x] = BOARD_HEIGHT - y
                    break
        return heights

    def count_holes(self):
        holes = 0
        for x in range(BOARD_WIDTH):
            found_block = False
            for y in range(BOARD_HEIGHT):
                if self.grid[y][x]:
                    found_block = True
                elif found_block:
                    holes += 1
        return holes

    def count_wells(self):
        wells = 0
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT):
                if self.grid[y][x]:
                    break
                left_wall = (x == 0) or (self.grid[y][x-1] != EMPTY)
                right_wall = (x == BOARD_WIDTH-1) or (self.grid[y][x+1] != EMPTY)
                if left_wall and right_wall:
                    wells += 1
        return wells

    def row_transitions(self):
        transitions = 0
        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH - 1):
                if bool(self.grid[y][x]) != bool(self.grid[y][x+1]):
                    transitions += 1
            if not self.grid[y][0]:
                transitions += 1
            if not self.grid[y][BOARD_WIDTH-1]:
                transitions += 1
        return transitions

    def col_transitions(self):
        transitions = 0
        for x in range(BOARD_WIDTH):
            for y in range(BOARD_HEIGHT - 1):
                if bool(self.grid[y][x]) != bool(self.grid[y+1][x]):
                    transitions += 1
            if not self.grid[BOARD_HEIGHT-1][x]:
                transitions += 1
        return transitions

    def aggregate_height(self):
        return sum(self.get_heights())

    def bumpiness(self):
        heights = self.get_heights()
        return sum(abs(heights[i] - heights[i+1]) for i in range(len(heights)-1))

    def max_height(self):
        return max(self.get_heights())

    def features(self):
        """Compute all board metrics in a few passes (hot path for the AI)."""
        grid = self.grid
        W = BOARD_WIDTH
        H = BOARD_HEIGHT

        heights = [0] * W
        holes = 0
        for x in range(W):
            top = -1
            for y in range(H):
                if grid[y][x]:
                    if top == -1:
                        top = y
                elif top != -1:
                    holes += 1
            heights[x] = (H - top) if top != -1 else 0

        agg = sum(heights)
        max_h = max(heights)
        bump = 0
        for i in range(W - 1):
            bump += abs(heights[i] - heights[i + 1])

        wells = 0
        for x in range(W):
            for y in range(H):
                if grid[y][x]:
                    break
                left = (x == 0) or grid[y][x - 1]
                right = (x == W - 1) or grid[y][x + 1]
                if left and right:
                    wells += 1

        # row transitions: walls count as filled on both sides
        row_tr = 0
        for y in range(H):
            row = grid[y]
            prev = True
            for x in range(W):
                cur = bool(row[x])
                if cur != prev:
                    row_tr += 1
                prev = cur
            if not prev:
                row_tr += 1

        # column transitions: floor counts as filled, ceiling does not
        col_tr = 0
        for x in range(W):
            prev = bool(grid[0][x])
            for y in range(1, H):
                cur = bool(grid[y][x])
                if cur != prev:
                    col_tr += 1
                prev = cur
            if not prev:
                col_tr += 1

        return {
            'heights': heights, 'agg': agg, 'max_h': max_h, 'bump': bump,
            'holes': holes, 'wells': wells, 'row_tr': row_tr, 'col_tr': col_tr,
        }


# Hand-tuned starting point (Dellacherie-style). `train.py` can learn better
# weights and write them to weights.json, which is loaded automatically below.
DEFAULT_WEIGHTS = {
    'lines_cleared': 7.6, 'holes': -5.8, 'aggregate_height': -0.51,
    'bumpiness': -0.38, 'wells': -0.5, 'row_transitions': -0.6,
    'col_transitions': -0.8, 'max_height': -0.2, 'perfect_clear': 20.0,
    'landing_height': -0.3,
}

WEIGHT_KEYS = list(DEFAULT_WEIGHTS.keys())


def load_weights(path=None):
    """Load learned weights from weights.json if present, else return None."""
    import json
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights.json')
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get('weights', data)
    except (FileNotFoundError, ValueError, OSError):
        return None


def load_weights_meta(path=None):
    """Return the 'meta' block from weights.json (training progress), or None."""
    import json
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights.json')
    try:
        with open(path) as f:
            return json.load(f).get('meta')
    except (FileNotFoundError, ValueError, OSError):
        return None


class TetrisAI:
    def __init__(self, weights=None):
        self.weights = dict(DEFAULT_WEIGHTS)
        learned = weights if weights is not None else load_weights()
        if learned:
            self.weights.update({k: v for k, v in learned.items() if k in self.weights})
        self.last_metrics = {}
        self.last_candidates = []

    def evaluate(self, board, lines_cleared, landing_height):
        f = board.features()
        metrics = {
            'lines': lines_cleared, 'holes': f['holes'],
            'height': f['agg'], 'bumpy': f['bump'],
            'wells': f['wells'], 'row_tr': f['row_tr'],
            'col_tr': f['col_tr'], 'max_h': f['max_h'],
            'land_h': landing_height,
        }
        w = self.weights
        score = (w['lines_cleared'] * lines_cleared + w['holes'] * f['holes'] +
                 w['aggregate_height'] * f['agg'] + w['bumpiness'] * f['bump'] +
                 w['wells'] * f['wells'] + w['row_transitions'] * f['row_tr'] +
                 w['col_transitions'] * f['col_tr'] + w['max_height'] * f['max_h'] +
                 w['landing_height'] * landing_height)
        if f['agg'] == 0:
            score += w['perfect_clear']
        return score, metrics

    def find_best_move(self, board, piece_name, next_piece_name=None, track=True):
        shape = PIECES[piece_name]
        rotations = get_rotations(shape)
        best_score = -math.inf
        best_move = None
        best_metrics = None
        all_scores = []

        for rot_idx, rot_shape in enumerate(rotations):
            piece_h = len(rot_shape)
            for x in range(-2, BOARD_WIDTH + 1):
                if not board.valid_position(rot_shape, x, 0):
                    if not board.valid_position(rot_shape, x, -piece_h):
                        continue
                drop_y = 0
                while board.valid_position(rot_shape, x, drop_y + 1):
                    drop_y += 1
                if not board.valid_position(rot_shape, x, drop_y):
                    continue

                test_board = board.copy()
                color = PIECE_COLORS[piece_name]
                test_board.place_piece(rot_shape, x, drop_y, color)
                lines = test_board.clear_lines()
                landing_h = BOARD_HEIGHT - drop_y

                if next_piece_name:
                    inner_best = -math.inf
                    inner_metrics = None
                    next_shape = PIECES[next_piece_name]
                    next_rots = get_rotations(next_shape)
                    for nr_shape in next_rots:
                        for nx in range(-2, BOARD_WIDTH + 1):
                            nd = 0
                            while test_board.valid_position(nr_shape, nx, nd + 1):
                                nd += 1
                            if not test_board.valid_position(nr_shape, nx, nd):
                                continue
                            tb2 = test_board.copy()
                            tb2.place_piece(nr_shape, nx, nd, PIECE_COLORS[next_piece_name])
                            l2 = tb2.clear_lines()
                            lh2 = BOARD_HEIGHT - nd
                            s, m = self.evaluate(tb2, lines + l2, (landing_h + lh2) / 2)
                            if s > inner_best:
                                inner_best = s
                                inner_metrics = m
                    if inner_best > -math.inf:
                        score, metrics = inner_best, inner_metrics
                    else:
                        score, metrics = self.evaluate(test_board, lines, landing_h)
                else:
                    score, metrics = self.evaluate(test_board, lines, landing_h)

                if track:
                    all_scores.append(score)
                if score > best_score:
                    best_score = score
                    best_move = (rot_idx, x, drop_y, rot_shape)
                    best_metrics = metrics

        if track:
            if best_metrics:
                self.last_metrics = best_metrics
            self.last_candidates = sorted(all_scores, reverse=True)[:30]
        return best_move


def simulate(weights, seed=0, max_pieces=500, lookahead=False):
    """Play one headless game with the given weights. Returns stats dict.

    Used by train.py to score candidate weight vectors. No rendering, no
    animation — pieces are placed directly at their chosen landing spot.
    """
    rng = random.Random(seed)
    board = Board()
    ai = TetrisAI(weights=weights)
    bag = []

    def next_piece():
        if not bag:
            new = PIECE_NAMES[:]
            rng.shuffle(new)
            bag.extend(new)
        return bag.pop()

    current = next_piece()
    nxt = next_piece()
    lines_total = 0
    pieces = 0
    tetrises = 0

    while pieces < max_pieces:
        move = ai.find_best_move(board, current, nxt if lookahead else None, track=False)
        if move is None:
            break
        _, x, drop_y, shape = move
        board.place_piece(shape, x, drop_y, PIECE_COLORS[current])
        cleared = board.clear_lines()
        lines_total += cleared
        if cleared == 4:
            tetrises += 1
        pieces += 1

        current, nxt = nxt, next_piece()
        spawn = PIECES[current]
        sx = BOARD_WIDTH // 2 - len(spawn[0]) // 2
        if not board.valid_position(spawn, sx, 0):
            break

    return {'lines': lines_total, 'pieces': pieces, 'tetrises': tetrises,
            'topped_out': pieces < max_pieces}


class TetrisGame:
    SCORING = {0: 0, 1: 100, 2: 300, 3: 500, 4: 800}

    def __init__(self):
        self.board = Board()
        self.ai = TetrisAI()
        self.score = 0
        self.lines = 0
        self.level = 1
        self.pieces_placed = 0
        self.bag = []
        self.current_piece = None
        self.next_piece = None
        self.current_shape = None
        self.piece_x = 0
        self.piece_y = 0
        self.game_over = False
        self.ai_target = None
        self.ai_phase = 'new'
        self.drop_speed = 0.03
        self.stats = {name: 0 for name in PIECE_NAMES}
        self.hole_history = []
        self.height_history = []
        self.bump_history = []
        self.lines_per_bucket = []
        self.bucket_lines = 0
        self.bucket_pieces = 0

        self._refill_bag()
        self.current_piece = self.bag.pop()
        self.next_piece = self.bag.pop()
        self._spawn_piece()

    def _refill_bag(self):
        if len(self.bag) < 2:
            new_bag = PIECE_NAMES[:]
            random.shuffle(new_bag)
            self.bag = new_bag + self.bag

    def _spawn_piece(self):
        self.current_shape = PIECES[self.current_piece]
        pw = len(self.current_shape[0])
        self.piece_x = BOARD_WIDTH // 2 - pw // 2
        self.piece_y = -len(self.current_shape)
        self.ai_phase = 'new'
        self.ai_target = None
        if not self.board.valid_position(self.current_shape, self.piece_x, self.piece_y + 1):
            if not self.board.valid_position(self.current_shape, self.piece_x, 0):
                self.game_over = True

    def _lock_piece(self):
        color = PIECE_COLORS[self.current_piece]
        self.board.place_piece(self.current_shape, self.piece_x, self.piece_y, color)
        lines = self.board.clear_lines()
        self.lines += lines
        self.score += self.SCORING.get(lines, 800) * self.level
        self.pieces_placed += 1
        self.level = self.lines // 10 + 1
        self.stats[self.current_piece] += 1

        self.hole_history.append(self.board.count_holes())
        self.height_history.append(self.board.max_height())
        self.bump_history.append(self.board.bumpiness())
        max_hist = 80
        if len(self.hole_history) > max_hist:
            self.hole_history = self.hole_history[-max_hist:]
            self.height_history = self.height_history[-max_hist:]
            self.bump_history = self.bump_history[-max_hist:]

        self.bucket_lines += lines
        self.bucket_pieces += 1
        if self.bucket_pieces >= 10:
            self.lines_per_bucket.append(self.bucket_lines)
            self.bucket_lines = 0
            self.bucket_pieces = 0
            if len(self.lines_per_bucket) > 30:
                self.lines_per_bucket = self.lines_per_bucket[-30:]

        self._refill_bag()
        self.current_piece = self.next_piece
        self.next_piece = self.bag.pop()
        self._spawn_piece()

    def ai_step(self):
        if self.game_over:
            return
        if self.ai_phase == 'new':
            move = self.ai.find_best_move(self.board, self.current_piece, self.next_piece)
            if move:
                self.ai_target = move
                self.ai_phase = 'rotate'
            else:
                self.ai_phase = 'drop'
        if self.ai_phase == 'rotate':
            _, target_x, _, target_shape = self.ai_target
            if self.current_shape != target_shape:
                new_shape = rotate_cw(self.current_shape)
                if self.board.valid_position(new_shape, self.piece_x, self.piece_y):
                    self.current_shape = new_shape
                    return
                for kick in [1, -1, 2, -2]:
                    if self.board.valid_position(new_shape, self.piece_x + kick, self.piece_y):
                        self.current_shape = new_shape
                        self.piece_x += kick
                        return
            self.ai_phase = 'move'
        if self.ai_phase == 'move':
            _, target_x, _, _ = self.ai_target
            if self.piece_x < target_x:
                if self.board.valid_position(self.current_shape, self.piece_x + 1, self.piece_y):
                    self.piece_x += 1
                    return
            elif self.piece_x > target_x:
                if self.board.valid_position(self.current_shape, self.piece_x - 1, self.piece_y):
                    self.piece_x -= 1
                    return
            self.ai_phase = 'drop'
        if self.ai_phase == 'drop':
            if self.board.valid_position(self.current_shape, self.piece_x, self.piece_y + 1):
                self.piece_y += 1
            else:
                self._lock_piece()


def safe_addstr(win, y, x, text, attr=0):
    try:
        max_y, max_x = win.getmaxyx()
        if y < 0 or y >= max_y or x < 0 or x >= max_x:
            return
        text = text[:max_x - x]
        win.addstr(y, x, text, attr)
    except curses.error:
        pass


def draw_block(win, y, x, color_pair):
    safe_addstr(win, y, x, "[]", curses.color_pair(color_pair))


def draw_piece_preview(win, shape, start_y, start_x, color):
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                draw_block(win, start_y + y, start_x + x * 2, color)


def sparkline(data, width, max_val=None):
    if not data:
        return " " * width
    trimmed = data[-width:]
    if max_val is None:
        max_val = max(trimmed) if max(trimmed) > 0 else 1
    result = ""
    for v in trimmed:
        idx = int(min(v / max_val, 1.0) * (len(SPARK_CHARS) - 1))
        result += SPARK_CHARS[idx]
    return result.ljust(width)


def horiz_bar(value, max_val, width):
    if max_val == 0:
        return " " * width
    ratio = min(abs(value) / max_val, 1.0)
    filled = ratio * width
    full_blocks = int(filled)
    remainder = filled - full_blocks
    result = BAR_FULL * full_blocks
    if remainder > 0 and full_blocks < width:
        idx = int(remainder * (len(BAR_CHARS) - 1))
        result += BAR_CHARS[idx]
    return result.ljust(width)


def draw_analytics(win, game, start_y, start_x):
    w = 34
    cp8 = curses.color_pair(8)
    cp10 = curses.color_pair(10) | curses.A_BOLD
    cp9 = curses.color_pair(9)
    cp5 = curses.color_pair(5)
    cp11 = curses.color_pair(11)
    cp1 = curses.color_pair(1)
    y = start_y

    safe_addstr(win, y, start_x, "─" * w, cp8 | curses.A_DIM)
    y += 1
    safe_addstr(win, y, start_x, "DECISION WEIGHTS", cp10)
    y += 1

    metrics = game.ai.last_metrics
    if metrics:
        ai_w = game.ai.weights
        items = [
            ('lines', metrics.get('lines', 0), ai_w['lines_cleared']),
            ('holes', metrics.get('holes', 0), ai_w['holes']),
            ('height', metrics.get('height', 0), ai_w['aggregate_height']),
            ('bumpy', metrics.get('bumpy', 0), ai_w['bumpiness']),
            ('wells', metrics.get('wells', 0), ai_w['wells']),
            ('row_tr', metrics.get('row_tr', 0), ai_w['row_transitions']),
            ('col_tr', metrics.get('col_tr', 0), ai_w['col_transitions']),
        ]
        max_weighted = 1
        for name, raw, weight in items:
            max_weighted = max(max_weighted, abs(raw * weight))

        for name, raw, weight in items:
            weighted = raw * weight
            bar = horiz_bar(weighted, max_weighted, 12)
            color = cp9 if weighted >= 0 else cp5
            safe_addstr(win, y, start_x, f" {name:<7}", cp8)
            safe_addstr(win, y, start_x + 8, bar, color)
            safe_addstr(win, y, start_x + 21, f"{weighted:>7.1f}", color)
            y += 1
    else:
        safe_addstr(win, y, start_x, " waiting...", cp8 | curses.A_DIM)
        y += 1
    y += 1

    safe_addstr(win, y, start_x, "─" * w, cp8 | curses.A_DIM)
    y += 1
    safe_addstr(win, y, start_x, "CANDIDATES", cp10)
    safe_addstr(win, y, start_x + 11, f" ({len(game.ai.last_candidates)} scored)", cp8 | curses.A_DIM)
    y += 1
    if game.ai.last_candidates:
        cands = game.ai.last_candidates
        mx = cands[0] if cands else 1
        mn = cands[-1] if cands else 0
        rng = mx - mn if mx != mn else 1
        bar_str = ""
        for s in cands:
            norm = (s - mn) / rng
            idx = int(norm * (len(SPARK_CHARS) - 1))
            bar_str += SPARK_CHARS[idx]
        safe_addstr(win, y, start_x + 1, bar_str[:w-2], cp9)
        y += 1
        safe_addstr(win, y, start_x + 1, f"best {mx:.1f}", cp9)
        safe_addstr(win, y, start_x + 16, f"worst {mn:.1f}", cp5)
    else:
        safe_addstr(win, y, start_x, " waiting...", cp8 | curses.A_DIM)
    y += 2

    safe_addstr(win, y, start_x, "─" * w, cp8 | curses.A_DIM)
    y += 1
    safe_addstr(win, y, start_x, "BOARD HEALTH", cp10)
    y += 1

    spark_w = w - 10
    safe_addstr(win, y, start_x, " holes ", cp5)
    max_h = max(game.hole_history) if game.hole_history and max(game.hole_history) > 0 else 5
    safe_addstr(win, y, start_x + 8, sparkline(game.hole_history, spark_w, max_h), cp5)
    curr = game.hole_history[-1] if game.hole_history else 0
    safe_addstr(win, y, start_x + 8 + spark_w, f" {curr}", cp5)
    y += 1

    safe_addstr(win, y, start_x, " max h ", cp1)
    max_mh = max(game.height_history) if game.height_history and max(game.height_history) > 0 else 10
    safe_addstr(win, y, start_x + 8, sparkline(game.height_history, spark_w, max_mh), cp1)
    curr = game.height_history[-1] if game.height_history else 0
    safe_addstr(win, y, start_x + 8 + spark_w, f" {curr}", cp1)
    y += 1

    safe_addstr(win, y, start_x, " bumpy ", cp11)
    max_b = max(game.bump_history) if game.bump_history and max(game.bump_history) > 0 else 10
    safe_addstr(win, y, start_x + 8, sparkline(game.bump_history, spark_w, max_b), cp11)
    curr = game.bump_history[-1] if game.bump_history else 0
    safe_addstr(win, y, start_x + 8 + spark_w, f" {curr}", cp11)
    y += 2

    safe_addstr(win, y, start_x, "─" * w, cp8 | curses.A_DIM)
    y += 1
    safe_addstr(win, y, start_x, "LINES / 10 PIECES", cp10)
    y += 1
    if game.lines_per_bucket:
        max_lp = max(game.lines_per_bucket) if max(game.lines_per_bucket) > 0 else 4
        safe_addstr(win, y, start_x + 1, sparkline(game.lines_per_bucket, min(len(game.lines_per_bucket), spark_w), max_lp), cp9)
        avg = sum(game.lines_per_bucket) / len(game.lines_per_bucket)
        safe_addstr(win, y + 1, start_x + 1, f"avg {avg:.1f} lines", cp8 | curses.A_DIM)
    else:
        safe_addstr(win, y, start_x, " waiting...", cp8 | curses.A_DIM)
    y += 3

    safe_addstr(win, y, start_x, "─" * w, cp8 | curses.A_DIM)
    y += 1
    safe_addstr(win, y, start_x, "COLUMN HEIGHTS", cp10)
    y += 1
    heights = game.board.get_heights()
    for i, h in enumerate(heights):
        bar = horiz_bar(h, BOARD_HEIGHT, 14)
        safe_addstr(win, y, start_x, f" {i}", cp8 | curses.A_DIM)
        safe_addstr(win, y, start_x + 3, bar, cp1)
        safe_addstr(win, y, start_x + 18, f"{h:>2}", cp8)
        y += 1


def main(stdscr):
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(10)

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_CYAN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_YELLOW)
    curses.init_pair(3, curses.COLOR_MAGENTA, curses.COLOR_MAGENTA)
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_GREEN)
    curses.init_pair(5, curses.COLOR_RED, -1)
    curses.init_pair(6, curses.COLOR_WHITE, curses.COLOR_WHITE)
    curses.init_pair(7, curses.COLOR_BLUE, curses.COLOR_BLUE)
    curses.init_pair(8, curses.COLOR_WHITE, -1)
    curses.init_pair(9, curses.COLOR_GREEN, -1)
    curses.init_pair(10, curses.COLOR_CYAN, -1)
    curses.init_pair(11, curses.COLOR_YELLOW, -1)
    curses.init_pair(12, curses.COLOR_MAGENTA, -1)

    game = TetrisGame()
    paused = False
    speed_mult = 1.0
    last_step = time.time()
    last_reload = time.time()
    train_meta = load_weights_meta()

    BOARD_X = 2
    BOARD_Y = 1
    PANEL_X = BOARD_X + BOARD_WIDTH * 2 + 3
    GRAPH_X = PANEL_X + 16

    while True:
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
        elif key == ord('r'):
            game = TetrisGame()
            paused = False
        elif key == ord('+') or key == ord('='):
            speed_mult = min(speed_mult * 1.5, 50.0)
        elif key == ord('-'):
            speed_mult = max(speed_mult / 1.5, 0.1)
        elif key == ord(' '):
            speed_mult = 1.0

        if not paused and not game.game_over:
            now = time.time()
            step_interval = game.drop_speed / speed_mult
            if now - last_step >= step_interval:
                game.ai_step()
                last_step = now

        # hot-reload weights while a background trainer improves them
        if time.time() - last_reload >= 3.0:
            last_reload = time.time()
            learned = load_weights()
            if learned:
                game.ai.weights.update({k: v for k, v in learned.items()
                                        if k in game.ai.weights})
            train_meta = load_weights_meta()

        stdscr.erase()

        title = "TETRIS AI"
        safe_addstr(stdscr, 0, BOARD_X, title, curses.color_pair(10) | curses.A_BOLD)
        status = "PAUSED" if paused else ("GAME OVER" if game.game_over else "RUNNING")
        s_color = curses.color_pair(11) if paused else (curses.color_pair(5) if game.game_over else curses.color_pair(9))
        safe_addstr(stdscr, 0, BOARD_X + len(title) + 1, status, s_color | curses.A_BOLD)

        if train_meta:
            tag = f"learning · gen {train_meta.get('generation', '?')} " \
                  f"({train_meta.get('fitness_mean_lines', '?')} lines)"
            safe_addstr(stdscr, 0, BOARD_X + len(title) + 2 + len(status) + 1, tag,
                        curses.color_pair(12) | curses.A_DIM)

        for y in range(BOARD_HEIGHT):
            safe_addstr(stdscr, BOARD_Y + y, BOARD_X - 1, "│", curses.color_pair(8) | curses.A_DIM)
            safe_addstr(stdscr, BOARD_Y + y, BOARD_X + BOARD_WIDTH * 2, "│", curses.color_pair(8) | curses.A_DIM)
        safe_addstr(stdscr, BOARD_Y + BOARD_HEIGHT, BOARD_X - 1, "└" + "─" * (BOARD_WIDTH * 2) + "┘", curses.color_pair(8) | curses.A_DIM)

        for y in range(BOARD_HEIGHT):
            for x in range(BOARD_WIDTH):
                if game.board.grid[y][x]:
                    draw_block(stdscr, BOARD_Y + y, BOARD_X + x * 2, game.board.color_grid[y][x])

        if not game.game_over:
            ghost_y = game.piece_y
            while game.board.valid_position(game.current_shape, game.piece_x, ghost_y + 1):
                ghost_y += 1
            for y, row in enumerate(game.current_shape):
                for x, cell in enumerate(row):
                    if cell and 0 <= ghost_y + y < BOARD_HEIGHT:
                        safe_addstr(stdscr, BOARD_Y + ghost_y + y, BOARD_X + (game.piece_x + x) * 2,
                                    "::", curses.color_pair(8) | curses.A_DIM)

            color = PIECE_COLORS[game.current_piece]
            for y, row in enumerate(game.current_shape):
                for x, cell in enumerate(row):
                    if cell and game.piece_y + y >= 0:
                        draw_block(stdscr, BOARD_Y + game.piece_y + y, BOARD_X + (game.piece_x + x) * 2, color)

        py = BOARD_Y
        safe_addstr(stdscr, py, PANEL_X, "SCORE", curses.color_pair(10) | curses.A_BOLD)
        safe_addstr(stdscr, py + 1, PANEL_X, f" {game.score:,}", curses.color_pair(11))
        py += 3
        safe_addstr(stdscr, py, PANEL_X, "LINES", curses.color_pair(10) | curses.A_BOLD)
        safe_addstr(stdscr, py + 1, PANEL_X, f" {game.lines}", curses.color_pair(8))
        py += 3
        safe_addstr(stdscr, py, PANEL_X, "LEVEL", curses.color_pair(10) | curses.A_BOLD)
        safe_addstr(stdscr, py + 1, PANEL_X, f" {game.level}", curses.color_pair(12))
        py += 3
        safe_addstr(stdscr, py, PANEL_X, "PIECES", curses.color_pair(10) | curses.A_BOLD)
        safe_addstr(stdscr, py + 1, PANEL_X, f" {game.pieces_placed}", curses.color_pair(8))
        py += 3
        safe_addstr(stdscr, py, PANEL_X, "NEXT", curses.color_pair(10) | curses.A_BOLD)
        py += 1
        draw_piece_preview(stdscr, PIECES[game.next_piece], py, PANEL_X + 1, PIECE_COLORS[game.next_piece])
        py += len(PIECES[game.next_piece]) + 1
        safe_addstr(stdscr, py, PANEL_X, "SPEED", curses.color_pair(10) | curses.A_BOLD)
        safe_addstr(stdscr, py + 1, PANEL_X, f" {speed_mult:.1f}x", curses.color_pair(8))
        py += 3
        safe_addstr(stdscr, py, PANEL_X, "STATS", curses.color_pair(10) | curses.A_BOLD)
        py += 1
        for name in PIECE_NAMES:
            c = PIECE_COLORS[name]
            safe_addstr(stdscr, py, PANEL_X, f" {name}", curses.color_pair(c) if c not in (1, 5, 12) else curses.color_pair(c))
            safe_addstr(stdscr, py, PANEL_X + 3, f" {game.stats[name]:>3}", curses.color_pair(8))
            py += 1

        draw_analytics(stdscr, game, BOARD_Y, GRAPH_X)

        controls_y = BOARD_Y + BOARD_HEIGHT + 2
        safe_addstr(stdscr, controls_y, BOARD_X - 1,
                    "+/- speed  SPC reset  p pause  r restart  q quit",
                    curses.color_pair(8) | curses.A_DIM)

        if game.game_over:
            msg = " GAME OVER "
            safe_addstr(stdscr, BOARD_Y + BOARD_HEIGHT // 2 - 1,
                        BOARD_X + BOARD_WIDTH - len(msg) // 2, msg,
                        curses.color_pair(5) | curses.A_BOLD)
            msg2 = " r to restart "
            safe_addstr(stdscr, BOARD_Y + BOARD_HEIGHT // 2 + 1,
                        BOARD_X + BOARD_WIDTH - len(msg2) // 2, msg2,
                        curses.color_pair(8))

        stdscr.refresh()


if __name__ == '__main__':
    import autotrain
    trainer = autotrain.start_background_training()
    try:
        curses.wrapper(main)
    finally:
        autotrain.stop_background_training(trainer)
