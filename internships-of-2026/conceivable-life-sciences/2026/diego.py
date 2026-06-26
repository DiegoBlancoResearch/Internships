import os
import sys
import threading
import time
import random
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2

from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exploration_utils import Meca500, MVCamera


WIN_LINES = [
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
]


def check_winner(board):
    for a, b, c in WIN_LINES:
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return board[a]
    return None


def is_draw(board):
    return all(v != 0 for v in board) and check_winner(board) is None


def calcular_distancia(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)


def es_numero_feliz(n):
    vistos = set()
    while n != 1 and n not in vistos:
        vistos.add(n)
        n = sum(int(d)**2 for d in str(n))
    return n == 1


class Q_learning_agent:
    def __init__(self):
        self.q_table = {}
        self.alpha = 0.5
        self.gamma = 0.9
        self.epsilon = 1.0

    def _state_for_player(self, board, player):
        opp = 3 - player
        return tuple(1 if v == player else (2 if v == opp else 0) for v in board)

    def choose_action(self, state, available):
        if np.random.random() < self.epsilon:
            return int(np.random.choice(available))
        q_vals = [self.q_table.get((state, a), 0.0) for a in available]
        return available[int(np.argmax(q_vals))]

    def choose_best_action(self, state, available):
        q_vals = [self.q_table.get((state, a), 0.0) for a in available]
        return available[int(np.argmax(q_vals))]

    def update_q_value(self, state, action, reward, next_state, next_available):
        current_q = self.q_table.get((state, action), 0.0)
        max_next = max(
            (self.q_table.get((next_state, a), 0.0) for a in next_available),
            default=0.0,
        )
        self.q_table[(state, action)] = current_q + self.alpha * (
            reward + self.gamma * max_next - current_q
        )

    def train_agent(self, episodes=100_000):
        print(f"entrenando IA {episodes} episodios...")
        for _ in range(episodes):
            board = [0] * 9
            current = 1
            prev: dict = {1: None, 2: None}
            done = False

            while not done:
                available = [i for i, v in enumerate(board) if v == 0]
                if not available:
                    for p, pa in prev.items():
                        if pa is not None:
                            s, a = pa
                            self.update_q_value(s, a, 0.5, self._state_for_player(board, p), [])
                    break

                state = self._state_for_player(board, current)
                action = self.choose_action(state, available)
                board[action] = current

                winner = check_winner(board)
                opp = 3 - current
                nxt_avail = [i for i, v in enumerate(board) if v == 0]
                nxt_state = self._state_for_player(board, current)

                if winner is not None:
                    self.update_q_value(state, action, 1.0, nxt_state, [])
                    if prev[opp] is not None:
                        ps, pa = prev[opp]
                        self.update_q_value(ps, pa, -1.0, self._state_for_player(board, opp), [])
                    done = True
                elif not nxt_avail:
                    self.update_q_value(state, action, 0.5, nxt_state, [])
                    if prev[opp] is not None:
                        ps, pa = prev[opp]
                        self.update_q_value(ps, pa, 0.5, self._state_for_player(board, opp), [])
                    done = True
                else:
                    if prev[opp] is not None:
                        ps, pa = prev[opp]
                        self.update_q_value(ps, pa, 0.0, self._state_for_player(board, opp), nxt_avail)

                prev[current] = (state, action)
                current = opp

            self.epsilon = max(0.05, self.epsilon * 0.99997)

        self.epsilon = 0.0
        print("listo")


CELL_POSES = [
    [164.591499,  56.368881, 66.195787, 120.0,      90.0,      0.0        ],
    [219.661525,  53.982017, 66.211951, 120.0,      90.0,      0.0        ],
    [261.533318,  55.421942, 63.444553, -0.546593,  88.975713, 120.538155 ],
    [171.169144,   0.0,      76.7567,   120.0,      90.0,      0.0        ],
    [219.455632,   4.782829, 67.070581, 120.0,      90.0,      0.0        ],
    [261.533318,  -1.39502,  64.944553, -0.546593,  88.975713, 120.538155 ],
    [166.391499, -41.965752, 68.095787, 120.0,      90.0,      0.0        ],
    [219.455632, -45.630353, 62.670581, 120.0,      90.0,      0.0        ],
    [260.136128, -44.633185, 65.955068, 120.0,      90.0,      0.0        ],
]


class Plotter:
    MARK_H = 8.0
    MARK_R = 7.0
    SAFE_DZ = 15.0

    def __init__(self, robot):
        self.robot = robot

    def draw_x(self, cell, sides=8):
        px, py, pz, rx, ry, rz = CELL_POSES[cell]
        r = self.MARK_H
        s = pz - self.SAFE_DZ
        self.robot.move_lin([px + r, py, pz, rx, ry, rz])
        self.robot.move_lin([px + r, py, s,  rx, ry, rz])
        for i in range(1, sides + 1):
            ang = 2 * math.pi * i / sides
            self.robot.move_lin([px + r*math.cos(ang), py + r*math.sin(ang), s, rx, ry, rz])
        self.robot.move_lin([px, py, pz, rx, ry, rz])

    def draw_o(self, cell, segments=16):
        px, py, pz, rx, ry, rz = CELL_POSES[cell]
        r = self.MARK_R
        s = pz - self.SAFE_DZ
        self.robot.move_lin([px+r, py, pz, rx, ry, rz])
        self.robot.move_lin([px+r, py, s,  rx, ry, rz])
        for i in range(1, segments + 1):
            ang = 2 * math.pi * i / segments
            self.robot.move_lin([px + r*math.cos(ang), py + r*math.sin(ang), s, rx, ry, rz])
        self.robot.move_lin([px, py, pz, rx, ry, rz])

    def erase_board(self):
        ERASE_DZ = 12.0
        HALF = 24.0
        STEP = 15.9
        print("borrando...")
        go_right = True
        for cell in range(9):
            px, py, pz, rx, ry, rz = CELL_POSES[cell]
            erz = rz + 180.0
            s = pz - ERASE_DZ
            direction = go_right
            for p in range(3):
                dy = (p - 1) * STEP
                x1 = px - HALF if direction else px + HALF
                x2 = px + HALF if direction else px - HALF
                self.robot.move_lin([x1, py + dy, pz, rx, ry, erz])
                self.robot.move_lin([x1, py + dy, s,  rx, ry, erz])
                self.robot.move_lin([x2, py + dy, s,  rx, ry, erz])
                self.robot.move_lin([x2, py + dy, pz, rx, ry, erz])
                direction = not direction
            go_right = not go_right
        self.robot.move_joints([0, 0, 0, 0, 0, 0])
        print("listo")

    def draw_circle(self, x, y, r=10.0):
        pass

    def draw_triangle(self, x, y, size=15.0):
        pass

    def draw_star(self, x, y, size=12.0):
        pass


class TicTacToeGame:
    ROBOT = 1
    HUMAN = 2

    def __init__(self, agent, plotter, mode="human_vs_robot", vision=None):
        self.agent = agent
        self.plotter = plotter
        self.mode = mode
        self.board = [0] * 9
        self.vision = vision

    def display_board(self):
        sym = {0: ".", self.ROBOT: "X", self.HUMAN: "O"}
        b = self.board
        print(f"\n  {sym[b[0]]} | {sym[b[1]]} | {sym[b[2]]}    1 | 2 | 3")
        print(f"  --+---+--    --+---+--")
        print(f"  {sym[b[3]]} | {sym[b[4]]} | {sym[b[5]]}    4 | 5 | 6")
        print(f"  --+---+--    --+---+--")
        print(f"  {sym[b[6]]} | {sym[b[7]]} | {sym[b[8]]}    7 | 8 | 9\n")

    def coin_flip(self):
        first = random.choice(["robot", "human"])
        return first

    def _keyboard_move(self, available):
        print(f"celdas: {[c + 1 for c in available]}")
        while True:
            try:
                cell = int(input("celda (1-9): ").strip()) - 1
                if cell in available:
                    return cell
                print("ocupada")
            except (ValueError, EOFError):
                print("escribe un numero")

    def get_human_move(self):
        available = [i for i, v in enumerate(self.board) if v == 0]
        if self.vision is not None and self.vision.baseline is not None:
            cell = self.vision.wait_for_human_move(available)
            if cell == -1:
                return self._keyboard_move(available)
            if cell not in available:
                return self.get_human_move()
            return cell
        return self._keyboard_move(available)

    def _robot_pick(self, player):
        available = [i for i, v in enumerate(self.board) if v == 0]
        state = self.agent._state_for_player(self.board, player)
        return self.agent.choose_best_action(state, available)

    def _apply_move(self, cell, player):
        self.board[cell] = player
        if player == self.ROBOT:
            try:
                self.plotter.draw_x(cell)
            except Exception as e:
                print(f"error celda {cell+1}: {e}")
                try:
                    self.plotter.robot.clean_errors()
                except Exception:
                    pass
            try:
                self.plotter.robot.move_joints([0, 0, 0, 0, 0, 0])
            except Exception:
                pass
        elif self.mode == "robot_vs_robot":
            try:
                self.plotter.draw_o(cell)
            except Exception as e:
                print(f"error celda {cell+1}: {e}")
                try:
                    self.plotter.robot.clean_errors()
                except Exception:
                    pass
            try:
                self.plotter.robot.move_joints([0, 0, 0, 0, 0, 0])
            except Exception:
                pass

    def play_turn(self, player):
        if self.mode == "robot_vs_robot" or player == self.ROBOT:
            cell = self._robot_pick(player)
            print(f"robot: celda {cell + 1}")
        else:
            cell = self.get_human_move()
        self._apply_move(cell, player)

    def play_game(self):
        self.board = [0] * 9
        first = "robot" if self.mode == "robot_vs_robot" else "human"

        if self.vision is not None:
            self.vision.capture_baseline()

        order = [self.ROBOT, self.HUMAN] if first == "robot" else [self.HUMAN, self.ROBOT]
        turn = 0

        while True:
            player = order[turn % 2]
            self.play_turn(player)

            winner = check_winner(self.board)
            if winner is not None:
                if self.mode == "robot_vs_robot":
                    print("gana robot X" if winner == self.ROBOT else "gana robot O")
                elif winner == self.ROBOT:
                    print("gana el robot")
                else:
                    print("ganaste")
                try:
                    self.plotter.erase_board()
                except Exception as e:
                    print(f"error borrando: {e}")
                break
            if is_draw(self.board):
                print("empate")
                try:
                    self.plotter.erase_board()
                except Exception as e:
                    print(f"error borrando: {e}")
                break
            turn += 1

        return check_winner(self.board) or 0


class CameraManager:
    def __init__(self, cam, window_name="Camera"):
        self.cam = cam
        self.window_name = window_name
        self.stop_event = threading.Event()
        self.thread = None
        self.frame_count = 0
        self.line_counts = []

    def calculate_red_dots(self, frame):
        if frame is None:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lr = cv2.inRange(hsv, (0, 90, 50), (10, 255, 255))
        dr = cv2.inRange(hsv, (160, 90, 50), (180, 255, 255))
        mask = cv2.bitwise_or(lr, dr)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if 200 <= cv2.contourArea(c) <= 5000]
        if len(valid) > 8:
            areas = np.array([cv2.contourArea(c) for c in valid])
            valid = sorted(valid, key=lambda c: abs(cv2.contourArea(c) - np.median(areas)))[:8]
        centers = []
        for cnt in valid:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        return centers

    def get_birds_eye_view(self, frame, centers):
        if len(centers) != 8:
            return None
        h, w = frame.shape[:2]
        scale, cx, cy = 100, w // 2, h // 2
        mm_pts = np.array([[-1,-3],[-1,3],[1,-3],[1,3],[-3,-1],[3,-1],[-3,1],[3,1]], dtype=np.float32)
        dst = np.array([[cx + p[0]*scale, cy + p[1]*scale] for p in mm_pts], dtype=np.float32)
        src = np.array(centers, dtype=np.float32)
        def by_angle(pts):
            c = pts.mean(axis=0)
            return pts[np.argsort(np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0]))]
        H, _ = cv2.findHomography(by_angle(src), by_angle(dst))
        return cv2.warpPerspective(frame, H, (w, h)) if H is not None else None

    def _get_angle(self, x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1))

    def _mean(self, segs):
        if not segs:
            return None, None
        ys = [s[1] for s in segs] + [s[3] for s in segs]
        xs = [s[0] for s in segs] + [s[2] for s in segs]
        return float(np.mean(ys)), float(np.mean(xs))

    def detectar_tablero_de_gato(self, frame):
        self.frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        k = np.ones((3, 3), np.uint8)
        binary = cv2.erode(binary, k, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
        lines = cv2.HoughLinesP(binary, 1, np.pi/180, threshold=200, minLineLength=50, maxLineGap=45)
        if lines is None:
            return frame
        alto, ancho = frame.shape[:2]
        horiz, vert = [], []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            ang = self._get_angle(x1, y1, x2, y2) % 180
            if ang < 20 or ang > 160:
                horiz.append((x1, y1, x2, y2))
            elif 60 < ang < 120:
                vert.append((x1, y1, x2, y2))
        y_top, _ = self._mean([s for s in horiz if s[1] < alto/2])
        y_bot, _ = self._mean([s for s in horiz if s[1] >= alto/2])
        _, x_lft = self._mean([s for s in vert if s[0] < ancho/2])
        _, x_rgt = self._mean([s for s in vert if s[0] >= ancho/2])
        green = (0, 255, 0)
        if y_top is not None: cv2.line(frame, (0, int(y_top)), (ancho, int(y_top)), green, 2)
        if y_bot is not None: cv2.line(frame, (0, int(y_bot)), (ancho, int(y_bot)), green, 2)
        if x_lft is not None: cv2.line(frame, (int(x_lft), 0), (int(x_lft), alto), green, 2)
        if x_rgt is not None: cv2.line(frame, (int(x_rgt), 0), (int(x_rgt), alto), green, 2)
        return frame

    def run_timed_preview(self, duration=10.0):
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 894, 600)
        except Exception:
            pass

        t0 = time.time()
        while True:
            elapsed = time.time() - t0
            remaining = int(duration - elapsed)
            if elapsed >= duration:
                break
            try:
                frame = self.cam.get_frame()
            except Exception:
                time.sleep(0.05)
                continue
            if frame is None:
                time.sleep(0.01)
                continue
            try:
                centers = self.calculate_red_dots(frame.copy())
                frame = self.detectar_tablero_de_gato(frame)
                bev = self.get_birds_eye_view(frame, centers)
                show = bev if bev is not None else frame
            except Exception:
                show = frame
            try:
                show = cv2.resize(show, (894, 600))
            except Exception:
                pass
            cv2.putText(show, f"{remaining}s  q=salir  s=foto",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.imshow(self.window_name, show)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                os.makedirs("snapshots", exist_ok=True)
                fname = time.strftime("snapshots/snap_%Y%m%d-%H%M%S.png")
                cv2.imwrite(fname, show)
                print(f"foto: {fname}")

        try:
            cv2.destroyWindow(self.window_name)
        except Exception:
            pass

    def start_preview(self):
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._preview_loop, daemon=True)
        self.thread.start()
        return self.thread, self.stop_event

    def _preview_loop(self):
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 894, 600)
        except Exception:
            pass
        try:
            while not self.stop_event.is_set():
                try:
                    frame = self.cam.get_frame()
                except Exception:
                    time.sleep(0.05)
                    continue
                if frame is None:
                    time.sleep(0.01)
                    continue
                try:
                    show = cv2.resize(frame, (894, 600))
                except Exception:
                    show = frame
                cv2.imshow(self.window_name, show)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("s"):
                    os.makedirs("snapshots", exist_ok=True)
                    fname = time.strftime("snapshots/snap_%Y%m%d-%H%M%S.png")
                    cv2.imwrite(fname, show)
                    print(f"foto: {fname}")
                elif key == ord("q"):
                    self.stop_event.set()
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass


class BoardVision:

    _SCALE = 100

    def __init__(self, cam):
        self.cam = cam
        self.baseline = None
        self.H = None

    def _red_dots(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lr = cv2.inRange(hsv, (0, 90, 50), (10, 255, 255))
        dr = cv2.inRange(hsv, (160, 90, 50), (180, 255, 255))
        mask = cv2.bitwise_or(lr, dr)
        _, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in contours if 200 <= cv2.contourArea(c) <= 5000]
        if len(valid) > 8:
            areas = np.array([cv2.contourArea(c) for c in valid])
            valid = sorted(valid, key=lambda c: abs(cv2.contourArea(c) - np.median(areas)))[:8]
        centers = []
        for cnt in valid:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            centers.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        return centers

    def _make_bev(self, frame, centers):
        if len(centers) != 8:
            return None
        h, w = frame.shape[:2]
        sc = self._SCALE
        cx, cy = w // 2, h // 2
        mm_pts = np.array([[-1,-3],[-1,3],[1,-3],[1,3],[-3,-1],[3,-1],[-3,1],[3,1]], dtype=np.float32)
        dst = np.array([[cx + p[0]*sc, cy + p[1]*sc] for p in mm_pts], dtype=np.float32)
        src = np.array(centers, dtype=np.float32)
        def by_angle(pts):
            c = pts.mean(axis=0)
            return pts[np.argsort(np.arctan2(pts[:,1]-c[1], pts[:,0]-c[0]))]
        H, _ = cv2.findHomography(by_angle(src), by_angle(dst))
        if H is not None:
            self.H = H
            return cv2.warpPerspective(frame, H, (w, h))
        return None

    def get_bev(self):
        for _ in range(8):
            try:
                frame = self.cam.get_frame()
            except Exception:
                time.sleep(0.05)
                continue
            if frame is None:
                time.sleep(0.02)
                continue
            if self.H is not None:
                h, w = frame.shape[:2]
                return cv2.warpPerspective(frame, self.H, (w, h))
            centers = self._red_dots(frame.copy())
            bev = self._make_bev(frame, centers)
            if bev is not None:
                return bev
        return None

    def cell_roi(self, cell, w, h, half=60):
        cx, cy = w // 2, h // 2
        row, col = divmod(cell, 3)
        off = 2 * self._SCALE
        ccx = cx + (col - 1) * off
        ccy = cy + (row - 1) * off
        return (max(0, ccx - half), max(0, ccy - half),
                min(w, ccx + half), min(h, ccy + half))

    def capture_baseline(self):
        print("calibrando...", end="", flush=True)
        frames = []
        t0 = time.time()
        while time.time() - t0 < 5.0 and len(frames) < 8:
            bev = self.get_bev()
            if bev is not None:
                frames.append(bev.astype(np.float32))
        if frames:
            self.baseline = np.median(frames, axis=0).astype(np.uint8)
            print(" ok")
        else:
            self.baseline = None
            print(" sin camara, usando teclado")

    def _cell_changed(self, bev, cell, threshold=0.07):
        if self.baseline is None:
            return False
        h, w = bev.shape[:2]
        x1, y1, x2, y2 = self.cell_roi(cell, w, h)
        roi = bev[y1:y2, x1:x2]
        base = self.baseline[y1:y2, x1:x2]
        if roi.size == 0 or base.size == 0:
            return False
        diff = cv2.absdiff(roi, base)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 18, 255, cv2.THRESH_BINARY)
        return float(np.sum(mask > 0)) / mask.size > threshold

    def wait_for_human_move(self, empty_cells):
        WIN = "tu turno"
        HOLD = 1.0

        try:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(WIN, 700, 560)
        except Exception:
            pass

        candidate = None
        candidate_t = 0.0
        last_bev = None

        while True:
            bev = self.get_bev()
            if bev is not None:
                last_bev = bev
            elif last_bev is not None:
                bev = last_bev
            else:
                time.sleep(0.05)
                continue

            h, w = bev.shape[:2]
            display = cv2.resize(bev.copy(), (700, 560))
            sx, sy = 700 / w, 560 / h

            changed = [i for i in empty_cells if self._cell_changed(bev, i)]
            now = time.time()

            for i in empty_cells:
                x1, y1, x2, y2 = self.cell_roi(i, w, h)
                dx1, dy1 = int(x1*sx), int(y1*sy)
                dx2, dy2 = int(x2*sx), int(y2*sy)
                color = (0, 110, 255) if i in changed else (0, 210, 0)
                cv2.rectangle(display, (dx1, dy1), (dx2, dy2), color, 2)
                cv2.putText(display, str(i + 1),
                            ((dx1 + dx2) // 2 - 8, (dy1 + dy2) // 2 + 9),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

            if changed:
                det = changed[0]
                if candidate == det:
                    elapsed = now - candidate_t
                    if elapsed >= HOLD:
                        try:
                            cv2.destroyWindow(WIN)
                        except Exception:
                            pass
                        return det
                    pct = int(elapsed / HOLD * 100)
                    cv2.putText(display, f"celda {det+1} {pct}%",
                                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 110, 255), 2)
                else:
                    candidate, candidate_t = det, now
            else:
                candidate = None
                cv2.putText(display, "coloca tu O",
                            (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 210, 0), 2)

            try:
                cv2.imshow(WIN, display)
            except Exception:
                pass
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                try:
                    cv2.destroyWindow(WIN)
                except Exception:
                    pass
                return -1


def main():
    robot = Meca500("192.168.0.100")
    robot.clean_errors()
    robot.activate()
    robot.home()
    robot.move_joints([0, 0, 0, 0, 0, 120])

    cam = None
    vision = None
    try:
        _c = MVCamera(ip_address="169.254.23.186")
        _c.set_gain(30)
        _c._set_exposure(38)
        _c.set_ROI(1440, 1180, 2034, 1824)
        cam = _c
        vision = BoardVision(cam)
        CameraManager(cam).run_timed_preview(duration=10.0)
    except Exception as e:
        print(f"camara: {e}")

    plotter = Plotter(robot)

    agent = Q_learning_agent()
    agent.train_agent(episodes=100_000)

    print("1) tu vs robot   2) robot vs robot   q) salir")
    while True:
        choice = input("opcion: ").strip().lower()
        if choice == "1":
            mode = "human_vs_robot"
            break
        if choice == "2":
            mode = "robot_vs_robot"
            break
        if choice in ("q", "quit", "salir"):
            robot.move_joints([0, 0, 0, 0, 0, 120])
            if cam is not None and hasattr(cam, "close"):
                try:
                    cam.close()
                except Exception:
                    pass
            return
        print("1, 2 o q")

    try:
        while True:
            active_vision = vision if mode == "human_vs_robot" else None
            TicTacToeGame(agent, plotter, mode=mode, vision=active_vision).play_game()
            ans = input("otra? (s / m=modo / q=salir): ").strip().lower()
            if ans == "m":
                print("1) tu vs robot   2) robot vs robot")
                while True:
                    choice = input("opcion: ").strip().lower()
                    if choice == "1":
                        mode = "human_vs_robot"
                        break
                    if choice == "2":
                        mode = "robot_vs_robot"
                        break
            elif ans not in ("s", "si", "y"):
                break
    finally:
        try:
            robot.clean_errors()
        except Exception:
            pass
        try:
            robot.move_joints([0, 0, 0, 0, 0, 120])
        except Exception:
            pass
        if cam is not None and hasattr(cam, "close"):
            try:
                cam.close()
            except Exception:
                pass


if __name__ == "__main__":
    main()
