import os
import threading
import time
from typing import List, Tuple, Union
from matplotlib import lines
from matplotlib.pyplot import gray
import numpy as np

import cv2
from numpy import cos, pi, sin, arctan2

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from exploration_uils import Meca500, MVCamera




class Plotter:
    # ==============================================y==============================
    # CLASES DE CONTROL DEL ROBOT
    # ============================================================================
    """Clase encargada de la gestión del dibujo y trayectorias del Meca500."""

    def __init__(
        self,
        robot: Meca500,
        z_draw: float,
        safe_height: float = 20.0,
        orientation: Tuple[float, float, float] = (0.0, 90.0, 120.0)
    ):
        self.robot: Meca500 = robot
        self.z_draw: float = z_draw
        self.safe_height: float = safe_height
        self.orientation: Tuple[float, float, float] = orientation
        self.current_x: Union[float, None] = None
        self.current_y: Union[float, None] = None
        self.pen_is_down: bool = False

    def move_to_home(
        robot: Meca500,
        drawing_tool_length: float,
        surface_position: List[float],
        grip_point: float
    ) -> float:
        """Mueve al robot a la posición de inicio sobre la superficie."""
        x: float = surface_position[0]
        y: float = surface_position[1]

        z_surface: float = surface_position[2] + drawing_tool_length * grip_point
        z_home: float = z_surface + 20.0  # 20 mm por encima de la superficie

        robot.move_pose([x, y, z_home, 0, 90, 120])
        return z_surface


class Q_learning_agent:
    def __init__(self):
        self.q_table = {}          # (estado, acción) → valor
        self.alpha = 0.5           # learning rate
        self.gamma = 0.9           # discount factor
        self.epsilon = 1.0         # exploración: empieza en 1, baja con tiempo

    def choose_action(self, state, available_actions):
        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        q_values = [self.q_table.get((state, a), 0) for a in available_actions]
        return available_actions[np.argmax(q_values)]

    def update_q_value(self, state, action, reward, next_state, available_actions):
        actual_q = self.q_table.get((state, action), 0)
        next_state_q_values = [self.q_table.get((next_state, a), 0) for a in available_actions]
        max_next_q = max(next_state_q_values)
        self.q_table[(state, action)] = actual_q + self.alpha * (reward + self.gamma * max_next_q - actual_q)

    def check_winner(self, board):
        winning_lines = [
            (0,1,2),
            (3,4,5),
            (6,7,8),
            (0,3,6),
            (1,4,7),
            (2,5,8),
            (0,4,8),
            (2,4,6)
        ]

        for a, b, c in winning_lines:
            if board[a] != 0 and board[a] == board[b] == board[c]:
                return board[a]

        return 0

    def train_agent(self, episodes, available_actions):
        for episode in range(episodes):
            board=(0,0,0,0,0,0,0,0,0)  # estado inicial del tablero
            done = False
            player1_turn= True

            while not done:
                zeros=[]
                for i, val in enumerate(board): 
                    if board[i]==0:
                        zeros.append(i)
                    else:
                        continue
                state = board
                if not player1_turn: 
                        next_state = tuple(0 if v==0 else (2 if v== 1 else 1) for v in board)
                action = self.choose_action(state, zeros)
                if player1_turn:
                    piece= 1
                else:
                    piece=2 
                board = list(board)
                board[action] = piece
                board = tuple(board)

                if piece == 1:
                    player1_moves += 1
                    moves = player1_moves
                else:
                    player2_moves += 1
                    moves = player2_moves
                winner = self.check_winner(board)
                reward = 0

                if winner == 1:

                    if moves == 3:
                        reward = 2.0

                    elif moves == 4:
                        reward = 1.5

                    else:
                        reward = 1.0

                    done = True
                if winner != 0:
                    done = True
                player1_turn = not player1_turn
                zeros=[]
                for i, val in enumerate(board): 
                    if board[i]==0:
                        zeros.append(i)
                    else:
                        continue
                if not zeros: 
                    done=True 
                next_actions = [i for i, v in enumerate(board) if v == 0]

                self.update_q_value(
                    state,
                    action,
                    reward,
                    next_state,
                    next_actions
                )
class CameraManager:
    # ============================================================================
    # CLASE DE GESTIÓN Y VISIÓN DE CÁMARA
    # ============================================================================
    def __init__(self, cam: MVCamera, window_name: str = "Microscopy IP Camera"):
        self.cam: MVCamera = cam
        self.window_name: str = window_name
        self.stop_event: threading.Event = threading.Event()
        self.thread: Union[threading.Thread, None] = None
        self.frame_count = 0
        self.line_counts = []

    def calculate_red_dots(self, frame):
        # BUG 6 fix: guard frame None aquí, no en hsv
        if frame is None:
            return []
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        light_red_mask = cv2.inRange(hsv, np.array([0, 90, 50]), np.array([10, 255, 255]))
        dark_red_mask = cv2.inRange(hsv, np.array([160, 90, 50]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(light_red_mask, dark_red_mask)
        _, thresh = cv2.threshold(red_mask, 127, 255, cv2.THRESH_BINARY)
        cleaned_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        MIN_DOT_AREA = 200
        MAX_DOT_AREA = 5000
        valid = [c for c in contours if MIN_DOT_AREA <= cv2.contourArea(c) <= MAX_DOT_AREA]

        # Si hay más de 8 (ruido), quedarse con los 8 más cercanos al área mediana
        if len(valid) > 8:
            areas_v = np.array([cv2.contourArea(c) for c in valid])
            median_area = np.median(areas_v)
            valid = sorted(valid, key=lambda c: abs(cv2.contourArea(c) - median_area))[:8]

        centers = []
        for cnt in valid:
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            centers.append((M['m10'] / M['m00'], M['m01'] / M['m00']))

        print(f"Puntos detectados: {len(centers)}")
        areas_all = sorted([cv2.contourArea(c) for c in contours], reverse=True)
        print("Top 15 áreas:", areas_all[:15])
        return centers

    def get_birds_eye_view(self, frame, centers):
        if len(centers) != 8:
            return None
        h, w = frame.shape[:2]
        scale = 100  # px por mm
        cx, cy = w // 2, h // 2

        # Posiciones físicas conocidas de los 8 puntos en mm
        mm_pts = np.array([[-1,-3], [-1,3], [1,-3], [1,3],[-3,-1], [3,-1], [-3,1], [3,1]], dtype=np.float32)

        # Convertir mm → píxeles de destino
        dst_pts_raw = np.array([[cx + p[0]*scale, cy + p[1]*scale] for p in mm_pts],dtype=np.float32)

        src_pts = np.array(centers, dtype=np.float32)

        def sort_by_angle(pts):
            centroid = pts.mean(axis=0)
            angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
            return pts[np.argsort(angles)]

        src_sorted = sort_by_angle(src_pts)
        dst_sorted = sort_by_angle(dst_pts_raw)

        H, _ = cv2.findHomography(src_sorted, dst_sorted)

        # BUG 2 fix: findHomography puede regresar None si los puntos son degenerados
        if H is None:
            return None

        return cv2.warpPerspective(frame, H, (w, h))

    def get_angle(self, x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1))

    def calculate_mean(self, segments):
        if not segments:
            return None, None
        ys = []
        xs = []
        for seg in segments:
            ys.append([seg[1], seg[3]])
            xs.append([seg[0], seg[2]])
        return float(np.mean(ys)), float(np.mean(xs))

    def detectar_tablero_de_gato(self, frame):
        self.frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.erode(binary_image, kernel, iterations=3)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        lines = cv2.HoughLinesP(binary_image, 1, np.pi / 180, threshold=200, minLineLength=50, maxLineGap=45)

        if lines is None:
            return frame

        n_lines = len(lines)

        if 50 <= self.frame_count <= 100:
            self.line_counts.append(n_lines)
        if self.frame_count == 100:
            print("Promedio:", np.mean(self.line_counts))
            print("Min:", np.min(self.line_counts))
            print("Max:", np.max(self.line_counts))
        print("Lineas detectadas:", n_lines)

        alto, ancho = frame.shape[:2]
        horizontales = []
        verticales = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = self.get_angle(x1, y1, x2, y2) % 180
            if angle < 20 or angle > 160:
                horizontales.append((x1, y1, x2, y2))
            elif 60 < angle < 120:
                verticales.append((x1, y1, x2, y2))

        linea_arriba = []
        linea_abajo = []
        for seg in horizontales:
            x1, y1, x2, y2 = seg
            if y1 < alto / 2:
                linea_arriba.append(seg)
            else:
                linea_abajo.append(seg)

        linea_izquierda = []
        linea_derecha = []
        for seg in verticales:
            x1, y1, x2, y2 = seg
            if x1 < ancho / 2:
                linea_izquierda.append(seg)
            else:
                linea_derecha.append(seg)

        y_arriba, _ = self.calculate_mean(linea_arriba)
        y_abajo, _ = self.calculate_mean(linea_abajo)
        _, x_izquierda = self.calculate_mean(linea_izquierda)
        _, x_derecha = self.calculate_mean(linea_derecha)

        if y_arriba is not None:
            cv2.line(frame, (0, int(y_arriba)), (ancho, int(y_arriba)), (0, 255, 0), 2)
        if y_abajo is not None:
            cv2.line(frame, (0, int(y_abajo)), (ancho, int(y_abajo)), (0, 255, 0), 2)
        if x_izquierda is not None:
            cv2.line(frame, (int(x_izquierda), 0), (int(x_izquierda), alto), (0, 255, 0), 2)
        if x_derecha is not None:
            cv2.line(frame, (int(x_derecha), 0), (int(x_derecha), alto), (0, 255, 0), 2)
        return frame

    def _preview_thread(self) -> None:
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        try:
            try:
                cv2.resizeWindow(self.window_name, 894, 600)
            except Exception:
                pass

            while not self.stop_event.is_set():
                try:
                    frame = self.cam.get_frame()
                except Exception as e:
                    print(f"Preview: error obteniendo frame: {e}")
                    time.sleep(0.05)
                    continue

                if frame is None:
                    time.sleep(0.01)
                    continue

                if frame.ndim == 2:
                    show = frame
                elif frame.shape[2] == 3:
                    show = frame
                else:
                    show = frame
                # BUG 7 fix: mostrar show redimensionado, no frame (que nunca se usaba)
                show = cv2.resize(show, (894, 600))
                cv2.imshow(self.window_name, show)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s'):
                    if not os.path.exists("snapshots"):
                        os.makedirs("snapshots")
                    fname = time.strftime("snapshots/snapshot_%Y%m%d-%H%M%S.png")
                    try:
                        cv2.imwrite(fname, show)
                        print(f"Snapshot guardado: {fname}")
                    except Exception as e:
                        print(f"Error guardando snapshot: {e}")
                elif key == ord('q'):
                    self.stop_event.set()
                    break
        finally:
            try:
                cv2.destroyWindow(self.window_name)
            except Exception:
                pass

    def start_preview(self) -> Tuple[threading.Thread, threading.Event]:
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._preview_thread,
            daemon=True
        )
        self.thread.start()
        return self.thread, self.stop_event


def main() -> None:
    # ============================================================================
    # FLUJO PRINCIPAL
    # ============================================================================
    robot = Meca500("192.168.0.100")
    cam = None
    camera_manager = None
    robot.clean_errors()
    robot.activate()
    robot.home()
    robot.move_joints([0, 0, 0, 0, 0, 120])
    try:
        ip = "169.254.23.186"
        cam = MVCamera(ip_address=ip)
        cam.set_gain(30)
        cam._set_exposure(38)
        cam.set_ROI(1440, 1180, 2034, 1824)
        camera_manager = CameraManager(cam, window_name="Camera Huateng")
        print("Preview de cámara iniciado. Pulsa 's' para snapshot, 'q' para cerrar.")
    except Exception as e:
        print(f"Advertencia: No se pudo iniciar la cámara: {e}")

    # --- FASE 1: VISUALIZACIÓN PREVIA DE LA CÁMARA ---
    if cam is not None and camera_manager is not None:
        print("Abriendo previsualización de cámara. Presiona 'q' para iniciar la rutina del robot...")

        while True:
            frame = cam.get_frame()
            # BUG 1 fix: un frame None no debe matar el loop — continue, no break
            if frame is None:
                print("Warning: frame None, reintentando...")
                continue

            try:
                # BUG 4 fix: detectar puntos ANTES de dibujar líneas sobre el frame
                centers = camera_manager.calculate_red_dots(frame)
                frame = camera_manager.detectar_tablero_de_gato(frame)
                birds_eye_view = camera_manager.get_birds_eye_view(frame, centers)
                cv2.imshow("Camera - Previsualizacion", birds_eye_view if birds_eye_view is not None else frame)
            except Exception as e:
                print(f"Error en pipeline de visión: {e}")
                cv2.imshow("Camera - Previsualizacion", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()

        try:
            if hasattr(cam, 'close'):
                cam.close()
            elif hasattr(cam, 'release'):
                cam.close()
            print("Recurso de la cámara liberado exitosamente.")
        except Exception as ex:
            print(f"No se pudo cerrar la cámara formalmente: {ex}")
    else:
        print("Saltando fase de cámara debido a un error de inicialización.")

    # --- FASE 2: MOVIMIENTOS Y RUTINA DEL ROBOT MECA500 ---
    print("Iniciando secuencia del Meca500...")
    robot.move_joints([0, 0, 0, 0, 0, 120])
    drawing_tool_length = 100.0
    surface_position = [150.0, 0.0, 20.0]
    grip_point = 0.5

    z_surface: float = Plotter.move_to_home(
        robot,
        drawing_tool_length,
        surface_position,
        grip_point
    )


if __name__ == "__main__":
    main()