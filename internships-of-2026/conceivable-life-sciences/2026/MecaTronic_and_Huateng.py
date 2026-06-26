import os
import threading
import time
from typing import List, Tuple, Union
import numpy as np

import cv2
from numpy import cos, pi, sin

from exploration_utils import Meca500, MVCamera


# ============================================================================
# CLASES DE CONTROL DEL ROBOT
# ============================================================================

class Plotter:
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

    def move_to(self, x: float, y: float) -> None:
        z_safe: float = self.z_draw + self.safe_height
        self.robot.move_pose([x, y, z_safe, *self.orientation])
        self.current_x = x
        self.current_y = y

    def pen_down(self) -> None:
        if self.pen_is_down:
            print("El bolígrafo ya está abajo.")
            return
        if self.current_x is None or self.current_y is None:
            raise RuntimeError("No se conoce la posición actual. Usa move_to primero.")
        self.robot.move_lin([self.current_x, self.current_y, self.z_draw, *self.orientation])
        self.pen_is_down = True
        print("Bolígrafo abajo.")

    def pen_up(self) -> None:
        if not self.pen_is_down:
            print("El bolígrafo ya está arriba.")
            return
        if self.current_x is None or self.current_y is None:
            raise RuntimeError("No se conoce la posición actual. Usa move_to primero.")
        self.robot.move_lin([self.current_x, self.current_y, self.z_draw + self.safe_height, *self.orientation])
        self.pen_is_down = False
        print("Bolígrafo arriba.")

    def draw_to(self, x: float, y: float) -> None:
        if not self.pen_is_down:
            raise RuntimeError("El bolígrafo está arriba. Usa pen_down() primero.")
        self.robot.move_lin([x, y, self.z_draw, *self.orientation])
        self.current_x = x
        self.current_y = y


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
    z_home: float = z_surface + 20.0
    robot.move_pose([x, y, z_home, 0, 90, 120])
    return z_surface


def draw_a_circle(robot: Meca500, radius: float, center: List[float], z: float, num_points: int) -> None:
    """Genera una trayectoria circular."""
    for i in range(num_points + 1):
        angle: float = 2 * pi * i / num_points
        x: float = center[0] + radius * cos(angle)
        y: float = center[1] + radius * sin(angle)
        robot.move_lin([x, y, z, 0, 90, 120])


def draw_something_arbitrary(robot: Meca500, points: List[Tuple[float, float]], z: float) -> None:
    """Sigue una lista arbitraria de puntos cartesianos."""
    for point in points:
        x, y = point
        robot.move_lin([x, y, z, 0, 90, 120])


# ============================================================================
# CLASE DE GESTIÓN Y VISIÓN DE CÁMARA
# ============================================================================

class CameraManager:
    def __init__(self, cam: MVCamera, window_name: str = "Microscopy IP Camera"):
        self.cam: MVCamera = cam
        self.window_name: str = window_name
        self.stop_event: threading.Event = threading.Event()
        self.thread: Union[threading.Thread, None] = None
        self.frame_count = 0
        self.line_counts = []

    def get_angle(self, x1, y1, x2, y2):
        return np.degrees(np.arctan2(y2 - y1, x2 - x1))

    def calculate_mean(self, segments):
        ys = []
        xs = []
        for seg in segments:
            ys.append([seg[1], seg[3]])
            xs.append([seg[0], seg[2]])
        return np.mean(ys), np.mean(xs)

    def detectar_tablero_contorno(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        clean_image = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=3)
        contours = cv2.findContours(clean_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def detectar_tablero_de_gato(self, frame):
        self.frame_count += 1

        # --- Preprocesamiento ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((3, 3), np.uint8)
        binary_image = cv2.erode(binary_image, kernel, iterations=3)
        binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

        # --- Detección de líneas ---
        lines = cv2.HoughLinesP(
            binary_image, 1, np.pi / 180,
            threshold=200, minLineLength=50, maxLineGap=45
        )

        if lines is None:
            return frame

        n_lines = len(lines)

        # --- Diagnóstico ---
        if 50 <= self.frame_count <= 100:
            self.line_counts.append(n_lines)
        if self.frame_count == 100:
            print("Promedio:", np.mean(self.line_counts))
            print("Min:", np.min(self.line_counts))
            print("Max:", np.max(self.line_counts))
        print("Lineas detectadas:", n_lines)

        # --- Clasificación por ángulo ---
        alto, ancho = frame.shape[:2]
        horizontales = []
        verticales = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = self.get_angle(x1, y1, x2, y2) % 180
            if angle < 10 or angle > 170:
                horizontales.append((x1, y1, x2, y2))
            elif 80 < angle < 100:
                verticales.append((x1, y1, x2, y2))

        # --- Separación arriba/abajo, izquierda/derecha ---
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

        # --- Promedios estables de cada línea ---
        y_arriba = None
        y_abajo = None
        x_izquierda = None
        x_derecha = None

        if len(linea_arriba) > 0:
            y_arriba, _ = self.calculate_mean(linea_arriba)
        if len(linea_abajo) > 0:
            y_abajo, _ = self.calculate_mean(linea_abajo)
        if len(linea_izquierda) > 0:
            _, x_izquierda = self.calculate_mean(linea_izquierda)
        if len(linea_derecha) > 0:
            _, x_derecha = self.calculate_mean(linea_derecha)

        # --- Dibujar sobre el frame (overlay) ---
        overlay = frame.copy()
        # Colores: horizontales -> verde, verticales -> rojo, endpoints -> azul
        color_h = (0, 255, 0)
        color_v = (0, 0, 255)
        color_ep = (255, 0, 0)

        # Dibujar líneas promedio completas (de borde a borde)
        try:
            if y_arriba is not None and not np.isnan(y_arriba):
                cv2.line(overlay, (0, int(y_arriba)), (ancho, int(y_arriba)), color_h, 2)
            if y_abajo is not None and not np.isnan(y_abajo):
                cv2.line(overlay, (0, int(y_abajo)), (ancho, int(y_abajo)), color_h, 2)
            if x_izquierda is not None and not np.isnan(x_izquierda):
                cv2.line(overlay, (int(x_izquierda), 0), (int(x_izquierda), alto), color_v, 2)
            if x_derecha is not None and not np.isnan(x_derecha):
                cv2.line(overlay, (int(x_derecha), 0), (int(x_derecha), alto), color_v, 2)
        except Exception:
            # Si hay valores inesperados, ignorar el dibujo de líneas
            pass

        # Dibujar endpoints de los segmentos detectados (para depuración visual)
        for seg in linea_arriba + linea_abajo + linea_izquierda + linea_derecha:
            try:
                x1, y1, x2, y2 = seg
                cv2.circle(overlay, (int(x1), int(y1)), 3, color_ep, -1)
                cv2.circle(overlay, (int(x2), int(y2)), 3, color_ep, -1)
            except Exception:
                continue

        # Mezclar overlay con el frame original para que se vea la superposición
        alpha = 0.9
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        return frame

    def _preview_thread(self) -> None:
        """Hilo interno para renderizar el streaming de la cámara."""
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

                # Procesar y dibujar overlay del tablero antes de mostrar
                try:
                    processed = self.detectar_tablero_de_gato(frame.copy())
                except Exception:
                    processed = frame

                try:
                    resized = cv2.resize(processed, (894, 600))
                except Exception:
                    resized = processed

                cv2.imshow(self.window_name, resized)
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
        """Arranca el hilo secundario de la cámara."""
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._preview_thread, daemon=True)
        self.thread.start()
        return self.thread, self.stop_event


# ============================================================================
# FLUJO PRINCIPAL
# ============================================================================
def main() -> None:
    robot = Meca500("192.168.0.100")
    cam = None
    camera_manager = None
    robot.move_joints([0, 0, 0, 0, 0, 120])
    try:
        ip = "169.254.23.186"
        cam = MVCamera(ip_address=ip)
        cam.set_gain(30)
        cam._set_exposure(28)
        cam.set_ROI(1236, 1340, 1398, 2008)

        camera_manager = CameraManager(cam, window_name="Camera Huateng")

        print("Preview de cámara iniciado. Pulsa 's' para snapshot, 'q' para cerrar.")
    except Exception as e:
        print(f"Advertencia: No se pudo iniciar la cámara: {e}")

    if cam is not None and camera_manager is not None:
        print("Abriendo previsualización de cámara. Presiona 'q' para iniciar la rutina del robot...")

        while True:
            frame = cam.get_frame()
            if frame is None:
                print("Error: No se puede recibir el fotograma.")
                break

            frame = camera_manager.detectar_tablero_de_gato(frame)
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

    print("Iniciando secuencia del Meca500...")
    robot.move_joints([0, 0, 0, 0, 0, 120])

    poses: List[List[float]] = [[130, 0, -130, 0, 0, 120]]

    for p in poses:
        try:
            robot.catch_errors()
            print("Probando:", p)
            robot.move_joints(p)
            print("OK")
        except Exception as e:
            print("ERROR:", e)

    print(robot._commands.GetJoints())

    drawing_tool_length = 100.0
    surface_position = [150.0, 0.0, 20.0]
    grip_point = 0.5

    z_surface: float = move_to_home(robot, drawing_tool_length, surface_position, grip_point)


if __name__ == "__main__":
    main()