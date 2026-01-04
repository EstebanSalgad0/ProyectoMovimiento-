import time
import math
import cv2
import numpy as np
import mediapipe as mp

MODEL_PATH = "pose_landmarker_lite.task"

# ---------- Utilidades matemáticas ----------
def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Ángulo (grados) entre dos vectores."""
    v1n = v1 / (np.linalg.norm(v1) + 1e-9)
    v2n = v2 / (np.linalg.norm(v2) + 1e-9)
    dot = float(np.clip(np.dot(v1n, v2n), -1.0, 1.0))
    return math.degrees(math.acos(dot))

def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Ángulo (grados) en la articulación b, formado por a-b-c."""
    ba = a - b
    bc = c - b
    return angle_between(ba, bc)

def lm_to_np(lm) -> np.ndarray:
    """Landmark normalizado (x,y,z) a numpy."""
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)

def midpoint(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    return (p1 + p2) / 2.0

# ---------- MediaPipe Tasks ----------
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_poses=1
)

# Índices de landmarks (MediaPipe Pose: 33 puntos)
# 11: left_shoulder, 12: right_shoulder
# 23: left_hip, 24: right_hip
# 25: left_knee, 26: right_knee
# 27: left_ankle, 28: right_ankle
# 13: left_elbow, 14: right_elbow
# 15: left_wrist, 16: right_wrist
L_SHO, R_SHO = 11, 12
L_HIP, R_HIP = 23, 24
L_KNE, R_KNE = 25, 26
L_ANK, R_ANK = 27, 28
L_ELB, R_ELB = 13, 14
L_WRI, R_WRI = 15, 16

# Conexiones para dibujar el esqueleto
POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (25, 26), (27, 28), (29, 30), (31, 32),
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)
]

# Umbrales (ajustables)
TRUNK_TILT_DEG_THRESHOLD = 12.0      # tronco inclinado (lateral) sobre ~12°
KNEE_FLEXION_THRESHOLD = 160.0       # rodilla < 160° => flexión notoria
SHOULDER_ASYM_Y_THRESHOLD = 0.03     # diferencia en y normalizada (~3% altura)

def put_lines(frame, lines, x=20, y0=30, dy=28):
    y = y0
    for s in lines:
        cv2.putText(frame, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        y += dy

start_time = time.time()

# Paso 3: Reabrir la cámara después de crear el Landmarker
cap_temp = cv2.VideoCapture(0)
cap_temp.release()

with PoseLandmarker.create_from_options(options) as landmarker:
    cap = cv2.VideoCapture(0)  # Reabrir
    if not cap.isOpened():
        raise RuntimeError("No se pudo abrir la cámara")

    # Opcional: setear propiedades antes de leer
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    cv2.namedWindow("Pose Feedback (MediaPipe)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Pose Feedback (MediaPipe)", 1280, 720)
    while cap.isOpened():
        ok, frame_bgr = cap.read()
        if not ok:
            print("cap.read() falló")
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        timestamp_ms = int((time.time() - start_time) * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        feedback = []
        metrics = []

        # Dibujar pose si existe
        if result.pose_landmarks:
            pose_lms = result.pose_landmarks[0]

            # Dibujo del esqueleto
            height, width = frame_bgr.shape[:2]
            for start_idx, end_idx in POSE_CONNECTIONS:
                if start_idx < len(pose_lms) and end_idx < len(pose_lms):
                    start_lm = pose_lms[start_idx]
                    end_lm = pose_lms[end_idx]
                    start_point = (int(start_lm.x * width), int(start_lm.y * height))
                    end_point = (int(end_lm.x * width), int(end_lm.y * height))
                    cv2.line(frame_bgr, start_point, end_point, (0, 255, 0), 2)

            # ---- Extraer puntos relevantes ----
            l_sho = lm_to_np(pose_lms[L_SHO])
            r_sho = lm_to_np(pose_lms[R_SHO])
            l_hip = lm_to_np(pose_lms[L_HIP])
            r_hip = lm_to_np(pose_lms[R_HIP])
            l_elb = lm_to_np(pose_lms[L_ELB])
            r_elb = lm_to_np(pose_lms[R_ELB])
            l_wri = lm_to_np(pose_lms[L_WRI])
            r_wri = lm_to_np(pose_lms[R_WRI])

            # ---- 1) Inclinación del tronco (lateral) ----
            sho_mid = midpoint(l_sho, r_sho)
            hip_mid = midpoint(l_hip, r_hip)

            torso_vec = sho_mid - hip_mid  # vector cadera->hombros
            # Eje vertical en coordenadas de imagen normalizada: y crece hacia abajo,
            # por lo que "arriba" es (0, -1).
            vertical = np.array([0.0, -1.0, 0.0], dtype=np.float32)

            trunk_angle = angle_between(torso_vec, vertical)
            metrics.append(f"Tronco inclinado: {trunk_angle:.1f}°")

            if trunk_angle > TRUNK_TILT_DEG_THRESHOLD:
                # Direccionalidad aproximada (lateral) con componente x
                if torso_vec[0] > 0:
                    feedback.append("Enderece el tronco (inclinacion hacia la derecha).")
                else:
                    feedback.append("Enderece el tronco (inclinacion hacia la izquierda).")
            else:
                feedback.append("Tronco: estable.")

            # ---- 2) Ángulo de codos ----
            left_elbow_angle = joint_angle(l_sho, l_elb, l_wri)
            right_elbow_angle = joint_angle(r_sho, r_elb, r_wri)
            metrics.append(f"Codo izq: {left_elbow_angle:.1f}°")
            metrics.append(f"Codo der: {right_elbow_angle:.1f}°")

            elbow_threshold = 160.0  # umbral para flexión
            if left_elbow_angle < elbow_threshold:
                feedback.append("Codo izquierdo flexionado.")
            else:
                feedback.append("Codo izquierdo extendido.")

            if right_elbow_angle < elbow_threshold:
                feedback.append("Codo derecho flexionado.")
            else:
                feedback.append("Codo derecho extendido.")

            # ---- 3) Asimetría de hombros (altura) ----
            shoulder_y_diff = float(abs(l_sho[1] - r_sho[1]))  # y normalizada
            metrics.append(f"Asimetria hombros (Dy): {shoulder_y_diff:.3f}")

            if shoulder_y_diff > SHOULDER_ASYM_Y_THRESHOLD:
                # En coordenadas de imagen, menor y = más alto
                if l_sho[1] < r_sho[1]:
                    feedback.append("Hombro izquierdo mas alto; nivele los hombros.")
                else:
                    feedback.append("Hombro derecho mas alto; nivele los hombros.")
            else:
                feedback.append("Hombros: alineados.")

        else:
            feedback.append("No se detecta la pose. Acerque el cuerpo a camara y mejore la iluminacion.")

        # Mostrar métricas + feedback en pantalla
        height, width = frame_bgr.shape[:2]
        put_lines(frame_bgr, ["MÉTRICAS"] + metrics, x=20, y0=30)
        put_lines(frame_bgr, ["FEEDBACK"] + feedback, x=width-400, y0=height-250)

        cv2.imshow("Pose Feedback (MediaPipe)", frame_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()