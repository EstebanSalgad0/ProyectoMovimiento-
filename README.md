# Proyecto de Feedback de Pose para Fútbol

Este proyecto utiliza MediaPipe Pose Landmarker para detectar la pose del cuerpo y proporcionar feedback en tiempo real sobre la postura, enfocándose en el tren superior (tronco, hombros, codos).

## Requisitos

- Python 3.10+
- Webcam
- Bibliotecas: mediapipe, opencv-python, numpy

## Instalación

1. Clona el repositorio:
   ```
   git clone https://github.com/EstebanSalgad0/ProyectoMovimiento-.git
   cd ProyectoMovimiento-
   ```

2. Crea un entorno virtual:
   ```
   python -m venv venv
   venv\Scripts\activate  # En Windows
   ```

3. Instala las dependencias:
   ```
   pip install mediapipe opencv-python numpy
   ```

## Uso

Ejecuta el script:
```
python pose_feedback_webcam.py
```

- Se abrirá una ventana con la webcam y el esqueleto dibujado.
- Las métricas aparecen en la esquina superior izquierda.
- El feedback en la esquina inferior derecha.
- Presiona ESC para salir.

## Funcionalidades

- Detección de pose con MediaPipe.
- Cálculo de ángulos para tronco, codos y hombros.
- Feedback en tiempo real para corrección postural.

## Notas

- Si hay problemas con la webcam, usa un archivo de video MP4 cambiando `cap = cv2.VideoCapture("video.mp4")`.
- El modelo lite es más rápido, el full más preciso.