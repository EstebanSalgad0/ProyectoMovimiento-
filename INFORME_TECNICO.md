# Informe tecnico del proyecto

## 1. Identificacion del proyecto

**Nombre del proyecto:** ProyectoMovimiento-

**Tipo de proyecto:** Aplicacion de vision por computador para feedback de postura y movimiento en tiempo real.

**Archivo principal:** `pose_feedback_webcam.py`

**Objetivo general:** detectar la pose corporal de una persona usando una webcam y entregar retroalimentacion visual inmediata sobre la ejecucion de ejercicios, principalmente sentadilla, zancada, curl de biceps sentado y press de hombros sentado.

## 2. Resumen ejecutivo

El proyecto implementa un sistema de analisis de movimiento corporal en tiempo real. Para lograrlo se usa la camara del computador, OpenCV para capturar y mostrar video, MediaPipe Pose Landmarker para detectar puntos corporales, y NumPy para calcular angulos y diferencias entre articulaciones.

La aplicacion muestra la imagen de la webcam con un esqueleto dibujado sobre el cuerpo detectado. Segun el modo seleccionado, el sistema enfoca el analisis en el tren inferior o superior, calcula metricas relevantes y muestra recomendaciones tecnicas para corregir la postura o confirmar una ejecucion adecuada.

## 3. Que se hizo

Se desarrollo un script de Python capaz de:

1. Abrir la webcam y seleccionar automaticamente una camara funcional.
2. Cargar un modelo de deteccion de pose de MediaPipe.
3. Procesar cada frame de video en tiempo real.
4. Detectar landmarks corporales normalizados.
5. Dibujar conexiones del cuerpo segun el ejercicio activo.
6. Calcular angulos articulares de rodillas y codos.
7. Evaluar reglas tecnicas por ejercicio.
8. Mostrar metricas y feedback dentro de la ventana de video.
9. Permitir cambiar de ejercicio con teclas numericas.
10. Entregar instrucciones de instalacion y uso en el README.
11. Incluir modelos localmente para que el sistema pueda ejecutarse sin descargar el modelo al iniciar.

## 4. Como se hizo

### 4.1 Captura de video

La captura se realiza con OpenCV mediante `cv2.VideoCapture`. El programa intenta abrir primero el indice de camara definido por la variable de entorno `CAMERA_INDEX`. Si esta variable no existe o no es valida, usa por defecto el indice `1`.

Ademas, se implemento una funcion de fallback llamada `open_camera_with_fallback`, que escanea indices desde `0` hasta `10` para encontrar una camara disponible. En Windows se intenta primero el backend `cv2.CAP_DSHOW` y luego `cv2.CAP_ANY`.

Para validar que una camara funciona, el programa no se conforma con que la camara se abra, sino que intenta leer varios frames y verifica que sean imagenes validas.

### 4.2 Deteccion de pose

La deteccion corporal se realiza con MediaPipe Tasks Vision:

- `PoseLandmarker`
- `PoseLandmarkerOptions`
- `RunningMode.VIDEO`
- `BaseOptions`

El modelo usado por defecto en el codigo es:

```text
pose_landmarker_full.task
```

Tambien se incluye:

```text
pose_landmarker_lite.task
```

El modelo `full` prioriza mayor precision, mientras que el modelo `lite` puede ser usado cuando se necesita mejor rendimiento.

Cada frame de OpenCV se convierte de BGR a RGB, luego se transforma en un objeto `mp.Image` con formato `SRGB`. Despues se ejecuta `detect_for_video`, usando un timestamp en milisegundos para respetar el modo de procesamiento de video.

### 4.3 Conversion de landmarks

MediaPipe entrega landmarks corporales con coordenadas normalizadas `x`, `y`, `z`. El proyecto convierte esos puntos a arreglos de NumPy para facilitar los calculos matematicos.

La funcion principal para esto es:

```python
def lm_to_np(lm) -> np.ndarray:
    return np.array([lm.x, lm.y, lm.z], dtype=np.float32)
```

Los landmarks principales utilizados son:

| Punto | Indice MediaPipe | Uso |
| --- | ---: | --- |
| Hombro izquierdo | 11 | Tren superior y postura |
| Hombro derecho | 12 | Tren superior y postura |
| Codo izquierdo | 13 | Curl y press |
| Codo derecho | 14 | Curl y press |
| Muneca izquierda | 15 | Curl y press |
| Muneca derecha | 16 | Curl y press |
| Cadera izquierda | 23 | Sentadilla y zancada |
| Cadera derecha | 24 | Sentadilla y zancada |
| Rodilla izquierda | 25 | Sentadilla y zancada |
| Rodilla derecha | 26 | Sentadilla y zancada |
| Tobillo izquierdo | 27 | Sentadilla y zancada |
| Tobillo derecho | 28 | Sentadilla y zancada |

### 4.4 Calculo de angulos

Para evaluar la tecnica se calcularon angulos articulares. El proyecto usa dos funciones principales:

- `angle_between`: calcula el angulo entre dos vectores.
- `joint_angle`: calcula el angulo en una articulacion usando tres puntos.

La logica es:

1. Se toman tres puntos: anterior, articulacion central y posterior.
2. Se forman dos vectores desde la articulacion central.
3. Se normalizan los vectores.
4. Se calcula el producto punto.
5. Se usa `acos` para convertir el resultado en un angulo en grados.

Ejemplo conceptual:

```text
cadera - rodilla - tobillo = angulo de rodilla
hombro - codo - muneca = angulo de codo
```

## 5. Tecnologias utilizadas

| Tecnologia | Uso en el proyecto |
| --- | --- |
| Python | Lenguaje principal de desarrollo |
| OpenCV | Captura de webcam, dibujo en pantalla, ventana de visualizacion |
| MediaPipe | Deteccion de pose corporal mediante modelos Pose Landmarker |
| NumPy | Operaciones vectoriales, calculo de angulos y diferencias |
| Math | Conversion trigonometrica y calculos de angulos |
| OS | Lectura de variable de entorno `CAMERA_INDEX` |
| Time | Generacion de timestamps para el procesamiento de video |
| Git | Control de versiones del proyecto |
| VS Code | Tareas configuradas para ejecutar el script desde el editor |

## 6. Estructura del proyecto

```text
ProyectoMovimiento-/
├── .gitignore
├── .vscode/
│   └── tasks.json
├── README.md
├── pose_feedback_webcam.py
├── pose_landmarker_full.task
└── pose_landmarker_lite.task
```

### Descripcion de archivos

| Archivo | Descripcion |
| --- | --- |
| `pose_feedback_webcam.py` | Script principal. Contiene captura de video, deteccion de pose, calculos, reglas de feedback y visualizacion. |
| `pose_landmarker_full.task` | Modelo MediaPipe de mayor precision usado por defecto. |
| `pose_landmarker_lite.task` | Modelo MediaPipe mas liviano, util para mejorar rendimiento. |
| `README.md` | Documentacion de instalacion, uso, pruebas y solucion de problemas. |
| `.gitignore` | Evita versionar entornos virtuales, cache de Python y archivos temporales. |
| `.vscode/tasks.json` | Tareas para ejecutar el script desde Visual Studio Code. |

## 7. Funcionamiento general del sistema

El flujo de ejecucion es el siguiente:

1. Se definen constantes, umbrales y modos de ejercicio.
2. Se configura MediaPipe Pose Landmarker.
3. Se abre la webcam con busqueda automatica de camara disponible.
4. Se configura la resolucion sugerida de captura en `1280x720`.
5. Se crea una ventana redimensionable de OpenCV.
6. En cada iteracion se captura un frame.
7. El frame se convierte a RGB.
8. Se ejecuta la deteccion de pose.
9. Si existe una pose detectada, se extraen landmarks.
10. Se calculan angulos y metricas.
11. Se aplica la logica del ejercicio seleccionado.
12. Se dibujan esqueleto, landmarks destacados, metricas y feedback.
13. Se leen teclas para cambiar de modo o salir.
14. Al cerrar, se libera la camara y se destruyen las ventanas.

## 8. Modos de ejercicio implementados

### 8.1 Sentadilla

Modo activado con la tecla `1`.

Puntos usados:

- Caderas: 23 y 24
- Rodillas: 25 y 26
- Tobillos: 27 y 28

Metricas calculadas:

- Angulo de rodilla izquierda.
- Angulo de rodilla derecha.
- Promedio de angulos de rodilla.
- Diferencia entre ambas rodillas.
- Diferencia vertical entre caderas.
- Alineacion horizontal entre rodilla y tobillo.

Feedback entregado:

- Nivelacion de pelvis.
- Profundidad de sentadilla.
- Fase del movimiento: arriba, abajo o transicion.
- Distribucion del peso entre piernas.
- Alineacion rodilla-pie.

Umbrales principales:

| Umbral | Valor | Significado |
| --- | ---: | --- |
| `SQUAT_DOWN_ANGLE` | 115 grados | Bajo este valor se considera fase abajo. |
| `SQUAT_UP_ANGLE` | 160 grados | Sobre este valor se considera fase arriba. |
| `SQUAT_KNEE_ASYM_THRESHOLD` | 14 grados | Diferencia maxima aceptada entre rodillas. |
| `SQUAT_KNEE_TRACK_THRESHOLD` | 0.10 | Desviacion horizontal maxima rodilla-tobillo. |

### 8.2 Zancada

Modo activado con la tecla `2`.

Puntos usados:

- Caderas: 23 y 24
- Rodillas: 25 y 26
- Tobillos: 27 y 28

Metricas calculadas:

- Pierna delantera estimada.
- Angulo de rodilla delantera.
- Angulo de rodilla trasera.
- Separacion horizontal entre tobillos.
- Alineacion rodilla-tobillo de la pierna delantera.

La pierna delantera se estima como la pierna con mayor flexion, es decir, la que tiene menor angulo de rodilla.

Feedback entregado:

- Flexionar mas la rodilla delantera.
- Extender mas la rodilla trasera.
- Aumentar separacion entre pies.
- Mejorar alineacion rodilla-tobillo.
- Confirmar ejecucion estable.

Umbrales principales:

| Umbral | Valor | Significado |
| --- | ---: | --- |
| `LUNGE_FRONT_KNEE_MAX` | 125 grados | Maximo para considerar buena flexion delantera. |
| `LUNGE_REAR_KNEE_MIN` | 130 grados | Minimo para considerar suficiente extension trasera. |
| `LUNGE_STEP_WIDTH_MIN` | 0.10 | Separacion minima entre tobillos. |
| `LUNGE_KNEE_TRACK_THRESHOLD` | 0.12 | Desviacion maxima rodilla-tobillo. |

### 8.3 Curl de biceps sentado

Modo activado con la tecla `3`.

Puntos usados:

- Hombros: 11 y 12
- Codos: 13 y 14
- Munecas: 15 y 16

Metricas calculadas:

- Angulo de codo izquierdo.
- Angulo de codo derecho.
- Promedio de angulos de codo.
- Asimetria entre codos.
- Deriva horizontal del codo respecto al hombro.
- Desviacion horizontal de muneca respecto al codo.

Feedback entregado:

- Fase del curl: arriba, abajo o media.
- Necesidad de flexionar mas.
- Control del cierre del codo.
- Simetria entre ambos brazos.
- Mantener codos cerca del torso.
- Evitar desviacion lateral excesiva de munecas.

Umbrales principales:

| Umbral | Valor | Significado |
| --- | ---: | --- |
| `CURL_FLEXED_MAX` | 75 grados | Bajo este valor se considera fase arriba. |
| `CURL_EXTENDED_MIN` | 150 grados | Sobre este valor se considera fase abajo. |
| `CURL_SYMMETRY_THRESHOLD` | 18 grados | Diferencia maxima entre codos. |
| `CURL_ELBOW_DRIFT_THRESHOLD` | 0.12 | Deriva maxima de codo respecto al hombro. |

### 8.4 Press de hombros sentado

Modo activado con la tecla `4`.

Puntos usados:

- Hombros: 11 y 12
- Codos: 13 y 14
- Munecas: 15 y 16

Metricas calculadas:

- Angulo de codo izquierdo.
- Angulo de codo derecho.
- Promedio de angulos de codo.
- Diferencia entre codos.
- Altura de muneca respecto al hombro.
- Diferencia vertical entre hombros.

Feedback entregado:

- Fase del press: arriba, abajo o intermedia.
- Necesidad de extender mas en la parte alta.
- Simetria entre brazos.
- Nivelacion de hombros durante el empuje.

Umbrales principales:

| Umbral | Valor | Significado |
| --- | ---: | --- |
| `PRESS_UP_ELBOW_MIN` | 155 grados | Minimo para considerar codo extendido arriba. |
| `PRESS_DOWN_ELBOW_MAX` | 95 grados | Maximo para considerar fase baja. |
| `PRESS_WRIST_ABOVE_SHOULDER` | 0.02 | Muneca claramente sobre hombro. |
| `PRESS_WRIST_NEAR_SHOULDER` | 0.10 | Muneca cercana al nivel del hombro. |
| `PRESS_SYMMETRY_THRESHOLD` | 16 grados | Diferencia maxima entre codos. |

## 9. Interfaz visual

La interfaz se construyo directamente sobre el frame de video usando OpenCV. No se implemento una interfaz web ni una aplicacion de escritorio separada, sino una ventana de OpenCV llamada:

```text
Pose Feedback (MediaPipe)
```

Elementos visuales:

- Video de la webcam en tiempo real.
- Lineas verdes para representar el esqueleto relevante.
- Puntos destacados para las articulaciones principales.
- Etiquetas de landmarks como hombro, codo, muneca, cadera, rodilla y tobillo.
- Panel de metricas.
- Panel de feedback.

Para mejorar la legibilidad se implementaron funciones de texto:

- `put_lines`: escribe lineas simples.
- `wrap_text_to_width`: divide textos largos segun ancho disponible.
- `draw_text_panel`: dibuja paneles semitransparentes con borde y texto.

Esto evita que los mensajes se corten o se mezclen con la imagen.

## 10. Controles de usuario

| Tecla | Accion |
| --- | --- |
| `1` | Cambiar a modo sentadilla. |
| `2` | Cambiar a modo zancada. |
| `3` | Cambiar a modo curl de biceps sentado. |
| `4` | Cambiar a modo press de hombros sentado. |
| `ESC` | Cerrar la aplicacion. |
| X de la ventana | Cerrar la aplicacion. |

## 11. Robustez y manejo de errores

El proyecto incluye varias medidas para evitar fallos comunes:

- Escaneo automatico de camaras disponibles.
- Validacion de frames antes de usar la camara.
- Mensaje claro si no se encuentra ninguna camara.
- Manejo de errores de lectura de camara con `try/except`.
- Deteccion de cierre por teclado con `KeyboardInterrupt`.
- Cierre limpio liberando la camara y destruyendo ventanas.
- Mensaje de feedback cuando no se detecta ninguna pose.

## 12. Instalacion y ejecucion

El README documenta los pasos principales:

1. Clonar el repositorio.
2. Crear entorno virtual.
3. Activar entorno virtual.
4. Instalar dependencias.
5. Ejecutar el script.

Comando de instalacion de dependencias:

```bash
pip install mediapipe opencv-python numpy
```

Comando de ejecucion:

```bash
python pose_feedback_webcam.py
```

Si se necesita forzar una camara especifica, se puede usar:

```powershell
$env:CAMERA_INDEX=0
python pose_feedback_webcam.py
```

## 13. Control de versiones

El repositorio usa Git. Segun el historial local, el proyecto tiene los siguientes hitos:

| Commit | Descripcion |
| --- | --- |
| `05592a6` | Version inicial con script de feedback de pose usando MediaPipe. |
| `f029006` | Actualizacion del README con instalacion paso a paso y solucion de problemas. |
| `bdb35cf` | Mejora del feedback por ejercicio y robustez de camara. |

Esto muestra una evolucion desde una primera version funcional hacia una version mejor documentada, con mas ejercicios y mejor manejo de camara.

## 14. Criterios tecnicos aplicados

El proyecto se basa en reglas biomecanicas simples y umbrales ajustables. No entrena un modelo propio, sino que usa MediaPipe para detectar la pose y luego aplica logica propia para interpretar el movimiento.

Los criterios usados son:

- Angulos articulares para medir flexion o extension.
- Diferencias entre lado izquierdo y derecho para detectar asimetrias.
- Distancias normalizadas para evaluar alineacion.
- Comparaciones verticales para evaluar altura de hombros, caderas o munecas.
- Fases de movimiento segun rangos de angulo.

Este enfoque es adecuado para un prototipo educativo o de demostracion, ya que permite explicar claramente por que se entrega cada recomendacion.

## 15. Limitaciones actuales

El sistema funciona como prototipo, pero tiene algunas limitaciones:

- Depende de buena iluminacion y de que el cuerpo sea visible para la camara.
- Los umbrales son generales y no estan calibrados por usuario.
- No registra historial ni repeticiones acumuladas.
- No guarda videos ni resultados.
- No usa profundidad real; trabaja principalmente con coordenadas normalizadas de la imagen.
- Puede confundirse si la persona esta girada, muy cerca, muy lejos o parcialmente tapada.
- Solo analiza una persona a la vez (`num_poses=1`).
- No existe una interfaz grafica avanzada fuera de la ventana de OpenCV.

## 16. Posibles mejoras futuras

Se recomienda considerar:

1. Agregar contador de repeticiones por ejercicio.
2. Guardar metricas en CSV o JSON.
3. Crear perfiles de usuario con calibracion de umbrales.
4. Permitir seleccionar modelo `lite` o `full` desde argumento de consola.
5. Agregar evaluacion de velocidad del movimiento.
6. Agregar una interfaz web o de escritorio.
7. Exportar reportes de sesion.
8. Incorporar pruebas automaticas para funciones matematicas.
9. Mejorar la deteccion de fases con maquinas de estado.
10. Agregar soporte para videos grabados ademas de webcam.

## 17. Conclusion

El proyecto cumple con el objetivo de entregar feedback de postura y movimiento en tiempo real usando vision por computador. Se construyo una solucion practica en Python que combina MediaPipe, OpenCV y NumPy para detectar articulaciones, calcular angulos y entregar recomendaciones visuales segun el ejercicio seleccionado.

La implementacion esta organizada en un unico script principal, lo que facilita su ejecucion y comprension. A la vez, los umbrales y modos definidos permiten ampliar el sistema hacia nuevos ejercicios o reglas mas avanzadas. Como prototipo tecnico, demuestra correctamente el uso de deteccion de pose para analisis de movimiento aplicado al entrenamiento fisico.
