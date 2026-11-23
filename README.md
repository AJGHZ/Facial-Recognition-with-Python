# Reconocimiento Facial con CNN (TensorFlow y OpenCV)

Proyecto de reconocimiento facial modular en Python dividido en captura de vídeo, creación de dataset y entrenamiento/ejecución con una Red Neuronal Convolucional (CNN) en TensorFlow.

## Estructura del proyecto

- `GrabarVideos.py`: Graba vídeos desde la cámara web organizados por persona.
- `CreacionDataset.py`: Extrae fotogramas, detecta y recorta rostros a `128x128` y genera el registro en CSV.
- `app.py`: Entrena la CNN, cuantifica la precisión en validación y realiza reconocimiento facial en tiempo real desde la cámara.

## Requisitos e instalación

Asegúrate de tener Python 3.9+ y las librerías necesarias instaladas:

```bash
pip install tensorflow keras opencv-python numpy pandas matplotlib


# COMÓ EJECUTAR EL PROYECTO PARA SU CORRECTO FUNCIONAMINETO:

1. Grabar videos de la persona:
    Primero se deberá de ejecutar el script "GrabarVideos.py", siguiendo las instrucciones que se muestran en pantalla (click en frame de grabación y pulsar la tecla 's' para iniciar la grabación, presionar la tecla 'q' para detener / cancelar grabación)

2. Creación del dataset y el archivo CSV:
    Se deberá de ejecutar el script "CreaciónDataset.py" posterior a la generación del video de las personas (Paso 1).

    Procesa los videos grabados para generar las imagenes necesarias de los rostros y genera el archivo dataset_rostros.csv.

3. Entrenamiento del modelo:
    Ejecuta el archivo "app.py" y selecciona la opción 1 para entrenar la red neuronal, verificar la precisión del modelo y guardarlo.

4. Prueba en cámara la funcionalidad del reconocimiento facial.
    Ejecuta de nuevo "app.py" y selecciona la opción 2 para habilitar la cámara y ver la detección con el nombre y porcentaje de confianza (Presiona 'q' para salir)

EXTRA:
Por cada nueva persona que se quiera agregar al dataset, se deberan de repetir todos los pasos para reeentrenar el modelo y cuente con la información requerida.
Tambien podrás añadir videos de cualquier otra persona (famoso o desconocido) para ampliar el dataset generando una nueva subcarpeta con el nombre del individuo dentro del directorio "videos".