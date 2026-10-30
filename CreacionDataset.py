import cv2
import os
import csv

#Rutas de configuración
VIDEOS_DIR = "videos"   #Carpeta donde se encuentran los videos
OUTPUT_DIR = "dataset_caras" #Carpeta donde se guardarán las imágenes de las caras detectadas
CSV_FILE = "dataset_rostros.csv" #Archivo CSV donde se guardarán las rutas de las imágenes y sus etiquetas

# Cargar el clasificador Haar Cascade para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Crear carpeta de salida si no existe
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Inicializar el archivo CSV y escribir la cabecera
with open(CSV_FILE, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["ruta_imagen", "etiqueta"])  #Cabecera del CSV

print("Iniciando la creación del dataset de rostros...")

# Recorrer las carpetas de videos (cada carpeta representa una persona/ etiqueta)
for persona in os.listdir(VIDEOS_DIR):
    persona_path = os.path.join(VIDEOS_DIR, persona)
    if not os.path.isdir(persona_path):
        continue  # Saltar si no es una carpeta

    # Crear una subcarpeta para la persona en el dataset de salida
    persona_output_dir = os.path.join(OUTPUT_DIR, persona)
    os.makedirs(persona_output_dir, exist_ok=True)

    # Recorrer los videos de la persona
    for video_name in os.listdir(persona_path):
        video_path = os.path.join(persona_path, video_name)
        if not video_path.lower().endswith(('.mp4', '.avi', '.mov')):
            continue  # Saltar si no es un archivo de video

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0

        print(f"Procesando video: {video_name} de la persona: {persona}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # Salir si no hay más frames

            # Extraer un fotograma cada 15 frames para evitar redundancia
            if frame_count % 15 == 0:
                gray = cv2.cvtcolor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(60, 60))

                for (x, y, w, h) in faces:
                    # Recortar la cara detectada
                    face_roi = frame[y:y+h, x:x+w] # Hacemos un recorte cuadrado para la cara

                    # Redimensionar la imagen de la cara a un tamaño estándar (por ejemplo, 128x128 píxeles)
                    face_roi = cv2.resize(face_roi, (128, 128))

                    # Guardar la imagen de la cara en la carpeta correspondiente
                    img_name = f"{persona}_{video_name[:-4]}_frame{frame_count}.jpg"
                    img_path = os.path.join(persona_output_dir, img_name)
                    # Guardar la imagen de la cara
                    cv2.imwrite(img_path, face_roi)

                    # Registrar la ruta de la imagen y la etiqueta en el archivo CSV
                    with open(CSV_FILE, mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerow([img_path, persona])
                    saved_count += 1

            frame_count += 1

            cap.release()
            print(f"Se han guardado {saved_count} imágenes de la persona: {persona} del video: {video_name}")

print(f'Dataset de rostros creado exitosamente. Datos guardados en: {CSV_FILE} y rostros guardados en la carpeta: {OUTPUT_DIR}.')