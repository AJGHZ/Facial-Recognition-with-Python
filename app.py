import os
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models
import matplotlib.pyplot as plt

# Cargar el dataset de rostros desde el archivo CSV
DATASET_DIR = "dataset_caras"
CSV_FILE = "dataset_rostros.csv"
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "modelo_reconocimiento_facial-keras"
CLASSES_PATH = "clases.json"
CONFIDENCE_THRESHOLD = 0.60  # Umbral de confianza para la predicción

def entrenar_modelo():
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"No se encontró el directorio del dataset: {DATASET_DIR}. Ejecuta primero el script CreacionDataset.py para generar el dataset de rostros.")

    # Cargar el dataset desde el archivo CSV
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE)
        print(f"Registros encontrados en el dataset '{CSV_FILE}': {len(df)} | Etiquetas únicas: {df['etiqueta'].nunique()}")

        # Carga y división 80/20 del dataset en entrenamiento y validación
        train_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR, 
            validation_split = 0.2, 
            subset= "training",
            seed = 123,
            image_size = (IMG_HEIGHT, IMG_WIDTH),
            batch_size = BATCH_SIZE
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            DATASET_DIR,
            validation_split = 0.2,
            subset = "validation",
            seed = 123,
            image_size = (IMG_HEIGHT, IMG_WIDTH),
            batch_size = BATCH_SIZE
        )

        class_names = train_ds.class_names
        num_classes = len(class_names)
        print(f"\nPersonas / Clases detectadas: ({num_classes}): {class_names}")

        # Guardar el orden exacto de las clases para su uso en tiempo real
        with open(CLASSES_PATH, "w", encoding="utf-8")as f:
            json.dump(class_names, f, ensure_ascii = False, indent = 2)

        AUTOTUNE = tf.data.AUTOTUNE
        train_ds = train_ds.cache().prefetch(buffer_size = AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size = AUTOTUNE)

        # Construcción del modelo CNN
        data_augmentation = models.Sequential([
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.8),
            layers.RandomZoom(0.1)
        ])

        model = models.Sequential([
            layers.Input(shape = (IMG_HEIGHT, IMG_WIDTH, 3)),
            data_augmentation,
            layers.Rescaling(1.0 / 255.0),

            layers.Conv2D(32, kernel_size = 3, padding = "same", activation = "relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(64, kernel_size = 3, padding = "same", activation = "relu"),
            layers.MaxPooling2D(),

            layers.Conv2D(128, kernel_size = 3, padding = "same", activation = "relu"),
            layers.MaxPooling2D(),

            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(128, activation = "relu"),
            layers.Dense(num_classes, activation = "softmax")
        ])

        model.compile(
            optimizer = "adam",
            loss = tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics = ["accuracy"]
        )

        model.summary()

        # Entrenamiento del modelo
        print("\nEntrenando el modelo...")
        history = model.fit(
            train_ds,
            validation_data = val_ds,
            epochs = EPOCHS
        )

        # Cuantificación del modelo de presición
        print("\nCUANTIFICACIÓN DE PRECISIÓN EN VALIDACIÓN...")
        val_loss, val_accuracy = model.evaluate(val_ds, verbose = 0)
        print(f"Pérdida (loss) en validación: {val_loss:.4f}")
        print(f"Precisión (accuracy) en validación: {val_accuracy * 100:.2f}")

        model.save(MODEL_PATH)
        print(f"\nModelo guardado en: '{MODEL_PATH}' y calsses en: '{CLASSES_PATH}'.")

        # Graficar la precisión y pérdida del entrenamiento
        epochs_ran = len(history.history["accuracy"])
        epochs_range = range(epochs_ran)

        plt.figure(figsize = (12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, history.history["accuracy"], label = "ENTRENAMIENTO")
        plt.plot(epochs_range, history.history["val_accuracy"], label = "VALIDACIÓN")
        plt.title("Precisión (accuracy) por época")
        plt.xlabel("Época")
        plt.ylabel("Precisión (accuracy)")
        plt.legend(loc = "upper right")

        plt.tight_layout()
        plt.show()


def reconocer_en_camara():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CLASSES_PATH):
        print(f"No se encontró '{MODEL_PATH}' o '{CLASSES_PATH}'. Ejecuta primero el script de entrenamiento para generar el modelo y las clases.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASSES_PATH, "r", encoding = "utf-8") as f:
        class_names = json.load(f)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("No se pudo abrir la cámara.")
        return

    print("\nIniciando reconocimiento facial en tiempo real. Presiona 'q' para salir.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR AL CAPTURAR EL FOTOGRAMA DE LA CÁMARA")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor = 1.3,
            minNeighbors = 5,
            minSize = (60, 60)
        )

        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue

            face_resized = cv2.resize(face_roi, (IMG_WIDTH, IMG_HEIGHT))
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            input_tensor = np.expand_dims(face_rgb.astype(np.float32), axis = 0)

            predictions = model.predict(input_tensor, verbose = 0)[0]
            class_idx = int(np.argmax(predictions))
            confidence = float(predictions[class_idx])

            if confidence >= CONFIDENCE_THRESHOLD:
                label = f"{class_names[class_idx]} ({confidence * 100:.1f}%)"
                color = (0, 255, 0)  # Verde para predicciones confiables
            else:
                label = "Desconocido"
                color = (0, 165, 255)  # Naranja para predicciones no confiables

            cv2.rectangle(frame, (x,y), (x + w, y+h), color, 2)
            cv2.putText(
                frame,
                label,
                (x, y - 10 if y - 10 >20 else y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )
        
        cv2.imshow("Reconocimiento Facial - CNN TensorFlow", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("\nCámara liberada y ventanas cerradas. Fin del reconocimiento facial.")

if __name__ == "__main__":
    print("=== RECONOCIMIENTO FACIAL CON CNN (TENSORFLOW) ===")
    print("1. Entrenar modelo")
    print("2. Reconocer en cámara")
    choice = input("Selecciona una opción (1 o 2): ").strip()

    if choice == "1":
        entrenar_modelo()
    elif choice == "2":
        reconocer_en_camara()
    else:
        print("Opción no válida. Por favor, selecciona 1 o 2.")