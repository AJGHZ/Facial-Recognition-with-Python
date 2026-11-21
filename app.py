import os
import json
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Cargar el dataset de rostros desde el archivo CSV
DATASET_DIR = "dataset_caras"
CSV_FILE = "dataset_rostros.csv"
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
MODEL_PATH = "modelo_reconocimiento_facial-keras"
CLASSES_FILE = "clases.json"
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