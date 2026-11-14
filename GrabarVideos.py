import cv2
import os

# Configuracón de carpetas y archivos
VIDEOS_DIR = "videos"  # Carpeta que contiene los videos de las personas

# Crear directorio si no existe
os.makedirs(VIDEOS_DIR, exist_ok=True)

print("Grabador de videos para creación del dataset...")
persona = input("Ingrese el nombre de la persona (etiqueta): ").strip()

persona_path = os.path.join(VIDEOS_DIR, persona)
# Crear carpeta para la persona si no existe
os.makedirs(persona_path, exist_ok=True)

# Bucle para grabar múltiples videos
while True:
    # Determinar nombre unico para el video actual basado en la cantidad de videos existentes
    existing_videos = [f for f in os.listdir(persona_path) if f.endswith(('.mp4', '.avi', '.mov'))]
    video_index = len(existing_videos) + 1
    video_filename = os.path.join(persona_path, f"video_{video_index}.mp4")

    # Inicializar la captura de video desde la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: No se pudo iniciar la grabacíon. Asegúrese de que la cámara esté conectada y funcionando.")
        break

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec para el archivo de salida
    fps = 20.0  # Frames por segundo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    print(f"\n[Grabando video para la persona: {persona} | Video # {video_index}].")
    print(" -> Presione **'s'** para comenzar a grabar (Mueva la cabeza lentamente), **'q'** para detener la grabación y salir.")

    grabando = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el video. Saliendo...")
            break

        # Mostrar el frame en una ventana
        display_frame = frame.copy()
        if grabando:
            cv2.putText(display_frame, "Grabando... (presione 'q' para detener)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            out.write(frame)  # Guardar el frame en el archivo de video
        else:
            cv2.putText(display_frame, f"Video # {video_index} - Presione 's' para grabar | 'q' para salir", (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Grabador de Rostros", display_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and not grabando:
            grabando = True
            print("Grabación iniciada. Mueva la cabeza lentamente para capturar diferentes ángulos.")
        elif key == ord('q'):
            break  # Salir del bucle de grabación

    # Liberar la cámara y el escritor de video
    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if not grabando:
        print("Grabación cancelada. Saliendo...")
        break
    else:
        print(f"Video {video_index} guardado con exito en: {video_filename}")

    # Preguntar al usuario si desea grabar otro video
    continuar = input("¿Desea grabar otro video para la misma persona? (s/n): ").strip().lower()
    if continuar != 's':
        print(f"Finalizando la grabación de videos. Los video de la persona '{persona}' se han guardado en la carpeta: {persona_path}")
        break