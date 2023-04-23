import cv2
import time
import datetime
import pytz
import os

# Setze die Zeitzone auf Europa/Berlin
timezone = pytz.timezone('Europe/Berlin')

# Starte die Kamera
cap = cv2.VideoCapture(0)

# Definiere die Codec-Einstellungen
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Verringere die Auflösung auf 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Lade den Kaskaden-Klassifikator für Gesichter
face_cascade = cv2.CascadeClassifier(r'/home/kali/Documents/Face-Recognition/haarcascade_frontalface_default.xml')

# Erstelle den Ordner zum Speichern der erkannten Gesichter
output_folderF = 'detected_faces'
if not os.path.exists(output_folderF):
    os.makedirs(output_folderF)

# Öffne ein Fenster zur Anzeige des Videos
cv2.namedWindow("Video")

# Definiere den Dateinamen und Pfad für das Video
output_folderV = 'recordings'
if not os.path.exists(output_folderV):
    os.makedirs(output_folderV)
current_date = datetime.datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")
video_filename = f"{output_folderV}/{current_date}.mp4"

# Definiere die Video-Aufnahme-Einstellungen
fps = 20.0
frame_size = (640, 480)

# Erstelle den VideoWriter
out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)

# Setze das Start-Zeitstempel für die Video-Aufnahme
start_time = time.time()

while True:
    # Lese ein Frame von der Kamera
    ret, frame = cap.read()

    # Wenn das Frame gelesen wurde
    if ret:
        # Füge die aktuelle Uhrzeit in der oberen rechten Ecke hinzu
        font = cv2.FONT_HERSHEY_SIMPLEX
        current_time = datetime.datetime.now(timezone).strftime("%H:%M:%S")
        cv2.putText(frame, current_time, (470, 20), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Konvertiere das Frame in Graustufen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Suche nach Gesichtern im Frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Zeichne ein Viereck um jedes erkannte Gesicht und speichere es in den Ordner
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            timestamp = int(time.time())
            filename = f"{output_folderF}/{timestamp}.jpg"
            cv2.imwrite(filename, frame[y:y+h, x:x+w])

        # Schreibe das Frame in das Video
        out.write(frame)

        # Füge ein Fadenkreuz in der Mitte hinzu
        center_coordinates = (320, 240)
        thickness = 2
        cv2.line(frame, (center_coordinates[0]-10, center_coordinates[1]), (center_coordinates[0]+10, center_coordinates[1]), (255, 255, 255), thickness)
        cv2.line(frame, (center_coordinates[0], center_coordinates[1]-5), (center_coordinates[0], center_coordinates[1]+5), (255, 255, 255), thickness)

        # Zeige das aktuelle Frame im Fenster an
        cv2.imshow("Video", frame)

        # Wenn die Zeit für die Video-Aufnahme abgelaufen ist, speichere das Video und starte eine neue Aufnahme
        elapsed_time = time.time() - start_time
        if elapsed_time > 600:
            out.release()
            current_date = datetime.datetime.now(timezone).strftime("%Y-%m-%d_%H-%M-%S")
            video_filename = f"{output_folderV}/{current_date}.mp4"
            out = cv2.VideoWriter(video_filename, fourcc, fps, frame_size)
            start_time = time.time()

        # Wenn die Taste "q" gedrückt wird, breche die Schleife ab
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Gib die Ressourcen frei und schließe das Fenster
cap.release()
out.release()
cv2.destroyAllWindows()
