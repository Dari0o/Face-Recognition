import cv2
import time
import os

# Pfad zur XML-Datei mit dem vortrainierten Modell
face_cascade = cv2.CascadeClassifier(r'C:\Users\MiNiD\Documents\Visual Studio Code\Python\Gesichtserkennung\haarcascade_frontalface_default.xml')

# Starte die Kamera
cap = cv2.VideoCapture(0)

# Definiere die Codec-Einstellungen und erstelle den Video Writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = None

# Starte die Zeitmessung
start_time = time.time()

# Definiere die maximale Aufnahmezeit in Sekunden (30 Minuten)
max_record_time = int(input("How long should the video be recorded? : ")) * 60


# Definiere die aktuelle Aufnahmezeit
current_record_time = 0

# Erstelle Ordner für Fotos, wenn er nicht existiert
if not os.path.exists('face_images'):
    os.makedirs('face_images')

# Erstelle Ordner für Videos, wenn er nicht existiert
if not os.path.exists('recordings'):
    os.makedirs('recordings')

while(cap.isOpened()):
    # Lese ein Frame von der Kamera
    ret, frame = cap.read()

    # Wenn das Frame gelesen wurde
    if ret:
        # Konvertiere das Frame in Graustufen
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Erkenne Gesichter im Frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # Zeichne Rechteck um jedes erkannte Gesicht
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            # Speichere das erkannte Gesicht als Bild
            cv2.imwrite("face_images/face_{}.jpg".format(int(time.time())), frame[y:y+h, x:x+w])

        # Wenn die Videoaufnahme noch nicht gestartet wurde oder die maximale Aufnahmezeit erreicht wurde
        if not out or current_record_time >= max_record_time:
            # Wenn bereits eine Videoaufnahme läuft, stoppe sie
            if out:
                out.release()

            # Öffne eine neue Video-Datei für die Fortsetzung der Aufnahme
            out = cv2.VideoWriter('recordings/output_{}.avi'.format(int(time.time())), fourcc, 20.0, (640,480))

            # Starte die Zeitmessung für die nächste Aufnahme
            start_time = time.time()

            # Setze die aktuelle Aufnahmezeit zurück
            current_record_time = 0

        # Schreibe das Frame in die Video-Datei
        out.write(frame)

        # Zeige das aktuelle Frame
        cv2.imshow('frame',frame)

        # Aktualisiere die aktuelle Aufnahmezeit
        current_record_time = time.time() - start_time

        # Wenn die 'q'-Taste gedrückt wird, beende das Programm
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Beende alles
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
