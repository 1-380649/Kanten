# achtung hier werden die mtf kanten gefunden
# das bild zumtesten hatte allerdings eine auflösung von 444dpi (361dpi funktioniert auch)
# bzw. mit der neuen (24.2.2024) hough-transformation in mehreren Auflösungen
# für utt bilder anderer auflösung müssen ggf. werte angepasst werden - beispielsweise grenzwerte in pixel
# die geraden linien der erkannten kannten in filtered_lines setzen sich erstmal teilweise aus mehreren einzelenen linien zusammen
# mit erschieden linienfarben können solche gestückelten linen gut erkennbar gemacht werden
# mit den funktionen im bereich merge und connect werden diese stücke über einen grenzwert verschmolzen so
# dass die linien durchgehend sind und ihre anzahl 36 beträgt (je vier kanten von neun quadraten bzw. rhomben)
import os
import cv2
import numpy as np
import matplotlib.pyplot as pp
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import time
from PIL import Image, ExifTags
import colorsys
import subprocess  # zum auslesen des Profilnamens mit ExifTool notwendig
import json  # zum auslesen des Profilnamens mit ExifTool notwendig
# from sklearn.linear_model import RANSACRegressor # zur berechnung des größten rechtecks notwendig (zu kompliziert)
import sys
from pathlib import Path
import scipy.signal              # pip install scipy
import scipy.ndimage.morphology  # pip install scipy (nicht mehr empfohlen -> deshalb "from scipy.ndimage import binary_opening")
from scipy.ndimage import binary_opening
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist
# from skimage.restoration import estimate_sigma
# from skimage import io, img_as_float
from skimage import io, exposure
from skimage.filters import median
import random
import logging # wurde am 20.1.24 aufgrund einer fehlermeldung bei einem bestimmten target implementiert - kann evtl. wieder weg
import tempfile
import shutil
import datetime

global image_path
# Bild laden und vorbereiten
image_path = 'testfiles/utt_060224_361dpi.jpg'  # -> geht
# image_path = 'testfiles/color_neu_300_scharfgezeichnet.jpg'  # ->
# image_path = 'testfiles/utt_060224_361dpi_stark_geschaerft.jpg' # -> geht
# image_path = 'testfiles/utt_060224_444dpi.jpg'  # -> geht
# image_path = 'testfiles/utt_060224_300.jpg'  # -> geht wenn in zeile 195 maxLineGap= von 30 auf 40 oder bei 30 und threshold=45 -> DAS MUSS NOCH BEI DEN ANDEREN GETESTET WERDEN
# image_path = 'testfiles/utt_4k_14_170814_UTT_SG_ECI_TW9.jpg'  # -> geht nicht
# image_path = 'testfiles/PAL_UTT262_240209_002_anBLHA.jpg'  # -> geht nicht bei maxLineGap=40
# zeile 195 ist ein interessanter filter
# momentan würde ich entweder maxLineGap= irgendwas zwischen 30 und 39 versuchen oder 30 lassen weil bewährt und lieber mit threshold weiter spielen

image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# ich vermute, dass es im Hinblick auf die ausgerechnete MTF-Qualität einen Zusammenhang zwischen der ROI-Größe und der Auflösung gibt
# das bedeutet, dass umso höher die Auflösung des Bildes ist, umso größer die ROI sein soll (muss)
# Grund: in der bisherigen Praxis zeigt sich, dass bei steigender Auflösung des Bildes die MTF-Qualität nicht ansteigt - was ich aber unterstellen würde
# ebenso scheint die Berechnungsbreite der kanten (EDGE_groesse) in pixeln eine Rolle für die MTF-Qualität zu spielen
# bisher war die ROI-Größe von 110x110 Pixeln auf eine Auflösung von 300dpi ausgelegt (ist jetzt dynamisch)
# bisher war die Berechnungsbreite der Kanten auf 90 Pixel festgelegt (ist jetzt dynamisch)
# siehe ab zeile 111

# ----------------- Anfang sammeln von Infos zum Bild ---------------------------------------

def gather_image_info(image_path, print_info=True):
    # Grundlegende Bildinformationen mit OpenCV
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    current_pixels = width * height

    # Bild mit Pillow öffnen für DPI und EXIF
    img_pil = Image.open(image_path)
    dpi = img_pil.info.get('dpi', (None, None))

    # Sammle EXIF-Daten, falls vorhanden
    exif_data = img_pil._getexif()
    exif = {}
    original_pixels = None
    if exif_data:
        exif = {
            ExifTags.TAGS.get(tag, tag): value
            for tag, value in exif_data.items()
            if tag in ExifTags.TAGS
        }
        # Ermittle die ursprüngliche Pixelgröße aus EXIF
        image_width_exif = exif.get('ImageWidth')
        image_length_exif = exif.get('ImageLength')
        if image_width_exif and image_length_exif:
            original_pixels = image_width_exif * image_length_exif

    # Berechne die Veränderung und die prozentuale Veränderung, falls möglich
    if original_pixels:
        pixel_change = current_pixels - original_pixels
        percent_change = (pixel_change / original_pixels) * 100
        change_direction = "zugenommen" if pixel_change > 0 else "abgenommen"
        change_info = f"Die Pixelanzahl hat im Vergleich zur Erstaufnahme um {abs(percent_change):.2f}% {change_direction}."
    else:
        change_info = "Keine Informationen über die ursprüngliche Pixelgröße verfügbar."

    # Daten sammeln
    image_info = {
        "Dateiname": os.path.basename(image_path),
        "Größe in Pixel": f"{width} x {height}",
        "Farbkanäle": channels,
        "DPI": dpi if dpi != (None, None) else "DPI-Information nicht verfügbar",
        "EXIF-Daten": exif,
        "Veränderung": change_info
    }

    # Bei Bedarf Informationen ausgeben
    if print_info:
        print('---------------------------------------------')
        for key, value in image_info.items():
            if key != "EXIF-Daten":
                print(f"{key}: {value}")
        print("EXIF-Daten:")
        for tag, value in image_info["EXIF-Daten"].items():
            print(f"  {tag}: {value}")
        print('---------------------------------------------')

    return image_info

# Aufruf der Funktion
gather_image_info(image_path, print_info=True)

# ----------------- Ende sammeln von Infos zum Bild ---------------------------------------

# ----------------Anpassen der ROI-Größe an die Auflösung des Bildes---------------------------
ROI_groesse = 110 # Groesse der Region of Interest (ROI) in Pixeln als default bei 300dpi

image11 = Image.open(image_path)
dpi_x, dpi_y = image11.info.get("dpi", (None, None)) # wird gebraucht um später die ROI-Größe dynamisch an die Auflösung des Bildes anzupassen

ROI_groesse = int(dpi_x * 110 / 300)
print('Kalkulierte ROI-Groesse:', ROI_groesse)

QUADRAT_groesse = ROI_groesse  # Größe der Quadrate in Pixeln
EDGE_groesse = QUADRAT_groesse - 1  # Berechnungsbreite der Kanten in Pixeln
# ----------------ENDE Anpassen der ROI-Größe an die Auflösung des Bildes---------------------------

# ------ hier werden die zonen der 9 slanted edge-felder  mittels mustererkennung erkannt, welche später wichtig für die linienerkennung wichtig sind ------

def non_max_suppression(boxes, overlapThresh):
    if len(boxes) == 0:
        return []

    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    pick = []
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        overlap = (w * h) / area[idxs[:last]]

        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

    return boxes[pick].astype("int")

# skalieree template und zielbild auf die gleich auflöungsgröße damit die mustererkennung besser funktioniert
def adjust_template_to_target_dpi(template, target_dpi, template_dpi):
    scale_factor = target_dpi / template_dpi
    width = int(template.shape[1] * scale_factor)
    height = int(template.shape[0] * scale_factor)
    resized_template = cv2.resize(template, (width, height), interpolation=cv2.INTER_AREA)
    return resized_template


# DPI-Werte
target_dpi = int(dpi_y)
template_dpi = 300

# Lade das Zielbild und das Template-Muster
target_image = gray  # Zielbild in Graustufen
template = cv2.imread('testfiles/edge_grau.jpg', 0)  # Template-Muster zur erkennung der bereich für slanted edge in Graustufen

# Liste zur Speicherung der Fundorte
locations = []

# Skaliere das Template basierend auf DPI-Werten
template_adjusted = adjust_template_to_target_dpi(template, target_dpi, template_dpi)

# Führe Template Matching mit dem angepassten Template durch
result = cv2.matchTemplate(target_image, template_adjusted, cv2.TM_CCOEFF_NORMED)

# Schwellenwert festlegen
threshold = 0.8

# Positionen bestimmen, die den Schwellenwert überschreiten
locations = np.where(result >= threshold)

# Rechtecke für jede gefundene Position erstellen (angepasst an skaliertes Template)
rectangles = []
for pt in zip(*locations[::-1]):  # Switch columns and rows
    rectangles.append([int(pt[0]), int(pt[1]), int(pt[0] + template_adjusted.shape[1]), int(pt[1] + template_adjusted.shape[0])])

rectangles = np.array(rectangles)

# Non-Maximum Suppression anwenden
pick = non_max_suppression(rectangles, 0.5)  # Überlappungsschwelle

# Erstelle eine Kopie des Zielbildes für die Anzeige
target_image_display = target_image.copy()

# Markiere die detektierten Bereiche im Bild (angepasst an skaliertes Template)
# for (startX, startY, endX, endY) in pick:
#    cv2.rectangle(target_image_display, (startX, startY), (endX, endY), (0, 255, 255), 2)

'''
# Erstelle eine Matplotlib-Figur und eine Achse
fig, ax = pp.subplots(figsize=(18, 12))
# Zeige das Bild auf der Achse
ax.imshow(target_image_display, cmap='gray')
# Füge grüne Rechtecke für die erkannten Bereiche hinzu
for (startX, startY, endX, endY) in pick:
    # Erstelle ein Rechteck-Patch
    rect = patches.Rectangle((startX, startY), endX - startX, endY - startY, linewidth=1, edgecolor='#00FF00', facecolor='none')
    # Füge das Rechteck dem Plot hinzu
    ax.add_patch(rect)
# Setze die Achsentitel und schalte die Achsen aus
ax.set_title('Detected Areas')
ax.axis('on')
# Zeige das Ergebnis
pp.show()
'''

# Erstelle eine leere Maske mit denselben Abmessungen wie `gray`
mask = np.zeros_like(gray)

# Fülle die Bereiche innerhalb der Rechtecke in der Maske mit Weiß
for (startX, startY, endX, endY) in pick:
    cv2.rectangle(mask, (startX, startY), (endX, endY), (255, 255, 255), -1)

# Wende die Maske auf das Bild an
gray = cv2.bitwise_and(gray, gray, mask=mask)
'''
# Zeige das Ergebnis
pp.figure(figsize=(18, 12))
pp.imshow(gray, cmap='gray')
pp.title('Bereiche innerhalb der Markierungen behalten')
pp.axis('on')
pp.show()
'''

# ------ bis hier wurden die zonen mittels mustererkennung erkannt, welche später wichtig für die linienerkennung wichtig sind ---ENDE---

def calculate_length(x1, y1, x2, y2):
    """Berechnet die Länge einer Linie."""
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def calculate_angle(x1, y1, x2, y2):
    """Berechnet den Winkel einer Linie in Grad relativ zur horizontalen Achse."""
    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
    # Normierung des Winkels auf einen Bereich von 0 bis 180 Grad
    normalized_angle = angle % 180
    return normalized_angle

def filter_lines(lines, min_length=150, max_length=400, min_angle=4, max_angle=8):
    """Filtert Linien basierend auf Länge, spezifischer Neigung und Ausschluss von Randbereichen."""
    filtered_lines = []
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    left_limit = width * 0.15
    right_limit = width * 0.85
    top_limit = height * 0.10
    bottom_limit = height * 0.90

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = calculate_length(x1, y1, x2, y2)
        angle = calculate_angle(x1, y1, x2, y2)

        # Prüfe, ob die Linie innerhalb der erlaubten Längen- und Winkelbereiche liegt
        # und ob die Endpunkte der Linie außerhalb der definierten Randbereiche liegen
        # zusätzlich werden alle linien gelöscht die innerhalb einer zone von 15% der bildbreite vom rechten und linken rand befinden und
        # die sich innerhalb einer zone von 10% der bildhöhe vom oberen und unteren bildrand befinden
        if (min_length <= length <= max_length and
            (angle >= min_angle and angle <= max_angle or
             angle >= (90 - max_angle) and angle <= (90 - min_angle) or
             angle >= (90 + min_angle) and angle <= (90 + max_angle) or
             angle >= (180 - max_angle) and angle <= (180 - min_angle)) and
            (x1 > left_limit and x1 < right_limit and
             x2 > left_limit and x2 < right_limit and
             y1 > top_limit and y1 < bottom_limit and
             y2 > top_limit and y2 < bottom_limit)):
                filtered_lines.append(line[0])  # Speichere die Linienkoordinaten direkt

    return filtered_lines

# brauche ich später um auf unberührtem originalbild die gelbe zeichnung, resultierend aus der connect und merge funktion zu visualisieren
image_fresh = cv2.imread(image_path)

# Startzeit für den ersten Abschnitt (Canny und Lines)
start_time_section1 = time.time()

edges = cv2.Canny(gray, 50, 150)
# edges = cv2.Canny(gray, 50, 150)  # Experimentieren Sie mit dem Wert von lower_threshold (erster wert)

# --------- bild nach canny anzeigen
# Kantenbild mit Matplotlib anzeigen (grundlegender test)
'''
pp.figure(figsize=(10, 10))
pp.imshow(edges, cmap='gray')
pp.title('Canny Kanten')
pp.axis('off')  # Achsen ausschalten für eine klarere Ansicht
# Interaktive Werkzeuge von Matplotlib nutzen (Zoom, Pan, etc.)
pp.show()
'''
# --------- ende bild anzeigen nach canny

# Hough-Transformation zur Linienfindung
'''
-> Hough-Transformation:
-> Die Hough-Transformation ist ein bekannter Algorithmus zur Erkennung von Linien in einem Bild.
-> In OpenCV wird die Funktion cv2.HoughLinesP verwendet, um die Hough-Transformation auf einem Kantenbild anzuwenden und Linien zu finden.
-> Parameter der Hough-Transformation:
--> edges: Das Eingabebild, auf dem die Hough-Transformation angewendet wird. Es handelt sich um das Bild mit den Kanten, das zuvor durch den Canny-Kanten-Detektor erstellt wurde.
--> rho: Die Auflösung des Abstandsrasters in Pixel. Hier ist der Wert 0.5 gewählt, was bedeutet, dass das Raster die halbe Pixelauflösung des Bildes hat.
--> theta: Die Auflösung des Winkelrasters in Radiant. Hier ist np.pi / 360 eingestellt, was bedeutet, dass das Raster 1 Grad Winkelauflösung hat.
--> threshold: Ein Schwellenwert, der angibt, wie viele Abstimmungen ein Punkt im Raum der Hough-Parameter benötigt, um als Linie betrachtet zu werden. Linien mit einer Stimmenanzahl über diesem Schwellenwert werden als Ergebnis zurückgegeben.
--> minLineLength: Die minimale Länge einer Linie. Linien, die kürzer sind als dieser Wert, werden ignoriert.
--> maxLineGap: Der maximale Lückenabstand, der noch als Teil derselben Linie betrachtet wird.
'''
# lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=100, maxLineGap=10)
# lines = cv2.HoughLinesP(edges, rho=0.5, theta=np.pi / 360, threshold=50, minLineLength=100, maxLineGap=40)  # funktioniert für 361dpi, 444dpi, nicht für 300dpiExperimentieren Sie mit diesen Werten wenn linien nicht gut erkannt werden
# lines = cv2.HoughLinesP(edges, rho=0.7, theta=np.pi / 360, threshold=50, minLineLength=90, maxLineGap=40) # funktioniert für 300dpi

# lines = cv2.HoughLinesP(edges, rho=0.7, theta=np.pi / 360, threshold=50, minLineLength=90, maxLineGap=30) # funktioniert für 300, 361 und 444 dpi
# lines = cv2.HoughLinesP(edges, rho=0.7, theta=np.pi / 1080, threshold=45, minLineLength=90, maxLineGap=30) # rho 6 ergbit besser erkannt linien bei einem target ist aber ansonsten nicht so gut .... 1080 ergibt einen feineren durchlauf weil der bogenmaßschritt kleiner ist (pi/1080 anstelle von pi/360)
lines = cv2.HoughLinesP(edges, rho=0.7, theta=np.pi / 720, threshold=50, minLineLength=90, maxLineGap=30)

# Ausführungszeit des ersten Abschnitts messen (Canny und Lines)
elapsed_time_section1 = time.time() - start_time_section1
print("Ausführungszeit Canny und Lines:", elapsed_time_section1, "Sekunden")

# --------- bild nach hough transformation anzeigen (grundlegender Test)

# Kopie des Originalbildes für das Zeichnen der Linien für slanted edge
image_with_lines = np.copy(image)

# Linien auf dem Bild zeichnen
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Bild mit Matplotlib anzeigen
'''
pp.figure(figsize=(10, 10))
# Umwandlung von BGR zu RGB
image_with_lines_rgb = cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB)
pp.imshow(image_with_lines_rgb)
pp.title('Bild mit erkannten Linien')
pp.axis('off')  # Achsenbeschriftungen entfernen für eine klarere Ansicht
pp.show()
'''
# --------- Ende bild anzeigen nach hough transformation

# --------- Linien filtern
filtered_lines = filter_lines(lines, min_length=150, max_length=400, min_angle=4, max_angle=8)

# Visualisierung der gefilterten Linien
for x1, y1, x2, y2 in filtered_lines:
    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Anzahl der Linien in lines
num_lines = len(lines)
print("Anzahl der Linien in lines:", num_lines)

# Anzahl der gefilterten Linien in filtered_lines vor connect_lines
num_filtered_lines_before = len(filtered_lines)
print("Anzahl der gefilterten Linien in filtered_lines vor connect_lines:", num_filtered_lines_before)


lines_array = np.array(filtered_lines) # filtered_lines von Tulpel in ein NumPy-Array konvertieren (für funktion filter_isolated_lines)


# ------------------------------ Linien in verschiedenen Farben zeichnen

image_with_lines = np.copy(image)
# Farben dynamisch generieren
num_colors = len(filtered_lines)
colors = [colorsys.hsv_to_rgb(i / num_colors, 1, 1) for i in range(num_colors)]

if filtered_lines is not None:
    for i, line in enumerate(filtered_lines):
        x1, y1, x2, y2 = line
        color = tuple(int(c * 255) for c in colors[i])  # Konvertieren Sie den Farbwert in den RGB-Bereich
        cv2.line(image_with_lines, (x1, y1), (x2, y2), color, 2)

# linien auf bild in verschiednen farben anzeigen
'''
image_with_lines_rgb = cv2.cvtColor(image_with_lines, cv2.COLOR_BGR2RGB)
pp.figure(figsize=(10, 10))
pp.imshow(image_with_lines_rgb)
pp.title('filtered_lines in verschiedenen Farben')
pp.axis('off')
pp.show()
'''
# ------------------------------ Ende Linien in verschiedenen Farben zeichnen

# ------------------------------ Anfang ERKANNTE Linien zeichnen

pp.figure(figsize=(10, 10))
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
pp.imshow(image_rgb)
pp.title('filtered_lines # '+os.path.basename(image_path)+" # "+"weitere Infos")
pp.axis('on')
# pp.show()
pp.pause(2)  # Anzeige für 1 Sekunde pausieren
pp.close()  # Fenster schließen

# ------------------------------ Ende Linien zeichnen

# ------------------------------ Anfang connect und merge lines

def merge_lines(line1, line2):
    """
    Nimmt zwei Linien und verschmilzt sie zu einer Linie,
    indem sie den am weitesten entfernten Startpunkt und Endpunkt auswählt.
    """
    # Extrahiere die Punkte
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Liste aller Punkte
    points = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    # Sortiere die Punkte basierend auf ihrer x-Koordinate
    sorted_points = sorted(points, key=lambda point: point[0])

    # Wähle den am weitesten links und den am weitesten rechts gelegenen Punkt als Endpunkte der neuen Linie
    start_point = sorted_points[0]
    end_point = sorted_points[-1]

    return (start_point[0], start_point[1], end_point[0], end_point[1])


# def connect_lines(filtered_lines, max_distance=50):
def connect_lines(filtered_lines, max_distance=50):
    connected_lines = []

    while filtered_lines:
        base_line = filtered_lines.pop(0)
        was_merged = False

        i = 0
        while i < len(connected_lines):
            connected_line = connected_lines[i]

            # Berechne den Mittelpunkt jeder Linie
            bx1, by1, bx2, by2 = base_line
            cx1, cy1, cx2, cy2 = connected_line
            base_mid = ((bx1 + bx2) / 2, (by1 + by2) / 2)
            connected_mid = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)

            # Berechne die Distanz zwischen den Mittelpunkten
            distance = np.sqrt((connected_mid[0] - base_mid[0]) ** 2 + (connected_mid[1] - base_mid[1]) ** 2)

            if distance < max_distance:
                # Verschmelze die Linien
                new_line = merge_lines(base_line, connected_line)
                connected_lines[i] = new_line  # Ersetze die verbundene Linie mit der verschmolzenen Linie
                was_merged = True
                break
            i += 1

        if not was_merged:
            connected_lines.append(base_line)

    return connected_lines

# Verbinden Sie die Linien in filtered_lines
connected_lines = connect_lines(filtered_lines)

# Anzahl der Linien in connected_lines nach connect_lines
num_connected_lines = len(connected_lines)
print("Anzahl der Linien nach connect_lines:", num_connected_lines)

# Überprüfen, ob die Anzahl verringert wurde
if num_connected_lines < num_filtered_lines_before:
    print("Die Menge der Linien hat sich durch connect_lines verringert.")
else:
    print("Die Menge der Linien hat sich nicht verringert.")

# ------------------------------ Anfang beschrifte Linien in einer strukturierten Form

def assign_line_to_cell(line, cell_width, cell_height):
    x1, y1, x2, y2 = line
    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
    col = int(mid_x // cell_width)
    row = int(mid_y // cell_height)
    return row, col


def calculate_distance_to_edges(line, cell_x_min, cell_y_min, cell_x_max, cell_y_max):
    x1, y1, x2, y2 = line
    left_distance = min(x1, x2) - cell_x_min
    top_distance = min(y1, y2) - cell_y_min
    right_distance = cell_x_max - max(x1, x2)
    bottom_distance = cell_y_max - max(y1, y2)
    return left_distance, top_distance, right_distance, bottom_distance


def sort_and_label_lines(cells, cell_width, cell_height, width, height):
    labels = {}
    id_counter = 1
    for row in range(3):
        for col in range(3):
            cell_lines = cells[(row, col)]
            cell_x_min = col * cell_width
            cell_y_min = row * cell_height
            cell_x_max = cell_x_min + cell_width
            cell_y_max = cell_y_min + cell_height

            distances = {}
            for line in cell_lines:
                # Überprüfen, ob line ein numpy.ndarray ist und entsprechend behandeln
                if isinstance(line, np.ndarray):
                    line_tuple = tuple(line.flatten())
                else:
                    # line ist bereits ein Tuple
                    line_tuple = line

                distances[line_tuple] = calculate_distance_to_edges(line_tuple, cell_x_min, cell_y_min, cell_x_max, cell_y_max)

            # Sortieren und Auswählen von Linien basierend auf ihrer Nähe zu den Kanten
            # Hinweis: Dieser Teil des Codes muss möglicherweise auch angepasst werden, um mit Tuples korrekt zu funktionieren
            sorted_keys = sorted(distances.keys(), key=lambda l: distances[l])  # Beispiel, wie man basierend auf den Distanzen sortieren könnte

            # Weitere Verarbeitung, um sorted_by_left, sorted_by_top, etc. zu ermitteln

            # IDs den Linien zuweisen
            for line_tuple in sorted_keys:  # Beispiel, um durch sortierte Schlüssel zu iterieren
                id_counter_str = str(id_counter).zfill(2)  # Führende Nullen hinzufügen
                labels[line_tuple] = f"E{id_counter_str}"
                id_counter += 1

    return labels

json_path_square = None  # Global definiert
json_path = None  # Global definiert
def draw_connected_lines_with_ids_by_cell(image, lines, color='yellow', thickness=2, text_size=12):
    height, width, _ = image.shape
    cell_width, cell_height = width / 3, height / 3

    # Weise Linien Zellen zu
    cells = {(row, col): [] for row in range(3) for col in range(3)}
    for line in lines:
        row, col = assign_line_to_cell(line, cell_width, cell_height)
        cells[(row, col)].append(line)

    # Sortiere Linien innerhalb jeder Zelle und weise ihnen IDs zu
    # labels = sort_and_label_lines(cells, cell_width, cell_height, width, height)
    labeled_lines = sort_and_label_lines(cells, cell_width, cell_height, width, height)

    # print("Inhalt von labeled_lines:")
    # print(labeled_lines)


    # Erstelle ein neues Dictionary für die IDs mit Orientierungssuffixen anhand der Steigung der Linie
    labels = {}
    for line, label in labeled_lines.items():
        (x1, y1, x2, y2) = line  # Entpacke die Koordinaten aus dem Tupel
        slope = (y2 - y1) / (x2 - x1)
        orientation_suffix = "__v" if abs(slope) > 1 else "__h"
        labels[line] = label + orientation_suffix

###
    def print_line_coordinates_and_ids(labels):
        # print("ID\tStartpunkt\tEndpunkt von Funktion print_line_coordinates_and_ids:")
        edges_data = {}  # Wörterbuch zum Speichern der Linieninformationen für JSON
        for line, label in labels.items():
            # Koordinaten als Ganzzahlen extrahieren
            x1, y1, x2, y2 = map(int, line)
            # print(f"{label}\t({x1}, {y1})\t({x2}, {y2})")

            # Füge Linieninformationen dem Wörterbuch hinzu
            edges_data[label] = [[x1, y1], [x2, y2]]

            # Erstelle das finale JSON-Objekt der Linieninformationen
            json_data = {
                "type": "mtf",
                "roi_width": ROI_groesse,  # Diesen Wert anpassen, falls nötig (am Anfang des Skripts definiert)
                "edges": edges_data
            }
            # Bestimme den Pfad für die JSON-Datei
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            global json_path
            json_path = os.path.join(os.path.dirname(image_path), f"{base_name}_lines.json")

        # Speichere die Linieninformationen in der JSON-Datei
        with open(json_path, 'w') as json_file:
            json.dump(json_data, json_file, indent=6)

        print(f"Linienkoordinaten und IDs gespeichert in: {json_path}")

        # Rückgabe des Pfads zur JSON-Datei für den Aufruf der nächsten Funktion
        return json_path

    # print_line_coordinates_and_ids(labels)

    def load_and_print_json_content(json_path):
        # Öffne die JSON-Datei und lese den Inhalt
        with open(json_path, 'r') as file:
            json_content = json.load(file)

        # Gib den Inhalt der JSON-Datei im Terminal aus
        # print("Inhalt der JSON-Datei:")
        # print(json.dumps(json_content, indent=4))

        squares_data = {}

        for label, coordinates in json_content["edges"].items():
            # Berechne den Mittelpunkt der Linie
            x1, y1 = coordinates[0]
            x2, y2 = coordinates[1]
            mid_x = int((x1 + x2) / 2)
            mid_y = int((y1 + y2) / 2)

            # Berechne die Koordinaten der Eckpunkte des Quadrats
            half_side = int(QUADRAT_groesse / 2)  # Hälfte der Seitenlänge des Quadrats
            square_coordinates = [
                [mid_x - half_side, mid_y - half_side],  # Oben links
                [mid_x + half_side, mid_y - half_side],  # Oben rechts
                [mid_x + half_side, mid_y + half_side],  # Unten rechts
                [mid_x - half_side, mid_y + half_side]  # Unten links
            ]
            # Konvertiere alle Koordinaten in Integer
            square_coordinates = [[int(point[0]), int(point[1])] for point in square_coordinates]
            squares_data[label] = square_coordinates

        # Speichere die Quadratkoordinaten in der neuen JSON-Datei
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        global json_path_square
        json_path_square = os.path.join(os.path.dirname(json_path), f"{base_name}_square.json")

        with open(json_path_square, 'w') as file:
            json.dump({"squares": squares_data}, file, indent=4)

        print(f"Quadratkoordinaten und IDs gespeichert in: {json_path_square}")

        # Rückgabe des Pfads zur JSON-Datei für den Aufruf der nächsten Funktion
        return json_path_square

    # Rufe die Funktion print_line_coordinates_and_ids auf und erhalte den Pfad zur Linien-JSON-Datei damit das für die Quadrate genutzt werden kann
    json_path = print_line_coordinates_and_ids(labels)
    # Rufe die Funktion load_and_print_json_content auf, um den Inhalt der JSON-Datei zu laden und in richtung Quadrate weiter zu verarbeiten
    json_path_square = load_and_print_json_content(json_path)

    def visualize_lines_and_squares(image, json_path, json_path_square, color='yellow', thickness=1, text_size=9):
        # Lade die Bilddaten
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Lade Linieninformationen
        with open(json_path, 'r') as file:
            lines_data = json.load(file)['edges']

        # Lade Quadratinformationen
        with open(json_path_square, 'r') as file:
            squares_data = json.load(file)['squares']

        # Vorbereitung der Figur für das Zeichnen
        fig, ax = pp.subplots(figsize=(18, 10))
        ax.imshow(image_rgb)
        ax.axis('off')
        ax.set_title('Visualisierung von Linien und Quadraten über JSON-Quellen')

        # Zeichne Linien und beschrifte sie
        for label, coords in lines_data.items():
            x1, y1, x2, y2 = coords[0] + coords[1]
            ax.plot([x1, x2], [y1, y2], color=color, linewidth=thickness)
            ax.text((x1 + x2) / 2, (y1 + y2) / 2, label, color='red', fontsize=text_size, ha='center', va='center')

        # Zeichne Quadrate und beschrifte sie
        for label, square in squares_data.items():
            # Umwandeln in ein Format, das mit Matplotlib's plot funktioniert
            square_np = np.array(square, np.int32)
            square_np = np.append(square_np, [square_np[0]],
                                  axis=0)  # Füge den ersten Punkt am Ende hinzu, um das Quadrat zu schließen
            x, y = square_np.T
            ax.plot(x, y, color='cyan', linewidth=thickness)
            mid_x, mid_y = np.mean(square_np, axis=0)
            ax.text(mid_x, mid_y, label, color='green', fontsize=text_size, ha='center', va='center')

        # pp.show()
        pp.pause(2)  # Anzeige für 1 Sekunde pausieren
        pp.close()  # Fenster schließen

    # Visualisiere Linien und Quadrate
    visualize_lines_and_squares(image, json_path, json_path_square, color='yellow', thickness=1, text_size=9)

    '''
    pp.figure(figsize=(10, 10))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pp.imshow(image_rgb)
    pp.axis('off')
    pp.title('beschriftete Kanten')

    # Zeichne Linien und beschrifte sie mit IDs
    for line, label in labels.items():
        x1, y1, x2, y2 = line
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        pp.plot([x1, x2], [y1, y2], color=color, linewidth=thickness)
        pp.text(mid_x, mid_y, label, color='red', fontsize=text_size, ha='center', va='center')

    pp.show()
    # pp.pause(1)  # Anzeige für 1 Sekunde pausieren
    # pp.close()   # Fenster schließen
    '''

# Rufe die Funktion mit der frischen Kopie des Bildes und den verbundenen Linien auf
draw_connected_lines_with_ids_by_cell(image_fresh, connected_lines, color='yellow', thickness=1, text_size=9)

# ------------------------------ Ende connect und merge lines

#-------------------------------------------- ab hier NEU ----------------------------------------------------------

#-------------------------------------------- übergabe der koordinaten und des bildes an die bisherigen funktionen von  - slanted_edge_mtf.py ------------------------------------

#---------------------------------------------bisherige hilfsfunktionen (welche über dem darauf folgenden Funktionen stehen müssen) - von slanted_edge_mtf.py ------------------------------------
def imread_lum(filename):
    """
    Read image and convert RGB -> Luminance
    :param filename:
    :return: luminocity data
    Der Code führt eine lineare Transformation auf einem RGB-Bild aus.
    Dabei wird das Bild in ein Graustufenbild umgewandelt.
    Die Koeffizienten [0.2125, 0.7154, 0.0721] entsprechen den empirisch
    ermittelten Werten für die menschliche Wahrnehmung von Rot, Grün und Blau.
    Diese Werte geben an, wie stark die einzelnen Kanäle zur Graustufenintensität beitragen.
    Nach Anwendung der formel liegt ein Graustufenbild vor.
    """
    image = Image.open(str(filename))
    image = np.dot(image, [0.2125, 0.7154, 0.0721])  # RGB => Luminance
    return image


def norm_data(image, bottom=0.1, top=99.9):
    """
    Obtain normalization  image settings
    :param image:
    :return:
    """
    black = np.percentile(image, bottom)
    white = np.percentile(image, top)
    return black, white


def normalize(image, black=None, white=None):
    """
    Normalize luminocity image
    :param image:
    :param black:
    :param white:
    :return:
    """
    if black is None or white is None:
        black, white = norm_data(image)
    # black = np.percentile(image, 0.1)
    # white = np.percentile(image, 99.9)
    image = (image - black) / (white - black)
    image = np.clip(image, 0, 1)
    return image


def roiread(roi_img):
    """
    This function is similar to imread but processes only part of the image (roi). The goal is to increase speed.
    :param roi_img: roi image
    :return: processed roi image
    """
    image = np.dot(roi_img, [0.2125, 0.7154, 0.0721])
    image = image #/ maxval
    image = normalize(image)
    # image = sharpen(image) # wird auskommentiert
    return image


def otsu(image):
    # Otsu's binary thresholding
    image = cv2.GaussianBlur(image, (5, 5), 0)  # simple noise removal
    image = (image * 255).astype(np.uint8)      # [0, 1] => [0, 255]
    # image = (image).astype(np.uint8)
    otsu_thr, otsu_map = cv2.threshold(image, 0, 255, cv2.THRESH_OTSU)
    return otsu_map


def morpho(mask):
    # morphological filtering of binary mask: 3 x (erosion + dilation)
    structure = np.ones((3,3))  # 8-connected structure
    # zeug aus dem paket ndimage ist nicht mehr empfohlen # mask = scipy.ndimage.morphology.binary_opening(mask, structure, iterations=3)
    # deshalb paket binary_opening aus skimage
    mask = binary_opening(mask, structure, iterations=3)
    return mask


def canny(image):
    # Canny edge detection
    # image = (image * 255).astype(np.uint8)  # [0, 1] => [0, 255]
    image = (image).astype(np.uint8)
    edge_map = cv2.Canny(image, image.min(), image.max(), apertureSize=3, L2gradient=True)
    return edge_map


def fft(lsf):
    # FFT of line spread function
    fft = np.fft.fft(lsf, 1024)  # even 256 would be enough
    fft = fft[:len(fft) // 2]    # drop duplicate half
    fft = np.abs(fft)            # |a + bi| = sqrt(a² + b²)
    fft = fft / fft.max()        # normalize to [0, 1]
    return fft

#------------------------------------------------ ENDE bisherige hilfsfunktionen - slanted_edge_mtf.py ------------------------------------

# ------------------------------ BEGINN SPEICHERUNG ROI-BILDER --------------------------------
# es müssen alle ROI in einen eindeutigen ordener gespeichert werden damit nur die ROI als bilder für die MTF-nerechnung genutzt werden können
# eine verarbeitung des gesamten targets zur vorbereitung der MTF berechnung ist nicht zielführend da für die statistischen sauberkeit wirklich
# nur die ROI genutzt werden dürfen

# hier wird ein eindeutiger name für den ordner generiert für die speicherung der ROI bilder generiert
def generate_unique_suffix():
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    random_number = random.randint(100, 999)
    return f"_{timestamp}_{random_number}"

# hier werden die roi aus dem target-bild über ihre koordinaten extrahiert und der eindeutige ordnername wird generiert
def show_roi_images(image_path, json_path_square):
    try:
        # Generiere ein eindeutiges Suffix
        unique_suffix = generate_unique_suffix()

        # Ordnername für temporäre Dateien
        temp_dir_name = f"roi_temp{unique_suffix}"

        # Bestimme den Pfad zum aktuellen Arbeitsverzeichnis
        current_directory = os.getcwd()

        # Erstelle den Pfad zum temporären Verzeichnis als Unterverzeichnis des aktuellen Arbeitsverzeichnisses
        temp_dir = os.path.join(current_directory, temp_dir_name)

        # Überprüfen, ob der temporäre Ordner bereits existiert und ggf. löschen
        if os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir)

        # Erstelle das temporäre Verzeichnis
        os.makedirs(temp_dir)

        # Überprüfen, ob das Bild vorhanden ist
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Bild '{image_path}' wurde nicht gefunden.")

        # Überprüfen, ob die JSON-Datei vorhanden ist
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON-Datei '{json_path_square}' wurde nicht gefunden.")

        # Bild laden
        image = cv2.imread(image_path)

        # Quadrat-Koordinaten aus der JSON-Datei lesen und nach ID sortieren
        with open(json_path_square, 'r') as file:
            squares_data = json.load(file)["squares"]

        roi_coordinates = []

        for roi_id, coords in squares_data.items():
            # Wir nehmen an, dass die Koordinaten im Uhrzeigersinn gespeichert sind und starten bei Top-Left
            # [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
            # Für die ROI-Extraktion benötigen wir nur Top-Left (index 0) und Bottom-Right (index 2)
            top_left = coords[0]
            bottom_right = coords[2]

            # Da die Koordinaten als Liste von Listen gespeichert sind, können wir sie direkt verwenden
            x1, y1 = top_left
            x2, y2 = bottom_right

            # Speichern der Koordinaten zusammen mit der ID
            roi_coordinates.append((x1, y1, x2, y2, roi_id))

        # Sortiere die roi_coordinates basierend auf der ID
        roi_coordinates.sort(key=lambda x: x[4])

        # Liste zum Speichern der ROI-Bilder und ihrer IDs
        roi_images = []
        roi_ids = []

        # ROIs extrahieren und Bilder speichern
        for roi_id, coords in squares_data.items():
            x1, y1 = coords[0]
            x2, y2 = coords[2]  # Wir nutzen die gegenüberliegenden Ecken für die ROI-Extraktion
            roi = image[y1:y2, x1:x2]  # ROI aus dem Bild extrahieren
            roi_images.append(roi)
            roi_ids.append(roi_id)

        # Größe des gemeinsamen Fensters berechnen
        window_height = 0
        window_width = 0
        for roi_coord in roi_coordinates:
            x1, y1, x2, y2, _ = roi_coord
            window_height = max(window_height, y2 - y1)
            window_width = max(window_width, x2 - x1)

        # Größeres Bild erstellen, um alle ROIs aufzunehmen
        combined_image = np.zeros((window_height * 6, window_width * 6, 3), dtype=np.uint8)

        # ROIs in das größere Bild einfügen und Namen anzeigen
        row_index = 0
        col_index = 0
        for roi_coord, roi_image in zip(roi_coordinates, roi_images):
            x1, y1, x2, y2, roi_id = roi_coord
            combined_image[row_index:row_index + (y2 - y1), col_index:col_index + (x2 - x1)] = roi_image

            # Name der ROI über dem Bild anzeigen
            cv2.putText(combined_image, roi_id, (col_index + 10, row_index + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            col_index += window_width
            if col_index >= window_width * 6:
                col_index = 0
                row_index += window_height

        # Größeres Bild anzeigen
        cv2.imshow("Combined ROIs from JSON-File", combined_image)
        cv2.waitKey(1000)
        # cv2.waitKey()
        cv2.destroyAllWindows()

        return temp_dir, roi_images, roi_ids  # ROI-Bilder, ihre IDs und temporärer Ordner zurückgeben


    except FileNotFoundError as e:
        print(e)
    except json.JSONDecodeError:
        print("Ungültige JSON-Datei.")
    except Exception as e:
        print(f"Fehler beim Verarbeiten des Bildes und der JSON-Datei: {str(e)}")

# ROI-Bilder von show_roi_images() erhalten
temp_dir, roi_images, roi_ids = show_roi_images(image_path, json_path_square)


# hier werden die ROI bilder in einem temporären ordner gespeichert
def save_roi_temp(roi_images, roi_ids, temp_dir):
    try:
        # Speichere die ROI-Bilder im temporären Verzeichnis
        for idx, (roi_image, roi_id) in enumerate(zip(roi_images, roi_ids)):
            roi_path = os.path.join(temp_dir, f"{roi_id}.png")
            roi_pil = Image.fromarray(roi_image)  # PIL-Image aus dem Numpy-Array erstellen
            roi_pil.save(roi_path)  # ROI-Bild mit PIL speichern

        print("ROI-Bilder wurden temporär gespeichert in:", temp_dir)
        return temp_dir

    except Exception as e:
        print("Fehler beim Speichern der ROI-Bilder:", str(e))
        return None

# ROI-Bilder an save_roi_temp() übergeben
temp_dir = save_roi_temp(roi_images, roi_ids, temp_dir)

# ------------------------------ ENDE SPEICHERUNG ROI-BILDER --------------------------------


def process_images(temp_dir):
    try:
        # Überprüfen, ob der Ordner existiert
        if not os.path.exists(temp_dir):
            raise FileNotFoundError(f"Der Ordner '{temp_dir}' existiert nicht.")

        # Liste aller Dateien im Ordner temp_dir
        files = os.listdir(temp_dir)

        # Bearbeite jedes Bild im Ordner
        for file in files:
            if file.endswith(".png"):
                # Bildpfad erstellen
                file_path = os.path.join(temp_dir, file)

                # Luminanz bearbeiten
                lum_image = imread_lum(file_path)

                # Anwendung der Gammakorrektur (optional)
                lum_image= exposure.adjust_gamma(lum_image, gamma=1.0)  # gamma=1.0 bedeutet keine Änderung

                # Normalisiere das Bild basierend auf den 1. und 99. Perzentilen
                # v_min, v_max = np.percentile(gamma_corrected, (0.1, 99.9))
                # normalisiert = exposure.rescale_intensity(gamma_corrected, in_range=(v_min, v_max))

                # setze den schwarz und weisspunkt auf 1-99% der Werte
                black = int(np.percentile(lum_image, 1)) # default
                white = int(np.percentile(lum_image, 99)) # default
                # print('black: ', black)
                # print('white: ', white)
                
                # normalisiere das bild
                normalisiert = ((lum_image - black) / (white - black)) * 255
                # normalisiert = np.clip(lum_image, 0, 1)
                # Sicherstellen, dass die Werte im gültigen Bereich liegen und NaNs ersetzen
                normalisiert = np.clip(normalisiert, 0, 255)
                normalisiert = np.nan_to_num(normalisiert, nan=0.0)


                # um die salz und pfeffer pixel zu reduzieren - optionales anwenden des Median-Filters auf das normalisierte Bild
                # normalisiert = median_filter(normalisiert, size=3)

                # Bild mit Präfix "normalisiert__" speichern
                output_path = os.path.join(temp_dir, f"normalisiert__{file}")
                # normalisierung nicht übergehen
                #lum_pil = Image.fromarray(normalisiert.astype(np.uint8))  # Konvertierung für PIL
                # normalisierung übergehen
                lum_pil = Image.fromarray(lum_image.astype(np.uint8))  # Konvertierung für PIL
                lum_pil.save(output_path)  # Speichern mit PIL

                # Bild anzeigen
                # cv2.imshow("Luminanzbild", lum_image)
                # cv2.waitKey(1000)  # Anzeige für 1 Sekunde
                # cv2.destroyAllWindows()

        print("Alle normlisierten Kantenbilder wurden erfolgreich bearbeitet und gespeichert.")

        # ---------- Speichern der normalisierten Dateinamen in Array, iteritieren durch das Array und Ausgabe --------
        normalisierte_files = [file for file in os.listdir(temp_dir) if
                          file.startswith("normalisiert__") and (file.endswith("__v.png") or file.endswith("__h.png"))]
        normalisierte_files.sort()  # Sortiert die Liste alphabetisch aufsteigend nach den Namen der Dateien

        '''
        for normalis_file in normalisierte_files:
            print("Datei:", temp_dir + '/' + normalis_file)
            '''

        print("Anzahl der normalisierten Dateien:", len(normalisierte_files))
        # -- ENDE -- Speichern der normalisierten Dateinamen in Array, iteritieren durch das Array und Ausgabe --------

    except Exception as e:
        print(f"Fehler beim Bearbeiten der Bilder: {str(e)}")

# Beispielaufruf
process_images(temp_dir)

# ---untersuche_rauschen_und_zeige(temp_dir)--------wegen fehlendem skimage-paket auf macbook auskommentiert
'''
def untersuche_rauschen_und_zeige(temp_dir):
    noise_values = []
    mean_values = []
    ids = []

    files = os.listdir(temp_dir)

    for file in files:
        if file.startswith("normalisiert__") and file.endswith("__h.png"):
            file_path = os.path.join(temp_dir, file)
            image = io.imread(file_path, as_gray=True)

            h, w = image.shape
            third = h // 3

            # Bestimme die ROIs und deren Durchschnittswerte
            roi1 = image[:third, :]
            roi2 = image[-third:, :]
            mean_roi1 = np.mean(roi1)
            mean_roi2 = np.mean(roi2)

            noise_roi1 = estimate_sigma(roi1)
            noise_roi2 = estimate_sigma(roi2)
            noise_value = (noise_roi1 + noise_roi2) / 2
            mean_value = (mean_roi1 + mean_roi2) / 2

            # Speichere die Werte in den Listen
            noise_values.append(noise_value)
            mean_values.append(mean_value)
            ids.append(file.split('__')[1])

    # Erstelle Liniendiagramme
    pp.figure(figsize=(15, 5))
    pp.subplot(1, 2, 1)
    pp.plot(ids, noise_values, label='Noise', marker='o')
    pp.title('Rauschwerte pro ID')
    pp.xlabel('ID')
    pp.ylabel('Rauschen')
    pp.legend()

    pp.subplot(1, 2, 2)
    pp.plot(ids, mean_values, label='Grauwert', marker='o')
    pp.title('Durchschnittliche Grauwerte pro ID')
    pp.xlabel('ID')
    pp.ylabel('Grauwert')
    pp.legend()

    pp.tight_layout()
    # pp.show()
    pp.pause(0.5)  # Anzeige für 1 Sekunde pausieren
    pp.close()  # Fenster schließen

    # Gib die durchschnittlichen Werte für Rauschen und Grauwerte aus
    if noise_values:
        durchschnitts_noise = np.mean(noise_values)
        durchschnitts_mean = np.mean(mean_values)
        print(f"Durchschnittliches Rauschen: {durchschnitts_noise:.5f}")
        print(f"Durchschnittlicher Grauwert: {durchschnitts_mean:.5f}")
    else:
        print("Keine Bilder gefunden.")

# Funktion aufrufen mit dem Pfad, in dem sich die Bilder befinden
untersuche_rauschen_und_zeige(temp_dir)
'''


'''
def untersuche_rauschen_und_helligkeit(temp_dir):
    noise_values = []
    mean_values = []

    # Liste aller Dateien im Verzeichnis
    files = os.listdir(temp_dir)

    # Für die Darstellung der Ergebnisse
    fig, axes = pp.subplots(nrows=2, ncols=len(files), figsize=(20, 10))

    for i, file in enumerate(files):
        if file.startswith("normalisiert__") and file.endswith("__v.png"):
            file_path = os.path.join(temp_dir, file)
            image = io.imread(file_path, as_gray=True)
            image = img_as_float(image)  # Konvertiere in float für die Analyse

            # Berechne das Rauschen und die Helligkeit im ersten und letzten Drittel des Bildes
            h, w = image.shape
            third = h // 3
            roi1 = image[:third, :]  # erstes Drittel
            roi2 = image[-third:, :]  # letztes Drittel

            # Rauschen bestimmen
            noise_roi1 = estimate_sigma(roi1)
            noise_roi2 = estimate_sigma(roi2)
            noise_value = (noise_roi1 + noise_roi2) / 2

            # Durchschnittliche Helligkeit bestimmen
            mean_roi1 = np.mean(roi1)
            mean_roi2 = np.mean(roi2)

            # Ergebnisse speichern
            noise_values.append(noise_value)
            mean_values.append((mean_roi1, mean_roi2))

            # Visuelle Darstellung der ROIs und ihrer Werte
            ax1 = axes[0, i]
            ax2 = axes[1, i]
            ax1.imshow(image, cmap='gray')
            ax2.imshow(image, cmap='gray')
            ax1.add_patch(Rectangle((0, 0), w, third, linewidth=2, edgecolor='r', facecolor='none'))
            ax2.add_patch(Rectangle((0, h - third), w, third, linewidth=2, edgecolor='b', facecolor='none'))
            ax1.text(5, 10, f'ID: {file}\nNoise: {noise_value:.5f}\nMean: {mean_roi1:.5f}', color='yellow')
            ax2.text(5, h - third + 10, f'ID: {file}\nNoise: {noise_value:.5f}\nMean: {mean_roi2:.5f}', color='yellow')

    # Anzeigen der Diagramme
    pp.tight_layout()
    pp.show()

    # Durchschnittswerte für Rauschen und Helligkeit berechnen und ausgeben
    avg_noise = np.mean(noise_values)
    avg_mean1 = np.mean([mean[0] for mean in mean_values])
    avg_mean2 = np.mean([mean[1] for mean in mean_values])
    print(f'Durchschnittliches Rauschen: {avg_noise:.5f}')
    print(f'Durchschnittliche Helligkeit ROI1: {avg_mean1:.5f}')
    print(f'Durchschnittliche Helligkeit ROI2: {avg_mean2:.5f}')

    # Plot der Rausch- und Helligkeitswerte in Liniendiagrammen
    ids = [file.split('__')[1] for file in files if file.startswith("normalisiert__") and file.endswith("__v.png")]
    pp.figure(figsize=(10, 5))
    pp.plot(ids, [val[0] for val in noise_values], label='Noise ROI1')
    pp.plot(ids, [val[1] for val in noise_values], label='Noise ROI2')
    pp.plot(ids, [val[0] for val in mean_values], label='Mean ROI1')
    pp.plot(ids, [val[1] for val in mean_values], label='Mean ROI2')
    pp.xlabel('ID')
    pp.ylabel('Value')
    pp.title('Noise and Mean Brightness Values for ROIs')
    pp.legend()
    pp.show()

# Aufrufen der Funktion mit dem entsprechenden Verzeichnispfad
untersuche_rauschen_und_helligkeit(temp_dir)
'''

def berechnung_mtf(temp_dir):
    # return in case of an error
    err = 0, 0, 0, 0, 0
    results = []  # Sammeln von Ergebnissen für alle Dateien
    edge_width = EDGE_groesse # Breite des Randes, der für die MTF-Berechnung verwendet wird (Am Anfang des Skripts definiert)
    # edge_width = 61

    # Liste aller Dateien im Ordner temp_dir
    files = os.listdir(temp_dir)

    # Bearbeite jedes Bild im Ordner
    for file in files:
        if file.startswith("normalisiert__") and (file.endswith("__v.png") or file.endswith("__h.png")):
        # if file.startswith("normalisiert__") and (file.endswith("E20__h.png") or file.endswith("E11__v.png")):
            print("Datei --------------- :", file)  # Ausgabe des Dateinamens

            # Extracting individual ID from filename
            individual_id = file.split('__')[1]

            # Bildpfad erstellen
            file_path = os.path.join(temp_dir, file)

            if file.endswith('v.png'):
                orient = 'v'
            elif file.endswith('h.png'):
                orient = 'h'
            else:
                return None
            print('file_path: ', file_path)

            try:
                # Bild lesen mit matplotlib als numpy array - weil als cv2 irgendwie nicht möglich
                image = pp.imread(file_path)
                if image is None:
                    raise FileNotFoundError(f"Fehler beim Lesen des Bildes: {file_path}")

                # Region entsprechend der Orientierung anpassen
                region = image[:, :]
                if orient == 'h':
                    region = np.swapaxes(region, 0, 1)

                roih, roiw = region.shape
                # print('roih: ', roih)
                # print('roiw: ', roiw)
                # print('orient: ', orient)

                # Führen Sie hier Ihre weiteren Berechnungen durch
                # detect edge pixels
                otsu_map = otsu(region)  # generate binary mask: 0=black, 1=white
                otsu_filt = morpho(otsu_map)  # filter out small non-contiguous regions
                otsu_edges = canny(otsu_filt)  # detect edges; there should be only one
                edge_coords = np.nonzero(otsu_edges)  # get (x, y) coordinates of edge pixels

                # Visualisierung der Ergebnisse ------ zum debuggen von otsu und canny ----------------
                '''
                fig, ax = pp.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(region, cmap='gray')
                ax[0].set_title(f'Original Region {individual_id} {orient}')
                ax[0].axis('on')

                ax[1].imshow(otsu_map, cmap='gray')
                ax[1].set_title(f'Otsu Binarization {individual_id} {orient}')
                ax[1].axis('on')

                ax[2].imshow(otsu_edges, cmap='gray')
                ax[2].set_title(f'Canny Edges {individual_id} {orient}')
                ax[2].axis('on')

                pp.tight_layout()
                pp.show()
                # pp.pause(2)  # Anzeige für 1 Sekunde pausieren
                # pp.close()  # Fenster schließen
                '''
                # ENDE Visualisierung der Ergebnisse ------ zum debuggen von otsu und canny ----------------


                # fit a straight line through the detected edge
                edge_coeffs = np.polyfit(edge_coords[0], edge_coords[1], 1)
                print('edge_coeffs: ', edge_coeffs)

                edge_angle = np.abs(np.rad2deg(np.arctan(edge_coeffs[0])))
                print('edge_angle: ', edge_angle)
                # TODO: verify angle is within limits

                p = np.poly1d(edge_coeffs)
                y_positions = np.arange(0, region.shape[0])
                x_positions = np.clip(p(y_positions).astype(int), 0, region.shape[1] - 1)

                # An dieser Stelle erstellen wir eine neue Matrix für edge_straight
                edge_straight = np.zeros((len(x_positions), edge_width))
                for idx, x in enumerate(x_positions):
                    x_range = np.clip(np.arange(x - edge_width // 2, x + edge_width // 2 + 1), 0, roiw - 1)
                    edge_straight[idx, :] = region[y_positions[idx], x_range]

                # ----------------------------Visualisierung der extrahierten Linie auf dem Originalbild------------------------
                '''
                edge_angle_anzeige = np.round(edge_angle, 2)
                pp.figure(figsize=(18, 10))
                pp.subplot(1, 2, 1)
                pp.imshow(region, cmap='gray')
                pp.plot(x_positions, y_positions, 'r-', linewidth=1)  # Zeichnen der Linie in Rot
                pp.title(f'Extrahierte Kante mit berechneter Linie - {individual_id}__{orient} | {edge_width}px | {edge_angle_anzeige}° | {roih}x{roiw} {y_positions[idx]}')

                # Berechne die durchschnittliche x-Position der Kante im Originalbild
                avg_x_position = np.mean(x_positions)

                pp.subplot(1, 2, 2)
                pp.imshow(edge_straight, cmap='gray')
                pp.axvline(x=avg_x_position, color='r', linestyle='-', linewidth=1) # Zeichnen der durchschnittlichen x-Position
                pp.title(f'Vertikale Kante mit berechneter Linie - {individual_id}__{orient}')
                pp.show()
                '''
                # ----------------------------ENDE Visualisierung der extrahierten Linie auf dem Originalbild------------------------



                if edge_straight is not None:
                    # Visualisieren des 'edge_straight'-Bildes zum Debuggen -----------------------
                    '''
                    pp.figure()
                    pp.imshow(edge_straight, cmap='gray')
                    pp.title(f'Edge Straight - {individual_id}')
                    pp.axis('on')
                    pp.show()
                    pp.pause(0.5)  # Anzeige für 1 Sekunde pausieren
                    pp.close()  # Fenster schließen
                    '''
                    # ENDE Visualisieren des 'edge_straight'-Bildes zum Debuggen -----------------------

                    # compute Edge Spread Function (ESF), Line Spread Function (LSF), and filtered LSF
                    edge = edge_straight
                    #-------------- einbau von sobel vor esf, ist irgendwie blöd weil erzeugt einen sinus-graphen / AUSSSERDEM ist nicht klar, ob sobel überhaupt sinnvoll ist--------------------
                    sobelx = cv2.Sobel(edge, cv2.CV_64F, 1, 0, ksize=3)  # x-Richtung
                    sobely = cv2.Sobel(edge, cv2.CV_64F, 0, 1, ksize=3)  # y-Richtung
                    # Berechnung der Magnitude des Gradienten
                    gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
                    #------------- bei verzicht auf sobel wird in der nächsten zeile edge statt gradient_magnitude verwendet
                    # esf = np.mean(gradient_magnitude, axis=0)
                    esf = np.mean(edge, axis=0)

                    # die funktionalität rising_edge ----------------------------------:
                    # Bestimme die 10% und 90% Werte der maximalen ESF
                    # es kann sein dass 0.1 bis 0.9 nicht mehr genug varianz in der lsf lässt mum fft zu berechnen und es dann einen fehler gibt
                    # dann vielleicht besser nicht mit esf_rising sondern mit esf weiterarbeiten oder anstelle von 0.1 und 0.9 mit 0 und 1 rechnen
                    esf_min = np.min(esf)
                    esf_max = np.max(esf)
                    esf_range = esf_max - esf_min
                    # lower_bound = esf_min + 0.01 * esf_range
                    # upper_bound = esf_min + 0.99 * esf_range
                    lower_bound = esf_min + 0 * esf_range
                    upper_bound = esf_min + 1 * esf_range
                    # Finde die Indizes, die den Bereich zwischen den Grenzen einschließen
                    indices = np.where((esf >= lower_bound) & (esf <= upper_bound))[0]

                    if len(indices) < 2:
                        # Nicht genug Punkte für eine sinnvolle Analyse
                        return None, None
                    start_index, end_index = indices[0], indices[-1]
                    # Schneide das ESF auf den Bereich mit steigender Kante
                    esf_rising = esf[start_index:end_index + 1]
                    # ENDE funktionalität rising_edge ----------------------------------:

                    # bei fehler von fft hier mit esf statt esf_rising weiterarbeiten

                    esf_rising = scipy.signal.wiener(esf_rising, 5)[3:-3]
                    # esf = scipy.signal.wiener(esf, 5)[3:-3]

                    lsf = np.gradient(esf_rising)[1:]
                    # lsf = np.gradient(esf)[1:]

                    lsfs = scipy.signal.wiener(lsf, 7)[4:-4] # default
                    # lsfs = scipy.signal.wiener(lsf, 11)[6:-6] # mehr glättung
                    # compute filtered & unfiltered MTF
                    mtf = fft(lsf)
                    mtfs = fft(lsfs)

                    print('len(esf)----------------: ', len(esf))
                    print('len(indices)------------: ', len(indices))
                    print('len(lsf)----------------: ', len(lsf))
                    print('len(lsfs)---------------: ', len(lsfs))
                    print('len(mtf)----------------: ', len(mtf))
                    print('len(mtfs)---------------: ', len(mtfs))

                    # compute MTF50 & MTF20 from filtered MTF
                    x_mtf = np.linspace(0, 1, len(mtf))
                    mtf50 = np.interp(0.5, mtfs[::-1], x_mtf[::-1])
                    mtf20 = np.interp(0.2, mtfs[::-1], x_mtf[::-1])
                    global mtf10  # toba
                    mtf10 = np.interp(0.1, mtfs[::-1], x_mtf[::-1])  # toba

                    lsfmax = np.max(lsf) * 1000  # toba
                    lsfmin = np.min(lsf) * 1000  # toba
                    # (mit lsf fuehrts zu plausiblen werten) contrast modulation: http://www.quickmtf.com/about-resolution.html
                    global lsfunc
                    lsfunc = (lsfmax - lsfmin) / (lsfmax + lsfmin)  # toba

                    results.append((esf, lsf, lsfs, mtf, mtfs, edge, edge_angle, x_mtf, mtf50, mtf20, mtf10, lsfmax, lsfmin, lsfunc, individual_id))

                    # return esf, lsf, lsfs, mtf, mtfs, edge, edge_angle, x_mtf, mtf50, mtf20, mtf10, lsfmax, lsfmin, lsfunc, individual_id, results

                # return None

            except Exception as e:
                print(f"Fehler beim Lesen des Bildes: {str(e)}")

    return results

results = berechnung_mtf(temp_dir)


def plot_mtf_and_esf(temp_dir, edge, individual_id, esf, lsf, lsfs, mtf, mtfs, edge_angle, x_mtf, mtf50, mtf20, mtf10, lsfmax, lsfmin, lsfunc):
    # Aktuelles Datum und Uhrzeit erhalten
    current_time = datetime.datetime.now()
    # Formatieren des Datums und der Uhrzeit als String
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

    # Plot ESF LSF MTF
    # Plot ESF
    pp.figure(figsize=(20, 5), dpi=110)
    pp.subplot(1, 4, 1)
    pp.plot(esf, label='ESF')
    pp.title(f'ESF - {individual_id}')
    pp.xlabel('Pixel Position')
    pp.ylabel('Intensity')
    pp.grid(True)
    pp.legend()

    # Plot LSF
    pp.subplot(1, 4, 2)
    pp.plot(lsf, label='LSF')
    pp.plot(lsfs, label='LSF Filtered')
    pp.title(f'LSF - {individual_id}')
    pp.xlabel('Pixel Position')
    pp.ylabel('Intensity')
    pp.grid(True)
    pp.legend()

    # Plot MTF
    pp.subplot(1, 4, 3)
    pp.plot(x_mtf, mtf, label='MTF')
    pp.plot(x_mtf, mtfs, label='MTF Filtered')
    pp.title(f'MTF - {individual_id}')
    pp.axhline(y=0.5, color='r', linestyle='--', label='MTF50')
    pp.axhline(y=0.2, color='g', linestyle='--', label='MTF20')
    pp.axhline(y=0.1, color='b', linestyle='--', label='MTF10')
    pp.xlabel('Frequency (cycles/pixel)')
    pp.ylabel('MTF')
    pp.grid(True)
    pp.legend()

    # Plot image
    pp.subplot(1, 4, 4)
    pp.imshow(edge, cmap='gray')
    pp.title(f'Edge Image - {individual_id} /ROI: {ROI_groesse}px')
    pp.axis('on')  # Turn off axis numbers and ticks
    pp.grid(False)

    # Achseneinteilung anpassen
    # Anzeige von Datum und Uhrzeit im Plot, oben links
    pp.figtext(0.01, 0.99, f'Aktualisiert am: {time_str} - {os.path.basename(image_path)}', ha="left", va="top", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    #---------sichern der ansicht ----------
    esf_mtf_filename = os.path.join(temp_dir, f"esf_lsf_mtf_{individual_id}.png")
    pp.savefig(esf_mtf_filename)
    # -Ende----sichern der ansicht ----------
    # pp.show()
    # pp.pause(0.9)  # Anzeige für 1 Sekunde pausieren
    pp.close()


    # Plot additional information
    '''
    pp.figure(figsize=(10, 5), dpi=110)
    pp.plot(x_mtf, mtfs, label='MTF Filtered')
    pp.axhline(y=0.5, color='r', linestyle='--', label='MTF50')
    pp.axhline(y=0.2, color='g', linestyle='--', label='MTF20')
    pp.axhline(y=0.1, color='b', linestyle='--', label='MTF10')
    pp.title(f'Modulation Transfer Function (MTF) - {individual_id}')
    pp.xlabel('Frequency (cycles/pixel)')
    pp.ylabel('MTF')
    pp.grid(True)
    pp.legend()
    # pp.show()
    # pp.pause(0.5)  # Anzeige für 1 Sekunde pausieren
    # pp.close()  # Fenster schließen
    '''
    print(individual_id)
    print(os.path.basename(image_path))
    print("MTF50  :", np.round(mtf50, 3))
    print("MTF20  :", np.round(mtf20, 3))
    print("MTF10  :", np.round(mtf10, 3))
    print("LSF Max:", np.round(lsfmax, 3))
    print("LSF Min:", np.round(lsfmin, 3))
    print("ROI    :", ROI_groesse)
    print("Contrast Modulation (LSF):", np.round(lsfunc, 4))
    print("Edge Angle:", np.round(edge_angle, 3))
    print("-------------------------------------------------")

if results:
    for result in results:
        # Entpacken Sie jedes Ergebnis und rufen Sie plot_mtf_and_esf auf
        esf, lsf, lsfs, mtf, mtfs, edge, edge_angle, x_mtf, mtf50, mtf20, mtf10, lsfmax, lsfmin, lsfunc, individual_id = result

        plot_mtf_and_esf(temp_dir, edge, individual_id, esf, lsf, lsfs, mtf, mtfs, edge_angle, x_mtf, mtf50, mtf20, mtf10, lsfmax, lsfmin, lsfunc)

else:
    print("Keine Ergebnisse gefunden.")


def plot_all_mtfs(results, **kwargs):
    sorted_results = sorted(results, key=lambda x: x[14])

    # Aktuelles Datum und Uhrzeit erhalten
    current_time = datetime.datetime.now()
    # Formatieren des Datums und der Uhrzeit als String
    time_str = current_time.strftime('%Y-%m-%d %H:%M:%S')

    pp.figure(figsize=(18, 10), dpi=110)
    # Für jede Messung in den Ergebnissen
    pp.grid(linestyle=":")
    for result in sorted_results:
        _, _, _, mtf, mtfs, _, _, x_mtf, mtf50, mtf20, mtf10, _, _, _, individual_id = result
        mtfs_rounded = np.round(mtfs, 3)
        mtf10_rounded = np.round(mtf10, 3)
        mtf20_rounded = np.round(mtf20, 3)
        mtf50_rounded = np.round(mtf50, 3)
        # Plot der gefilterten MTF-Werte für jede ID
        pp.plot(x_mtf, mtfs, label=f'ID: {individual_id} /mtf50: {mtf50_rounded} /mtf20: {mtf20_rounded} /mtf10: {mtf10_rounded}')
        pp.plot(x_mtf, mtf, **kwargs, color='lightgray', linewidth=1)
        pp.plot(0.80, 0.10, color='green', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=5)
    pp.axvline(x=0.5, linestyle=":", linewidth=0.5, color="red")
    pp.axvline(x=0.8, linestyle=":", linewidth=0.5, color="red")
    pp.axhline(y=0.5, linestyle=":", linewidth=0.5, color="r", label='MTF50')
    pp.axhline(y=0.2, linestyle=":", linewidth=0.5, color="g", label='MTF20')
    pp.axhline(y=0.1, linestyle=":", linewidth=0.5, color="b", label='MTF10')
    pp.text(0.505, 0.75, "Nyquist limit", color="red", rotation="vertical")
    pp.text(0.650, 0.51, "MTF50", color="red")
    pp.text(0.650, 0.11, "MTF10", color="red")
    kwargs["linestyle"] = "--"
    pp.title(f'Alle MTFs')
    pp.xlabel('Frequency (cycles/pixel)')
    pp.ylabel('MTF')
    # Anzeige von Datum und Uhrzeit im Plot, oben links
    pp.figtext(0.01, 0.99, f'Aktualisiert am: {time_str} - {os.path.basename(image_path)}', ha="left", va="top", fontsize=10, bbox={"facecolor": "orange", "alpha": 0.5, "pad": 5})
    pp.legend(loc='upper left', bbox_to_anchor=(1, 1.01))
    pp.subplots_adjust(right=0.75)
    # ---------sichern der ansicht ----------
    esf_mtf_filename1 = os.path.join(temp_dir, f"all_mtf_{individual_id}.png")
    pp.savefig(esf_mtf_filename1)
    # -Ende----sichern der ansicht ----------
    pp.show()
    # pp.pause(10)  # Anzeige für 1 Sekunde pausieren
    # pp.close()  # Fenster schließen

# Annahme: `results` ist die Liste der Ergebnisse, die von `berechnung_mtf(temp_dir)` zurückgegeben wurde
plot_all_mtfs(results)


def calculate_averages(results):
        mtf50_values = np.array([result[8] for result in results])
        mtf20_values = np.array([result[9] for result in results])
        mtf10_values = np.array([result[10] for result in results])
        lsfunc_values = np.array([abs(result[13]) for result in results])

        print(f'Durchschnittswerte: für alle 36 IDs des Targets: {os.path.basename(image_path)}')
        print(f"MTF50 :  {np.round(np.mean(mtf50_values), 3)} | Min: {np.round(np.min(mtf50_values), 3)}  Max: {np.round(np.max(mtf50_values), 3)}  Std: {np.round(np.std(mtf50_values), 3)}")
        print(f"MTF20 :  {np.round(np.mean(mtf20_values), 3)} | Min: {np.round(np.min(mtf20_values), 3)}  Max: {np.round(np.max(mtf20_values), 3)}  Std: {np.round(np.std(mtf20_values), 3)}")
        print(f"MTF10 :  {np.round(np.mean(mtf10_values), 3)} | Min: {np.round(np.min(mtf10_values), 3)}  Max: {np.round(np.max(mtf10_values), 3)}  Std: {np.round(np.std(mtf10_values), 3)}")
        print(f"LSFUNC: {np.round(np.mean(lsfunc_values), 4)} | Min: {np.round(np.min(lsfunc_values), 3)}  Max: {np.round(np.max(lsfunc_values), 3)}  Std: {np.round(np.std(lsfunc_values), 3)}")
        print('---------------------------------------------------')
        print('BTW: Die Standardabweichung ist ein Maß für die Streuung oder Variabilität von Datenpunkten in einer Datensammlung im Vergleich\nzum Durchschnitt (Mittelwert) dieser Daten. Sie gibt an, wie weit die einzelnen Datenpunkte im Durchschnitt vom Mittelwert entfernt liegen.\nEine geringe Standardabweichung bedeutet, dass die Datenpunkte im Allgemeinen nah am Mittelwert liegen, während eine hohe Standardabweichung\ndarauf hinweist, dass die Datenpunkte über einen breiteren Bereich verteilt sind.')

calculate_averages(results)