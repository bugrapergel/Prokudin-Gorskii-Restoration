import cv2
import numpy as np

def load_image(path):
    """Görüntüyü gri tonlamada yükler."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"{path} bulunamadı.")
    return img

def split_image(image):
    """Gri fotoğrafı üçe böler: üst=B, orta=G, alt=R."""
    height = image.shape[0]
    third = height // 3
    b = image[0:third, :]
    g = image[third:2*third, :]
    r = image[2*third:3*third, :]
    return b, g, r

def create_color_image(b, g, r):
    """Üç kanalı BGR renkli görüntü olarak birleştirir."""
    return cv2.merge([b, g, r])

def enhance_image(image, gamma=1.2, use_clahe=True):
    """
    Görüntüyü iyileştirir:
    - CLAHE ile adaptif kontrast
    - Gamma ile parlaklık ayarı
    """
    img = image.copy()

    # CLAHE (daha iyi kontrast)
    if use_clahe:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        lab = cv2.merge((cl, a, b))
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    else:
        # Normal histogram equalization
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Gamma düzeltme
    invGamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** invGamma * 255 for i in np.arange(0, 256)]).astype("uint8")
    img = cv2.LUT(img, table)

    return img

def auto_crop(image, margin=5):
    """
    Siyah kenarları kırpar ve hafif margin bırakır.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)
    coords = cv2.findNonZero(thresh)

    if coords is None:
        return image, (0, 0, image.shape[1], image.shape[0])

    x, y, w, h = cv2.boundingRect(coords)

    # Kenar kaybı yaşamamak için küçük boşluk bırak
    x = max(x - margin, 0)
    y = max(y - margin, 0)
    w = min(w + 2*margin, image.shape[1] - x)
    h = min(h + 2*margin, image.shape[0] - y)

    cropped = image[y:y+h, x:x+w]
    return cropped, (x, y, w, h)
