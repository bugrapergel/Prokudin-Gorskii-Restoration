import numpy as np
import cv2

def apply_alignment(channel, dx, dy):
    """np.roll yerine doğru kaydırma: OpenCV warpAffine kullanıldı."""
    transform_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(
        channel,
        transform_matrix,
        (channel.shape[1], channel.shape[0]),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return shifted

def ssd_metric(img1, img2):
    """Sum of Squared Differences (SSD) metriğini hesaplar. Düşük değer daha iyidir."""
    return np.sum((img1 - img2) ** 2)

def ncc_metric(img1, img2):
    """Normalized Cross-Correlation (NCC) metriğini hesaplar. Yüksek değer daha iyidir."""
    img1_mean = img1 - np.mean(img1)
    img2_mean = img2 - np.mean(img2)
    numerator = np.sum(img1_mean * img2_mean)
    denominator = np.sqrt(np.sum(img1_mean ** 2)) * np.sqrt(np.sum(img2_mean ** 2))
    return 0 if denominator == 0 else numerator / denominator

def align_channels(reference, target, search_range=15, metric='ncc', edge_crop=0.1):
    """Brute-force hizalama fonksiyonu."""
    h, w = reference.shape
    crop_h = int(h * edge_crop)
    crop_w = int(w * edge_crop)
    ref_crop = reference[crop_h:-crop_h, crop_w:-crop_w]

    best_score = float('-inf') if metric == 'ncc' else float('inf')
    best_dx, best_dy = 0, 0

    for dy in range(-search_range, search_range + 1):
        for dx in range(-search_range, search_range + 1):
            shifted = apply_alignment(target, dx, dy)
            shifted_crop = shifted[crop_h:-crop_h, crop_w:-crop_w]

            score = ncc_metric(ref_crop, shifted_crop) if metric == 'ncc' else ssd_metric(ref_crop, shifted_crop)

            if (metric == 'ncc' and score > best_score) or (metric == 'ssd' and score < best_score):
                best_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_score

def align_channels_pyramid(reference, target, search_range=15, metric='ncc', level=3):
    """Piramit tabanlı hizalama algoritması."""
    if level == 0:
        return align_channels(reference, target, search_range, metric)

    ref_small = cv2.resize(reference, (0, 0), fx=0.5, fy=0.5)
    target_small = cv2.resize(target, (0, 0), fx=0.5, fy=0.5)

    dx_small, dy_small, _ = align_channels_pyramid(ref_small, target_small, search_range, metric, level - 1)

    dx_coarse = dx_small * 2
    dy_coarse = dy_small * 2
    best_score = float('-inf') if metric == 'ncc' else float('inf')
    best_dx, best_dy = dx_coarse, dy_coarse

    for dy in range(dy_coarse - 2, dy_coarse + 3):
        for dx in range(dx_coarse - 2, dx_coarse + 3):
            shifted = apply_alignment(target, dx, dy)
            score = ncc_metric(reference, shifted) if metric == 'ncc' else ssd_metric(reference, shifted)

            if (metric == 'ncc' and score > best_score) or (metric == 'ssd' and score < best_score):
                best_score = score
                best_dx, best_dy = dx, dy

    return best_dx, best_dy, best_score
