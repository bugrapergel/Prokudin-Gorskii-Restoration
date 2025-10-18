import argparse
import time
import os
from pathlib import Path
import cv2
from anafonksiyonlar import load_image, split_image, create_color_image, enhance_image, auto_crop
from alignchannels import align_channels, apply_alignment, align_channels_pyramid


def process_image(input_path, output_dir, metric='ncc', use_pyramid=False):
    print(f"\n=== İşlem Başladı: {input_path} ===\n")
    start_time = time.time()

    # 1. Görüntüyü yükle
    img = load_image(input_path)
    print(f"[OK] Gri görüntü yüklendi: {img.shape}")

    # 2. Kanallara böl
    b, g, r = split_image(img)
    print(f"[OK] Kanallar ayrıldı: B={b.shape}, G={g.shape}, R={r.shape}")

    # 3. Hizalama
    print(f"[INFO] {metric.upper()} metrik ile hizalama başlatıldı {'(Pyramid)' if use_pyramid else ''}")
    if use_pyramid:
        dx_g, dy_g, _ = align_channels_pyramid(b, g, metric=metric)
        dx_r, dy_r, _ = align_channels_pyramid(b, r, metric=metric)
    else:
        dx_g, dy_g, _ = align_channels(b, g, metric=metric)
        dx_r, dy_r, _ = align_channels(b, r, metric=metric)

    print(f"[SHIFT] Green kaydırma: dx={dx_g}, dy={dy_g}")
    print(f"[SHIFT] Red kaydırma: dx={dx_r}, dy={dy_r}")

    g_aligned = apply_alignment(g, dx_g, dy_g)
    r_aligned = apply_alignment(r, dx_r, dy_r)

    # 4. Renkli görüntüler oluştur
    img_unaligned = create_color_image(b, g, r)
    img_aligned = create_color_image(b, g_aligned, r_aligned)

    # 5. Görüntü iyileştirme
    img_enhanced = enhance_image(img_aligned)

    # 6. Otomatik kırpma
    img_final, crop_info = auto_crop(img_enhanced)
    print(f"[OK] Otomatik kırpma uygulandı: {crop_info}")

    # 7. Kaydet
    base_name = Path(input_path).stem
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(f"{output_dir}/{base_name}_1_unaligned.jpg", img_unaligned)
    cv2.imwrite(f"{output_dir}/{base_name}_2_aligned.jpg", img_aligned)
    cv2.imwrite(f"{output_dir}/{base_name}_3_enhanced.jpg", img_enhanced)
    cv2.imwrite(f"{output_dir}/{base_name}_4_final.jpg", img_final)
    print(f"[DONE] Çıktılar kaydedildi → {output_dir}")

    elapsed = time.time() - start_time
    return {'image': base_name, 'g_shift': (dx_g, dy_g), 'r_shift': (dx_r, dy_r), 'time': elapsed}


def main():
    parser = argparse.ArgumentParser(description='Prokudin-Gorskii Görüntü Restorasyon Sistemi')
    parser.add_argument('--input', required=True, help='Girdi görüntüsü veya klasör yolu')
    parser.add_argument('--output', default='results', help='Çıktı klasörü adı')
    parser.add_argument('--metric', default='ncc', choices=['ssd', 'ncc'], help='Hizalama metriği seç')
    parser.add_argument('--pyramid', action='store_true', help='Piramit tabanlı hizalama kullan')
    args = parser.parse_args()

    input_path = Path(args.input)
    files = [input_path] if input_path.is_file() else sum([list(input_path.glob(ext)) for ext in ['*.jpg', '*.png', '*.tif']], [])

    if not files:
        print("[HATA] Girdi bulunamadı!")
        return

    results = []
    for f in files:
        results.append(process_image(str(f), args.output, args.metric, args.pyramid))

    print("\n=== ÖZET ===")
    print(f"{'Görüntü':<15} {'G Shift':<12} {'R Shift':<12} {'Süre (sn)':<10}")
    print("-" * 50)
    for r in results:
        print(f"{r['image']:<15} {str(r['g_shift']):<12} {str(r['r_shift']):<12} {r['time']:<10.2f}")


if __name__ == '__main__':
    main()
