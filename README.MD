# Görüntü İşleme Ödevi 1: Prokudin-Gorskii Renkli Fotoğraf Restorasyonu

Bu proje, 20. yüzyılın başlarında kullanılan Prokudin-Gorskii'nin renkli fotoğrafçılık tekniğini dijital ortamda taklit ederek, gri tonlamalı cam plakaları modern görüntü işleme teknikleriyle renkli fotoğraflara dönüştürmeyi amaçlamaktadır. Proje, temel görüntü işleme, metrik tabanlı hizalama ve görüntü iyileştirme becerilerini geliştirmek için tasarlanmıştır.

## Ödev Hedefleri

Proje, aşağıdaki temel hedefleri gerçekleştirmektedir:

* **Görüntü Bölme ve Hizalama**: Tek bir gri tonlamalı görüntüyü BGR kanallarına ayırma ve bu kanalları hizalama.
* **Metrik Kullanımı**: Hizalama için SSD (Sum of Squared Differences) ve NCC (Normalized Cross-Correlation) metriklerini uygulama.
* **Görüntü İyileştirme**: Hizalanmış görüntünün kalitesini artırmak için Histogram Eşitleme ve Gama Düzeltme tekniklerini kullanma.
* **Otomatik Kenar Kırpma (Bonus)**: Görüntü kenarlarındaki artifaktları otomatik olarak kırpma.
* **Piramit Tabanlı Hizalama (Bonus)**: Daha büyük görüntülerde hizalama süresini azaltmak için piramit tabanlı algoritma uygulama.

## Kurulum

Projeyi çalıştırmak için gerekli kütüphaneleri kurmanız gerekmektedir. Python 3.7 ve üzeri bir sürüm önerilir.

1.  **Sanal ortam oluşturun ve etkinleştirin:**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # macOS/Linux
    source venv/bin/activate
    ```

2.  **Gerekli kütüphaneleri yükleyin:**
    ```bash
    pip install numpy opencv-python matplotlib scikit-image pillow
    ```

