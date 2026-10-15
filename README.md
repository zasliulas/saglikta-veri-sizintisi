# Sağlık Veri Setlerinde Veri Sızıntısının Sınıflandırma Başarımına Etkisinin Deneysel Olarak İncelenmesi

Bu proje, makine öğrenmesi modellerinde sağlık ve biyomedikal veri setleri kullanılırken yapılan metodolojik veri sızıntısı (data leakage) hatalarının model performansı ve genellenebilirliği üzerindeki etkilerini deneysel olarak incelemeyi amaçlamaktadır.

## Veri Setleri
- **UCI Heart Disease (Özgün Sürüm):** 303 satır, 14 öznitelik (Referans/Sızıntısız zemin).
- **Kaggle Heart Disease:** 1025 satır, %70.54 tekrar eden satır (Data Duplication Leakage simülasyonu).
- **Pima Indians Diabetes:** 768 satır, 9 öznitelik, gizli biyolojik sıfır değerleri (Imputation Leakage simülasyonu).

## Kurulum ve Çalıştırma
Gerekli kütüphaneleri yüklemek için:
```bash
pip install -r requirements.txt