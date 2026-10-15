import pandas as pd
import numpy as np
from scipy.io import arff
from pathlib import Path

# Proje ana dizinini ve data klasörünün tam yolunu otomatik bulur
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

print(f"Veri klasörü aranıyor: {DATA_DIR}")

print("=" * 60)
print("1. ÖZGÜN UCI CLEVELAND HEART DISEASE İNCELEMESİ")
print("=" * 60)

uci_kolonlar = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "target"
]

uci_yolu = DATA_DIR / "processed.cleveland.data"
df_uci = pd.read_csv(uci_yolu, names=uci_kolonlar, na_values="?")

print(f"Satır Sayısı: {df_uci.shape[0]}, Sütun Sayısı: {df_uci.shape[1]}")
print(f"Tekrar Eden Satır Sayısı: {df_uci.duplicated().sum()}")
print("\nEksik Değerler (NaN):")
eksikler = df_uci.isnull().sum()
print(eksikler[eksikler > 0] if eksikler.sum() > 0 else "Eksik değer yok.")
print("\nHedef Sütun (target) Dağılımı:")
print(df_uci['target'].value_counts().sort_index())

print("\n" + "=" * 60)
print("2. KAGGLE HEART DISEASE İNCELEMESİ")
print("=" * 60)

kaggle_yolu = DATA_DIR / "heart.csv"
df_kaggle = pd.read_csv(kaggle_yolu)

print(f"Satır Sayısı: {df_kaggle.shape[0]}, Sütun Sayısı: {df_kaggle.shape[1]}")
tekrar_kaggle = df_kaggle.duplicated().sum()
print(f"Tekrar Eden Satır Sayısı: {tekrar_kaggle} (%{tekrar_kaggle / len(df_kaggle) * 100:.2f})")
print(f"Eksik Değer Sayısı (NaN): {df_kaggle.isnull().sum().sum()}")
print("\nHedef Sütun (target) Dağılımı:")
print(df_kaggle['target'].value_counts())

print("\n" + "=" * 60)
print("3. PIMA INDIANS DIABETES (.arff) İNCELEMESİ")
print("=" * 60)

pima_yolu = DATA_DIR / "dataset_37_diabetes.arff"
veri_arff, meta = arff.loadarff(pima_yolu)
df_pima = pd.DataFrame(veri_arff)

# Byte string sütunları düz metne çevir
for col in df_pima.select_dtypes(object).columns:
    df_pima[col] = df_pima[col].str.decode('utf-8')

print(f"Satır Sayısı: {df_pima.shape[0]}, Sütun Sayısı: {df_pima.shape[1]}")
print(f"Tekrar Eden Satır Sayısı: {df_pima.duplicated().sum()}")

# Pima için biyolojik sıfır değerleri (Gizli Eksik Veri Kontrolü)
biyolojik_kolonlar = ['plas', 'pres', 'skin', 'insu', 'mass']
print("\nSıfır Girilmiş Biyolojik Ölçümler (Gizli Eksik Değerler):")
for col in biyolojik_kolonlar:
    if col in df_pima.columns:
        sifir = (df_pima[col] == 0).sum()
        print(f"  {col:<6}: {sifir:>3} adet (%{sifir / len(df_pima) * 100:.1f})")

print("\nHedef Sütun (class) Dağılımı:")
print(df_pima.iloc[:, -1].value_counts())
print("=" * 60)