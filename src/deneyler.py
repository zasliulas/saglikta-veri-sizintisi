import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

# Veri yolu ayarı
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")


def degerlendir(y_test, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
        "F1-Score": f1_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred)
    }


def yazdir_sonuc(baslik, sonuclar):
    print(f"\n--- {baslik} ---")
    for model_adi, metrikler in sonuclar.items():
        print(f"{model_adi:20s} | Acc: {metrikler['Accuracy']:.4f} | "
              f"AUC: {metrikler['ROC-AUC']:.4f} | F1: {metrikler['F1-Score']:.4f} | "
              f"Recall: {metrikler['Recall']:.4f}")


modeller = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

# ==============================================================================
# 1. UCI CLEVELAND (REFERANS - SIZINTISIZ BAZ MODEL)
# ==============================================================================
cols_heart = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df_uci = pd.read_csv(os.path.join(DATA_DIR, "processed.cleveland.data"), names=cols_heart, na_values="?")
X_uci = df_uci.drop("target", axis=1)
y_uci = (df_uci["target"] > 0).astype(int)  # İkili sınıflandırma (0: Sağlıklı, 1: Hasta)

X_tr, X_te, y_tr, y_te = train_test_split(X_uci, y_uci, test_size=0.2, stratify=y_uci, random_state=42)

sonuc_uci = {}
for isim, clf in modeller.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clf)
    ])
    pipe.fit(X_tr, y_tr)
    sonuc_uci[isim] = degerlendir(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])

yazdir_sonuc("1. UCI CLEVELAND (Sızıntısız Referans Baz Model)", sonuc_uci)

# ==============================================================================
# 2. KAGGLE HEART DISEASE (DUPLICATION LEAKAGE KARŞILAŞTIRMASI)
# ==============================================================================
df_kaggle = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))

# A) Sızıntılı Akış (723 mükerrer satır temizlenmeden bölünüyor)
X_k_leaky = df_kaggle.drop("target", axis=1)
y_k_leaky = df_kaggle["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X_k_leaky, y_k_leaky, test_size=0.2, stratify=y_k_leaky, random_state=42)

sonuc_k_leaky = {}
for isim, clf in modeller.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    pipe.fit(X_tr, y_tr)
    sonuc_k_leaky[isim] = degerlendir(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])

# B) Sızıntısız Akış (Önce mükerrer kayıtlar eleniyor -> 302 benzersiz satır)
df_kaggle_clean = df_kaggle.drop_duplicates().reset_index(drop=True)
X_k_clean = df_kaggle_clean.drop("target", axis=1)
y_k_clean = df_kaggle_clean["target"]
X_tr, X_te, y_tr, y_te = train_test_split(X_k_clean, y_k_clean, test_size=0.2, stratify=y_k_clean, random_state=42)

sonuc_k_clean = {}
for isim, clf in modeller.items():
    pipe = Pipeline([("scaler", StandardScaler()), ("model", clf)])
    pipe.fit(X_tr, y_tr)
    sonuc_k_clean[isim] = degerlendir(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])

yazdir_sonuc("2A. KAGGLE HEART - SIZINTILI (Tekrarlı Kayıtlar Var)", sonuc_k_leaky)
yazdir_sonuc("2B. KAGGLE HEART - SIZINTISIZ (Tekrarlar Temizlendi)", sonuc_k_clean)

# ==============================================================================
# 3. PIMA INDIANS DIABETES (IMPUTATION & SCALING LEAKAGE KARŞILAŞTIRMASI)
# ==============================================================================
data, _ = arff.loadarff(os.path.join(DATA_DIR, "dataset_37_diabetes.arff"))
df_pima = pd.DataFrame(data)
df_pima["class"] = df_pima["class"].apply(lambda x: 1 if x == b"tested_positive" else 0)

sifir_sutunlar = ["plas", "pres", "skin", "insu", "mass"]
df_pima[sifir_sutunlar] = df_pima[sifir_sutunlar].replace(0, np.nan)

X_p = df_pima.drop("class", axis=1)
y_p = df_pima["class"]

# A) Sızıntılı Akış (Bölme ÖNCESİ tüm veri seti üzerinden doldurma ve ölçekleme)
X_p_imputed = SimpleImputer(strategy="median").fit_transform(X_p)
X_p_scaled = StandardScaler().fit_transform(X_p_imputed)
X_tr, X_te, y_tr, y_te = train_test_split(X_p_scaled, y_p, test_size=0.2, stratify=y_p, random_state=42)

sonuc_p_leaky = {}
for isim, clf in modeller.items():
    clf.fit(X_tr, y_tr)
    sonuc_p_leaky[isim] = degerlendir(y_te, clf.predict(X_te), clf.predict_proba(X_te)[:, 1])

# B) Sızıntısız Akış (Bölme SONRASI Pipeline içinde yalnızca train'den öğrenme)
X_tr, X_te, y_tr, y_te = train_test_split(X_p, y_p, test_size=0.2, stratify=y_p, random_state=42)

sonuc_p_clean = {}
for isim, clf in modeller.items():
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", clf)
    ])
    pipe.fit(X_tr, y_tr)
    sonuc_p_clean[isim] = degerlendir(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])

yazdir_sonuc("3A. PIMA DIABETES - SIZINTILI (Bölme Öncesi Imputation/Scaling)", sonuc_p_leaky)
yazdir_sonuc("3B. PIMA DIABETES - SIZINTISIZ (Pipeline İçi İzole Akış)", sonuc_p_clean)