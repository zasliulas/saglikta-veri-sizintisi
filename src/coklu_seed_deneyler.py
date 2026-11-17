import os
import pandas as pd
import numpy as np
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

SEEDS = [42, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def get_models(seed):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=seed)
    }


def metrik_hesapla(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "F1-Score": f1_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred)
    }


kayitlar = []

# Veri setlerini yükle
cols_heart = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]
df_uci = pd.read_csv(os.path.join(DATA_DIR, "processed.cleveland.data"), names=cols_heart, na_values="?")
X_uci = df_uci.drop("target", axis=1)
y_uci = (df_uci["target"] > 0).astype(int)

df_kaggle = pd.read_csv(os.path.join(DATA_DIR, "heart.csv"))
df_kaggle_clean = df_kaggle.drop_duplicates().reset_index(drop=True)

data_pima, _ = arff.loadarff(os.path.join(DATA_DIR, "dataset_37_diabetes.arff"))
df_pima = pd.DataFrame(data_pima)
df_pima["class"] = df_pima["class"].apply(lambda x: 1 if x == b"tested_positive" else 0)
sifir_sutunlar = ["plas", "pres", "skin", "insu", "mass"]
df_pima[sifir_sutunlar] = df_pima[sifir_sutunlar].replace(0, np.nan)
X_pima = df_pima.drop("class", axis=1)
y_pima = df_pima["class"]

print("10 farklı seed ile deneyler koşturuluyor, lütfen bekleyin...\n")

for seed in SEEDS:
    modeller = get_models(seed)

    # 1. UCI CLEVELAND (Referans Sızıntısız)
    X_tr, X_te, y_tr, y_te = train_test_split(X_uci, y_uci, test_size=0.2, stratify=y_uci, random_state=seed)
    for m_adi, clf in modeller.items():
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        m = metrik_hesapla(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])
        kayitlar.append({"Veri Seti": "UCI Cleveland", "Akış": "Sızıntısız (Referans)", "Model": m_adi, "Seed": seed, **m})

    # 2A. KAGGLE HEART - SIZINTILI
    X_kl, y_kl = df_kaggle.drop("target", axis=1), df_kaggle["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_kl, y_kl, test_size=0.2, stratify=y_kl, random_state=seed)
    for m_adi, clf in modeller.items():
        pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        m = metrik_hesapla(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])
        kayitlar.append({"Veri Seti": "Kaggle Heart", "Akış": "Sızıntılı (Tekrarlı)", "Model": m_adi, "Seed": seed, **m})

    # 2B. KAGGLE HEART - SIZINTISIZ
    X_kc, y_kc = df_kaggle_clean.drop("target", axis=1), df_kaggle_clean["target"]
    X_tr, X_te, y_tr, y_te = train_test_split(X_kc, y_kc, test_size=0.2, stratify=y_kc, random_state=seed)
    for m_adi, clf in modeller.items():
        pipe = Pipeline([("sc", StandardScaler()), ("clf", clf)])
        pipe.fit(X_tr, y_tr)
        m = metrik_hesapla(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])
        kayitlar.append({"Veri Seti": "Kaggle Heart", "Akış": "Sızıntısız (Temiz)", "Model": m_adi, "Seed": seed, **m})

    # 3A. PIMA DIABETES - SIZINTILI (Bölme Öncesi KNN Imputation + Scaling + Feature Selection)
    X_p_imp = KNNImputer(n_neighbors=5).fit_transform(X_pima)
    X_p_sc = StandardScaler().fit_transform(X_p_imp)
    X_p_sel = SelectKBest(f_classif, k=6).fit_transform(X_p_sc, y_pima)
    X_tr, X_te, y_tr, y_te = train_test_split(X_p_sel, y_pima, test_size=0.2, stratify=y_pima, random_state=seed)
    for m_adi, clf in modeller.items():
        clf.fit(X_tr, y_tr)
        m = metrik_hesapla(y_te, clf.predict(X_te), clf.predict_proba(X_te)[:, 1])
        kayitlar.append({"Veri Seti": "Pima Diabetes", "Akış": "Sızıntılı (Bölme Öncesi)", "Model": m_adi, "Seed": seed, **m})

    # 3B. PIMA DIABETES - SIZINTISIZ (Pipeline İçi KNN Imputation + Scaling + Feature Selection)
    X_tr, X_te, y_tr, y_te = train_test_split(X_pima, y_pima, test_size=0.2, stratify=y_pima, random_state=seed)
    for m_adi, clf in modeller.items():
        pipe = Pipeline([
            ("imp", KNNImputer(n_neighbors=5)),
            ("sc", StandardScaler()),
            ("sel", SelectKBest(f_classif, k=6)),
            ("clf", clf)
        ])
        pipe.fit(X_tr, y_tr)
        m = metrik_hesapla(y_te, pipe.predict(X_te), pipe.predict_proba(X_te)[:, 1])
        kayitlar.append({"Veri Seti": "Pima Diabetes", "Akış": "Sızıntısız (Pipeline)", "Model": m_adi, "Seed": seed, **m})

df_res = pd.DataFrame(kayitlar)
df_res.to_csv(os.path.join(RESULTS_DIR, "tum_seed_sonuclari.csv"), index=False)

# Özet Tablo (Ortalama ± Std)
ozet = df_res.groupby(["Veri Seti", "Akış", "Model"], sort=False)[["Accuracy", "ROC-AUC", "F1-Score", "Recall"]].agg(["mean", "std"])
ozet_formatli = pd.DataFrame(index=ozet.index)
for col in ["Accuracy", "ROC-AUC", "F1-Score", "Recall"]:
    ozet_formatli[col] = ozet[(col, "mean")].map("{:.4f}".format) + " ± " + ozet[(col, "std")].map("{:.4f}".format)

ozet_formatli.to_csv(os.path.join(RESULTS_DIR, "ozet_tablo.csv"))
print(ozet_formatli.to_string())
print(f"\nSonuçlar başarıyla kaydedildi: {RESULTS_DIR}")