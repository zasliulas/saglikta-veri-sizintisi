import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

df = pd.read_csv(os.path.join(RESULTS_DIR, "tum_seed_sonuclari.csv"))

plt.rcParams["font.sans-serif"] = "DejaVu Sans"
sns.set_style("whitegrid")

# Grafik 1: Kaggle Heart - Sızıntılı vs Sızıntısız (Duplication Leakage)
df_kaggle = df[df["Veri Seti"] == "Kaggle Heart"]

plt.figure(figsize=(9, 5), dpi=300)
ax = sns.barplot(
    data=df_kaggle,
    x="Model",
    y="Accuracy",
    hue="Akış",
    palette=["#C0392B", "#1B365D"],
    errorbar="sd",
    capsize=0.1
)
plt.title("Kaggle Kalp Hastalığı: Kayıt Tekrarı Sızıntısının (Duplication Leakage) Etkisi", fontsize=11.5, fontweight="bold", pad=12)
plt.ylabel("Doğruluk (Accuracy - 10 Seed Ortalaması)", fontsize=10)
plt.xlabel("Sınıflandırma Algoritması", fontsize=10)
plt.ylim(0.65, 1.08)
plt.legend(title="Deneysel Akış", frameon=True)

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=4, fontsize=8.5, fontweight="bold")

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "kaggle_duplication_leakage.png"))
plt.close()

# Grafik 2: Pima Diabetes - Sızıntılı vs Sızıntısız (Preprocessing Leakage)
df_pima = df[df["Veri Seti"] == "Pima Diabetes"]

plt.figure(figsize=(9, 5), dpi=300)
ax2 = sns.boxplot(
    data=df_pima,
    x="Model",
    y="ROC-AUC",
    hue="Akış",
    palette=["#D97706", "#0284C7"],
    width=0.5
)
plt.title("Pima Diyabet: Ön İşleme Sızıntısının (Imputation/Scaling) ROC-AUC Dağılımına Etkisi", fontsize=11.5, fontweight="bold", pad=12)
plt.ylabel("ROC-AUC Skoru (10 Farklı Seed)", fontsize=10)
plt.xlabel("Sınıflandırma Algoritması", fontsize=10)
plt.ylim(0.74, 0.90)
plt.legend(title="Deneysel Akış", frameon=True)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "pima_preprocessing_leakage.png"))
plt.close()

print(f"Grafikler başarıyla oluşturuldu: {FIGURES_DIR}")