import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "/workspace/data/fetal_health.csv"
OUTPUT_DIR = "/workspace/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1) Load
df = pd.read_csv(DATA_PATH)
original_shape = df.shape

# 2) Basic cleaning: normalize column names
normalized_columns = []
for c in df.columns:
    c_norm = c.strip().lower().replace(" ", "_").replace("-", "_")
    c_norm = c_norm.replace("__", "_")
    normalized_columns.append(c_norm)
df.columns = normalized_columns

# Ensure target column name
if "fetal_health" not in df.columns:
    raise ValueError("Target column 'fetal_health' not found after normalization.")

# 3) Basic diagnostics
num_missing = df.isna().sum().sum()
num_duplicates = df.duplicated().sum()

# Drop duplicates if any (non-destructive EDA)
if num_duplicates > 0:
    df = df.drop_duplicates().reset_index(drop=True)

# 4) Quick EDA
# Class distribution
class_counts = df["fetal_health"].value_counts().sort_index()
plt.figure(figsize=(5,4))
sns.barplot(x=class_counts.index.astype(int), y=class_counts.values, palette="viridis")
plt.title("Sınıf Dağılımı (fetal_health)")
plt.xlabel("Sınıf (1=Normal, 2=Suspect, 3=Pathological)")
plt.ylabel("Adet")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "class_balance.png"))
plt.close()

# Selected distributions
selected_features = [
    "baseline_value", "accelerations", "abnormal_short_term_variability",
    "histogram_width"
]
for feat in selected_features:
    if feat in df.columns:
        plt.figure(figsize=(5,4))
        sns.histplot(df[feat], kde=True, bins=30, color="#4C72B0")
        plt.title(f"Dağılım: {feat}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"dist_{feat}.png"))
        plt.close()

# Boxplot by class for baseline_value
if "baseline_value" in df.columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(data=df, x="fetal_health", y="baseline_value", palette="viridis")
    plt.title("Baseline Value - Sınıfa Göre Dağılım")
    plt.xlabel("fetal_health")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "box_baseline_value_by_class.png"))
    plt.close()

# Correlation heatmap
numeric_cols = df.drop(columns=["fetal_health"]).select_dtypes(include=[np.number]).columns
plt.figure(figsize=(10,8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0)
plt.title("Korelasyon Isı Haritası")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()

# 5) Split & preprocess
X = df.drop(columns=["fetal_health"])
y = df["fetal_health"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6) Models
# Logistic Regression with scaling
lr_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, multi_class="multinomial", n_jobs=None))
])

# Random Forest (no scaling needed)
rf_model = RandomForestClassifier(
    n_estimators=300, random_state=42, class_weight="balanced_subsample"
)

# Train
lr_pipeline.fit(X_train, y_train)
rf_model.fit(X_train, y_train)

# Predict
lr_pred = lr_pipeline.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Reports
lr_report = classification_report(y_test, lr_pred, digits=3)
rf_report = classification_report(y_test, rf_pred, digits=3)

# Confusion matrices
for name, pred in [("lr", lr_pred), ("rf", rf_pred)]:
    cm = confusion_matrix(y_test, pred, labels=[1,2,3])
    plt.figure(figsize=(4,3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=[1,2,3], yticklabels=[1,2,3])
    plt.xlabel("Tahmin")
    plt.ylabel("Gerçek")
    plt.title(f"Karmaşıklık Matrisi — {name.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"cm_{name}.png"))
    plt.close()

# 7) Console summary
print("=== Veri Özeti ===")
print(f"Orijinal şekil: {original_shape}, Duplicates: {num_duplicates}, Missing toplam: {num_missing}")
print("Sınıf dağılımı:\n", class_counts)

print("\n=== Logistic Regression Raporu ===")
print(lr_report)
print("\n=== Random Forest Raporu ===")
print(rf_report)

print(f"\nGörseller ve çıktılar: {OUTPUT_DIR}")