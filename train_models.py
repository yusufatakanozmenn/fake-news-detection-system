import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Klasörleri oluştur
os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Veri setlerini oku
fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

# Etiketler
fake["label"] = 0   # Sahte haber
true["label"] = 1   # Gerçek haber

# Veri setlerini birleştir
df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Başlık ve metni birleştir
df["content"] = df["title"].fillna("") + " " + df["text"].fillna("")

X = df["content"]
y = df["label"]

print("Toplam veri sayısı:", len(df))
print(df["label"].value_counts())

# Sınıf dağılım grafiği
label_counts = df["label"].value_counts()

plt.figure(figsize=(5, 4))
plt.bar(["Fake", "True"], [label_counts[0], label_counts[1]])
plt.title("Fake ve True Haber Dağılımı")
plt.xlabel("Sınıf")
plt.ylabel("Haber Sayısı")
plt.tight_layout()
plt.savefig("images/class_distribution.png")
plt.close()

# Train-test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_df=0.7,
    ngram_range=(1, 2),
    max_features=50000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Modeller
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = []

best_model = None
best_model_name = ""
best_f1 = 0

for name, model in models.items():
    print(f"\n{name} eğitiliyor...")

    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

    print(f"{name} Accuracy: {acc:.4f}")
    print(f"{name} F1 Score: {f1:.4f}")

    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        best_model_name = name
        best_preds = preds

# Sonuç tablosu
results_df = pd.DataFrame(results)
results_df.to_csv("models/model_results.csv", index=False)

print("\nModel Karşılaştırma Sonuçları:")
print(results_df)

print("\nEn iyi model:", best_model_name)

# Model karşılaştırma grafiği
plt.figure(figsize=(9, 5))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Model Accuracy Karşılaştırması")
plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=20)
plt.ylim(0.9, 1.0)
plt.tight_layout()
plt.savefig("images/model_comparison.png")
plt.close()

# En iyi model raporu
report = classification_report(
    y_test,
    best_preds,
    target_names=["Fake", "True"]
)

print("\nEn iyi model classification report:")
print(report)

with open("models/classification_report.txt", "w", encoding="utf-8") as f:
    f.write("Best Model: " + best_model_name + "\n\n")
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_test, best_preds)

plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Tahmin")
plt.ylabel("Gerçek")
plt.xticks([0, 1], ["Fake", "True"])
plt.yticks([0, 1], ["Fake", "True"])

for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig("images/confusion_matrix.png")
plt.close()

# En iyi modeli ve vectorizer'ı kaydet
with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("models/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("\nModel ve vectorizer kaydedildi.")
print("Proje çıktıları images/ ve models/ klasörlerine kaydedildi.")