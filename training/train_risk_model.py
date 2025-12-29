"""
Script di training supervisionato per il classificatore di rischio clinico.

- Estrae dati storici dal database
- Allena un modello supervisionato interpretabile
- Salva il modello per l'uso da parte dell'IntelligentAgent
"""

import os
import sqlite3
import numpy as np
import joblib

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# CONFIGURAZIONE
# ============================================================

DB_PATH = "database/telemedicina.db"
MODEL_OUTPUT_PATH = "models/risk_classifier.pkl"
RANDOM_STATE = 42


# ============================================================
# ESTRAZIONE DATI
# ============================================================

def load_data_from_db(db_path):
    """
    Estrae feature e label dal database.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = """
        SELECT
            pressione_sistolica,
            pressione_diastolica,
            frequenza_cardiaca,
            temperatura,
            saturazione_ossigeno,
            glicemia,
            livello_rischio
        FROM interazioni
        WHERE livello_rischio IN ('basso', 'medio', 'alto')
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        raise RuntimeError("❌ Nessun dato disponibile per il training.")

    X = []
    y = []

    for row in rows:
        X.append(row[:-1])  # feature
        y.append(row[-1])   # label

    return np.array(X, dtype=float), np.array(y)


# ============================================================
# TRAINING
# ============================================================

def train_model(X, y):
    """
    Allena un Decision Tree supervisionato.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=10,
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\n📊 VALUTAZIONE MODELLO")
    print("=" * 50)
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return model


# ============================================================
# MAIN
# ============================================================

def main():
    print("🏥 Training modello di rischio clinico...\n")

    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database non trovato: {DB_PATH}")

    os.makedirs("models", exist_ok=True)

    print("📥 Caricamento dati dal database...")
    X, y = load_data_from_db(DB_PATH)
    print(f"✅ Campioni caricati: {len(X)}")

    print("\n🧠 Addestramento modello supervisionato...")
    model = train_model(X, y)

    print("\n💾 Salvataggio modello...")
    joblib.dump(model, MODEL_OUTPUT_PATH)

    print(f"✅ Modello salvato in: {MODEL_OUTPUT_PATH}")
    print("\n🎉 Training completato con successo!")


if __name__ == "__main__":
    main()
