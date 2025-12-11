# train_complaint_classifier.py (corrected sample_weight handling)
import os
import random
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from keras import layers
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

warnings.filterwarnings("ignore")
os.environ["PYTHONHASHSEED"] = "0"

# ---------------------------
# Reproducibility
# ---------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------------------------
# Config
# ---------------------------
DATA_PATH = "complaint_database.csv"
OUTPUT_DIR = Path(".")
OUTPUT_DIR.mkdir(exist_ok=True)
PRIORITY_LABELS = {1: "Critical", 2: "High", 3: "Medium", 4: "Low"}

# ---------------------------
# Utilities
# ---------------------------
def safe_load_csv(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found. Place your dataset in the script folder.")
    return pd.read_csv(path)

def prepare_tf_idf(X_train_texts, X_test_texts, max_features=3000, ngram_range=(1,2)):
    vect = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )
    X_train_vec = vect.fit_transform(X_train_texts).toarray()
    X_test_vec = vect.transform(X_test_texts).toarray()
    return vect, X_train_vec, X_test_vec

def compute_task_sample_weights(y_train_class, y_train_priority, y_train_dept):
    # sample weights for each task separately (balanced)
    sw_complaint = compute_sample_weight("balanced", y_train_class)
    sw_priority = compute_sample_weight("balanced", y_train_priority)
    sw_dept = compute_sample_weight("balanced", y_train_dept)
    return sw_complaint, sw_priority, sw_dept

def plot_and_save_history(history, out_dir):
    # only plot priority accuracy & loss (most important)
    fig, ax = plt.subplots(1,2, figsize=(14,5))
    # ensure keys exist
    if "priority_accuracy" in history.history:
        ax[0].plot(history.history["priority_accuracy"], label="train")
    if "val_priority_accuracy" in history.history:
        ax[0].plot(history.history["val_priority_accuracy"], label="val")
    ax[0].axhline(0.80, color="red", linestyle="--", label="80% target")
    ax[0].set_title("Priority Accuracy")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("accuracy"); ax[0].legend()

    if "priority_loss" in history.history:
        ax[1].plot(history.history["priority_loss"], label="train")
    if "val_priority_loss" in history.history:
        ax[1].plot(history.history["val_priority_loss"], label="val")
    ax[1].set_title("Priority Loss")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("loss"); ax[1].legend()

    plt.tight_layout()
    p = out_dir/"priority_training_history_rewrite.png"
    plt.savefig(p, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved training history -> {p}")

# ---------------------------
# Load dataset
# ---------------------------
print("\nLoading dataset...")
df = safe_load_csv(DATA_PATH)
print(f"Loaded {len(df)} rows.")

# Minimal expected columns
required_columns = {"complaint_text", "complaint_class", "priority", "department"}
if not required_columns.issubset(set(df.columns)):
    raise ValueError(f"Dataset must contain columns: {required_columns}")

# Drop rows with missing text or priority
df = df.dropna(subset=["complaint_text", "priority"])
df["priority"] = df["priority"].astype(int)  # assume 1..4
print("Preprocessing complete.")

# ---------------------------
# Prepare labels and splits
# ---------------------------
X = df["complaint_text"].astype(str).values
y_class_raw = df["complaint_class"].astype(str).values
y_priority_raw = df["priority"].astype(int).values
y_dept_raw = df["department"].astype(str).values

# Encode categorical outputs
class_le = LabelEncoder()
y_class = class_le.fit_transform(y_class_raw)

dept_le = LabelEncoder()
y_dept = dept_le.fit_transform(y_dept_raw)

# priority: convert to 0..3 for training convenience
y_priority = (y_priority_raw - 1).astype(int)

# Stratified split on priority to keep class distribution
print("\nSplitting dataset (stratified by priority)...")
X_train, X_test, y_train_class, y_test_class, y_train_priority, y_test_priority, y_train_dept, y_test_dept = train_test_split(
    X, y_class, y_priority, y_dept,
    test_size=0.2,
    random_state=SEED,
    stratify=y_priority
)

print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

# ---------------------------
# Vectorize text
# ---------------------------
print("\nVectorizing text with TF-IDF...")
vectorizer, X_train_vec, X_test_vec = prepare_tf_idf(X_train, X_test, max_features=3000, ngram_range=(1,2))
print(f"Feature dimension: {X_train_vec.shape[1]}")

# ---------------------------
# Compute sample weights (per-task)
# ---------------------------
print("\nComputing per-task sample weights...")
sw_complaint, sw_priority, sw_dept = compute_task_sample_weights(y_train_class, y_train_priority, y_train_dept)

# Slightly emphasize priority (multiply by 1.5) but keep balanced shape
sw_priority_boosted = sw_priority * 1.5

# ---- CORRECT: use a DICT of numpy arrays matching the 'y' dict structure ----
sample_weight_dict = {
    "complaint_class": np.asarray(sw_complaint, dtype=np.float32),
    "priority": np.asarray(sw_priority_boosted, dtype=np.float32),
    "department": np.asarray(sw_dept, dtype=np.float32)
}
# --------------------------------------------------------------------------

print("Sample weights computed.")

# ---------------------------
# Build model (clean multi-task)
# ---------------------------
print("\nBuilding multi-task model...")
input_dim = X_train_vec.shape[1]
num_complaint = len(np.unique(y_class))
num_priority = 4
num_dept = len(np.unique(y_dept))

from tensorflow import keras

inp = layers.Input(shape=(input_dim,), name="input")
x = layers.Dense(512, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(inp)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.4)(x)

x = layers.Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)

x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.2)(x)

# Priority branch (lighter dropout to allow learning)
p = layers.Dense(128, activation="relu")(x)
p = layers.BatchNormalization()(p)
p = layers.Dropout(0.15)(p)
p = layers.Dense(64, activation="relu")(p)
p = layers.BatchNormalization()(p)
p = layers.Dropout(0.1)(p)
priority_output = layers.Dense(num_priority, activation="softmax", name="priority")(p)

# Complaint class branch
c = layers.Dense(64, activation="relu")(x)
c = layers.Dropout(0.15)(c)
complaint_output = layers.Dense(num_complaint, activation="softmax", name="complaint_class")(c)

# Department branch
d = layers.Dense(64, activation="relu")(x)
d = layers.Dropout(0.15)(d)
department_output = layers.Dense(num_dept, activation="softmax", name="department")(d)

model = keras.Model(inputs=inp, outputs=[complaint_output, priority_output, department_output])

# Compile: heavier weight for priority
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss={
        "complaint_class": "sparse_categorical_crossentropy",
        "priority": "sparse_categorical_crossentropy",
        "department": "sparse_categorical_crossentropy"
    },
    loss_weights={
        "complaint_class": 1.0,
        "priority": 8.0,   # strong emphasis on priority accuracy
        "department": 0.5
    },
    metrics={
        "complaint_class": ["accuracy"],
        "priority": ["accuracy"],
        "department": ["accuracy"]
    }
)

model.summary()

# ---------------------------
# Callbacks
# ---------------------------
checkpoint_path = str(OUTPUT_DIR/"best_priority_model_rewrite.h5")
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_priority_accuracy",
    patience=20,
    restore_best_weights=True,
    mode="max",
    verbose=1
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor="val_priority_accuracy",
    factor=0.3,
    patience=8,
    min_lr=1e-6,
    mode="max",
    verbose=1
)
checkpoint = keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    monitor="val_priority_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1
)

# ---------------------------
# Train
# ---------------------------
print("\nTraining... (this can take some minutes depending on dataset size)")
history = model.fit(
    X_train_vec,
    {
        "complaint_class": y_train_class,
        "priority": y_train_priority,
        "department": y_train_dept
    },
    sample_weight=sample_weight_dict,   # <-- pass dict (same structure as y)
    validation_split=0.25,
    epochs=64,
    batch_size=32,
    callbacks=[early_stopping, reduce_lr, checkpoint],
    verbose=1
)

# Save history plots
plot_and_save_history(history, OUTPUT_DIR)

# ---------------------------
# Load best and evaluate
# ---------------------------
if Path(checkpoint_path).exists():
    print(f"\nLoading best weights from {checkpoint_path}")
    model.load_weights(checkpoint_path)

print("\nRunning predictions on test set...")
preds = model.predict(X_test_vec, verbose=0)
y_pred_complaint = np.argmax(preds[0], axis=1)
y_pred_priority = np.argmax(preds[1], axis=1)
y_pred_dept = np.argmax(preds[2], axis=1)

complaint_acc = accuracy_score(y_test_class, y_pred_complaint)
priority_acc = accuracy_score(y_test_priority, y_pred_priority)
dept_acc = accuracy_score(y_test_dept, y_pred_dept)

print("\n=== Final Metrics ===")
print(f"Complaint classification accuracy: {complaint_acc*100:.2f}%")
print(f"Priority prediction accuracy:      {priority_acc*100:.2f}%")
print(f"Department allocation accuracy:    {dept_acc*100:.2f}%")
print("=======================")

# Per-priority breakdown (display in 1..4 format)
print("\nPriority breakdown (per-level accuracy):")
y_test_priority_display = y_test_priority + 1
y_pred_priority_display = y_pred_priority + 1
for p in [1,2,3,4]:
    mask = y_test_priority_display == p
    if mask.sum() == 0:
        print(f"  {PRIORITY_LABELS[p]:8s}:  No samples in test set")
    else:
        acc = accuracy_score(y_test_priority_display[mask], y_pred_priority_display[mask])
        print(f"  {PRIORITY_LABELS[p]:8s}:  {acc*100:.2f}% ({mask.sum()} samples)")

# Confusion matrix (counts)
cm = confusion_matrix(y_test_priority_display, y_pred_priority_display, labels=[1,2,3,4])
cm_df = pd.DataFrame(cm, index=[f"Actual {PRIORITY_LABELS[p]}" for p in [1,2,3,4]],
                     columns=[f"Pred {PRIORITY_LABELS[p]}" for p in [1,2,3,4]])
print("\nPriority confusion matrix (counts):")
print(cm_df)

# Save confusion matrix plot similar to original style
plt.figure(figsize=(9,8))
annot = np.empty_like(cm).astype(str)
cm_pct = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        annot[i,j] = f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)"

sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", xticklabels=[PRIORITY_LABELS[p] for p in [1,2,3,4]],
            yticklabels=[PRIORITY_LABELS[p] for p in [1,2,3,4]], cbar_kws={"label": "Count"}, square=True)
plt.title(f"Priority Confusion Matrix  |  Accuracy: {priority_acc*100:.2f}%")
plt.xlabel("Predicted Priority")
plt.ylabel("Actual Priority")
plt.tight_layout()
cm_path = OUTPUT_DIR/"priority_confusion_matrix_rewrite.png"
plt.savefig(cm_path, dpi=200, bbox_inches="tight")
plt.close()
print(f"Saved confusion matrix -> {cm_path}")

# Save model and encoders
model_save_path = OUTPUT_DIR/"complaint_multitask_model_rewrite.h5"
model.save(model_save_path)
print(f"Saved model -> {model_save_path}")

import pickle
with open(OUTPUT_DIR/"tfidf_vectorizer_rewrite.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(OUTPUT_DIR/"label_encoder_rewrite.pkl", "wb") as f:
    pickle.dump(class_le, f)
with open(OUTPUT_DIR/"dept_encoder_rewrite.pkl", "wb") as f:
    pickle.dump(dept_le, f)
print("Saved vectorizer and encoders.")

# Classification report for priority
print("\nPriority classification report:")
print(classification_report(y_test_priority_display, y_pred_priority_display,
                            target_names=[PRIORITY_LABELS[p] for p in [1,2,3,4]], zero_division=0))

# Final summary
print("\nTraining complete.")
if priority_acc >= 0.80:
    print(f"🎉 Priority accuracy target met: {priority_acc*100:.2f}% 🎉")
else:
    print(f"⚠️ Priority accuracy below 80%: {priority_acc*100:.2f}%")
    print("Consider: more data, metadata features, or transfer learning (transformer-based models).")
