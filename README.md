# Complaint-Classification-Priority-Prediction
A multi-task deep learning model that classifies complaint type, predicts department, and estimates priority from text. Achieves strong class accuracy, with priority prediction limited by data imbalance.
# Complaint Classification & Priority Prediction

**Overview**
• **Purpose:** Multi-task system that classifies complaint type, predicts responsible department, and estimates priority from free-text complaints.
• **Implementation:** TF-IDF vectorization + TensorFlow/Keras multi-task neural network; training, evaluation, and visualization scripts included.

---

## Requirements

• **Python:** 3.8–3.11 recommended
• **Core libraries:** numpy, pandas, scikit-learn, tensorflow, keras, matplotlib, seaborn, pickle
• **Install:** Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate        # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

• **requirements.txt (suggested):**

```
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow
joblib
```

---

## Repository structure (recommended)

• **/data/**: `complaint_database.csv` (raw dataset)
• **/models/**: saved model and encoders (`complaint_multitask_model_rewrite.h5`, `tfidf_vectorizer_rewrite.pkl`, `label_encoder_rewrite.pkl`, `dept_encoder_rewrite.pkl`)
• **/notebooks/**: exploratory notebooks
• **/assets/images/**: confusion matrices, training plots
• **train_complaint_classifier.py**: main training script. See code for details. fileciteturn1file0
• **README.md**: this file

---

## How to run (training)

• **Step 1:** Place `complaint_database.csv` in the project root or set `DATA_PATH` in `train_complaint_classifier.py`.
• **Step 2:** Activate virtualenv and install packages.
• **Step 3:** Run training:

```bash
python train_complaint_classifier.py
```

• **What the script does:**
• Loads and validates CSV, drops missing rows.
• Encodes labels (complaint class, department, priority).
• Splits data (stratified by priority).
• Vectorizes text using TF-IDF (max_features=3000, ngram=(1,2)).
• Builds a multi-task Keras model and trains with callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint).
• Saves: model (`complaint_multitask_model_rewrite.h5`), TF-IDF vectorizer, encoders, and plot images (`priority_training_history_rewrite.png`, `priority_confusion_matrix_rewrite.png`).

---

## How to run (inference)

• **Load model + encoders:** load the saved model and TF-IDF vectorizer.
• **Preprocess:** apply the vectorizer to new complaint text.
• **Predict:** call `model.predict()` and apply `argmax` to get labels. Map back to original strings using saved label encoders.

Example snippet:

```python
import pickle
from tensorflow import keras

vectorizer = pickle.load(open('tfidf_vectorizer_rewrite.pkl','rb'))
class_le = pickle.load(open('label_encoder_rewrite.pkl','rb'))
dept_le = pickle.load(open('dept_encoder_rewrite.pkl','rb'))
model = keras.models.load_model('complaint_multitask_model_rewrite.h5')

text = ["Water leakage near main pipeline causing big flood"]
X = vectorizer.transform(text).toarray()
preds = model.predict(X)
complaint = class_le.inverse_transform([preds[0].argmax()])[0]
priority = preds[1].argmax() + 1
department = dept_le.inverse_transform([preds[2].argmax()])[0]
```

---

## Architecture (flow)

• **Input:** Raw complaint text CSV (`complaint_text`, `complaint_class`, `priority`, `department`)
→ **Preprocessing:** Drop NA, encode labels, stratified split by priority
→ **TF-IDF Vectorizer:** fit on training text (max_features=3000, ngram=(1,2))
→ **Neural Network Embedding:** Dense layers → shared trunk → three task-specific heads (complaint_class, priority, department)
→ **Training:** multi-task loss with custom loss_weights (priority emphasized), per-task sample weights via `compute_sample_weight`
→ **Outputs:** classification labels, saved model & encoders, evaluation plots & confusion matrices.

(ASCII diagram)

```
[complaint_database.csv]
        |
   Preprocess & Encode
        |
   TF-IDF Vectorizer
        |
   Shared Dense Layers (512→256→128)
      /    |      \
 Class  Priority  Dept
 Head    Head     Head
 outputs outputs  outputs
```

---

## Known limitations & improvements

• **Current limitation:** Priority prediction underperforms (overall model accuracy ~39.33%) due to class imbalance and weak severity cues in text.
• **Suggested improvements:** transformer-based embeddings (BERT), metadata features (location, time, urgency flag), data augmentation/synthetic oversampling for minority priorities, cost-sensitive learning, and model ensembling.

---

## Contact

• **Author:** Sushant Sunil Kamble
• **Email:** [sushantk5357@gmail.com](mailto:sushantk5357@gmail.com)

---

## License

• MIT License (suggested)
