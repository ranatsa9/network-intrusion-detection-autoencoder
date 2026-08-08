# Network Intrusion Detection using a Deep Autoencoder

An unsupervised deep learning system that detects network attacks by learning
what *normal* traffic looks like — and flagging anything that deviates from it.

Built on the KDD Cup 1999 dataset as the final project for the Deep Learning
unit of the Tuwaiq Academy Data Science & AI Bootcamp.

---

## Problem

Traditional signature-based intrusion detection can only catch attacks it has
seen before. This project takes the opposite approach: an **autoencoder** is
trained exclusively on normal network connections. At inference time, any
connection the model cannot reconstruct accurately is flagged as an anomaly.

The practical advantage is that the detector can flag attack types that did not
exist when it was trained.

---

## Dataset

**KDD Cup 1999** (10% subset) — [Kaggle](https://www.kaggle.com/datasets/galaxyh/kdd-cup-1999-data)

| Property | Value |
|---|---|
| Raw records | 494,021 |
| Features | 41 (9 basic, 13 content, 9 time-window, 10 host-window) |
| Missing values | 0 |
| Duplicates | 348,435 (70.5%) |
| After deduplication | 145,586 |

### Class distribution

| Category | Count | Share |
|---|---|---|
| DoS | 391,458 | 79.2% |
| Normal | 97,278 | 19.7% |
| Probe | 4,107 | 0.83% |
| R2L | 1,126 | 0.23% |
| U2R | 52 | 0.01% |

> The dataset is dominated by two automated DoS attacks (`smurf`, `neptune`),
> which also explains the very high duplicate rate.

---

## Pipeline

```
Load → EDA → Clean → Split → Encode → Scale → Train → Threshold → Evaluate → Save → Infer
```

### Preprocessing decisions

| Decision | Rationale |
|---|---|
| Drop all duplicates | 70% of rows are byte-identical repeats from automated attacks |
| Drop `num_outbound_cmds`, `is_host_login` | Constant columns carry no information |
| Train on **normal traffic only** | Required by the autoencoder anomaly-detection approach |
| `OneHotEncoder(handle_unknown="ignore")` | Unseen service values map to all-zeros instead of raising an error |
| `MinMaxScaler` | Output layer uses sigmoid, so inputs must share the [0,1] range |
| Fit encoder and scaler on **train only** | Prevents data leakage |

### Data splits

| Split | Contents | Rows |
|---|---|---|
| Train | Normal only | 52,699 |
| Validation | Normal only | 17,566 |
| Test | Normal + all attacks | 75,321 |

---

## Model

```
Input (72) → 64 → 32 → [bottleneck] → 32 → 64 → Output (72)
```

| Component | Setting |
|---|---|
| Hidden activation | ReLU |
| Output activation | Sigmoid |
| Loss | MSE |
| Optimizer | Adam |
| Batch size | 256 |

### Regularization

| Technique | Configuration |
|---|---|
| L2 | 1e-5 on every Dense layer |
| Dropout | 0.1 on encoder layers |
| Early Stopping | patience 8, restores best weights |
| ReduceLROnPlateau | factor 0.5, patience 4 |

### Hyperparameter tuning

Three bottleneck sizes were compared under identical training settings, selected
on `val_loss` — reconstruction quality on unseen normal traffic.

| Bottleneck | val_loss | Precision | Recall | F1 |
|---|---|---|---|---|
| 8 | 0.001589 | 0.9969 | 0.9659 | 0.9812 |
| 16 | 0.001695 | 0.9966 | 0.9680 | 0.9821 |
| 32 | 0.001594 | 0.9960 | 0.9603 | 0.9778 |

Differences are marginal, indicating the architecture is not sensitive to
bottleneck size in this range.

> **Note on the selection criterion:** the model is chosen on `val_loss`, not
> F1. Selecting on F1 would require attack labels and would leak test
> information into model selection.

---

## Detection Method

For every connection the reconstruction error is computed as:

```
error = mean( (x - x̂)² )
```

The decision threshold is the **99th percentile of validation errors** —
derived from normal traffic only, never from test labels. It corresponds to an
accepted false-alarm rate of 1% on normal traffic, decided in advance.

```
error > threshold  →  ATTACK
error ≤ threshold  →  NORMAL
```

---

## Results

<!-- TODO: update with the final run's numbers -->

| Metric | Value |
|---|---|
| Precision | |
| Recall | |
| F1-Score | |
| ROC-AUC | |
| PR-AUC | |
| False Alarm Rate | |

### Recall by attack category

| Category | Samples | Recall |
|---|---|---|
| DoS | | |
| Probe | | |
| R2L | | |
| U2R | | |

> **The headline finding is not the aggregate F1.** DoS attacks dominate the
> data and inflate every summary metric. Broken down by family, R2L attacks —
> unauthorised remote access such as stolen credentials — are detected at a far
> lower rate, because they closely resemble legitimate traffic and the
> autoencoder reconstructs them successfully.

---

## Limitations

1. **Class balance does not reflect reality.** Attacks make up ~80% of this
   dataset; in a real network they are a tiny fraction of traffic.
2. **Deduplication changes what is measured.** Removing 70% duplicate rows means
   evaluation measures recognition of *distinct attack patterns*, not the share
   of packets caught in a live stream.
3. **The data is from 1998.** Network protocols, traffic profiles, and attack
   techniques have changed substantially since.
4. **The most dangerous categories are the least represented.** R2L and U2R —
   which correspond to actual system compromise — have the fewest samples and
   the weakest detection rates.
5. **Non-determinism.** GPU training is not fully deterministic; metrics vary by
   a few percent between runs.

---

## Repository Structure

```
├── notebooks/
│   └── DL_Project.ipynb        End-to-end pipeline
├── src/
│   └── inference.py            Deployment-facing prediction function
├── models/
│   ├── autoencoder.keras       Trained model
│   ├── encoder.pkl             Fitted OneHotEncoder
│   ├── scaler.pkl              Fitted MinMaxScaler
│   └── metadata.json           Threshold + column order
├── reports/
│   ├── presentation.pdf
│   └── figures/
├── requirements.txt
└── README.md
```

> All four files in `models/` are required. The model alone is not enough —
> without the encoder, scaler, and metadata the inference pipeline cannot
> reproduce the training-time representation.

---

## Usage

### Web app

Run the Streamlit deployment locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app supports a single-connection form and batch CSV inference. For free
hosting, connect this repository to Streamlit Community Cloud and set the main
file path to `app.py`.

### Install

```bash
git clone https://github.com/Maramjamaan/network-intrusion-detection-autoencoder.git
cd network-intrusion-detection-autoencoder
pip install -r requirements.txt
```

### Predict a single connection

```python
from src.inference import predict_connection

connection = {
    "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
    "src_bytes": 181, "dst_bytes": 5450,
    # ... remaining features
}

print(predict_connection(connection))
```

Output:

```json
{
  "prediction": "NORMAL",
  "anomaly_score": 0.000412,
  "threshold": 0.011763,
  "confidence": 0.04
}
```

`confidence` is how many times the error exceeds the threshold, capped at 10.

### Reproduce

Open `notebooks/DL_Project.ipynb` in Google Colab and run all cells. Download
the dataset from the Kaggle link above and upload it to the session.

---

## Requirements

```
tensorflow>=2.15
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
```

---

## Author

**Maram Alzahrani**
Data Science & AI Bootcamp — Tuwaiq Academy

[GitHub](https://github.com/Maramjamaan) · [LinkedIn](https://linkedin.com/in/maram-alzahrani314/)
