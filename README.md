<p align="center">
  <a href="https://www.uit.edu.vn/" title="Trường Đại học Công nghệ Thông tin" style="border: 5;">
    <img src="https://i.imgur.com/WmMnSRt.png" alt="Trường Đại học Công nghệ Thông tin | University of Information Technology">
  </a>
</p>

<!-- Title -->
<h1 align="center"><b>CS317.P21 - PHÁT TRIỂN VÀ VẬN HÀNH HỆ THỐNG MÁY HỌC</b></h1>

## COURSE INTRODUCTION
<a name="gioithieumonhoc"></a>
* *Course Title*: Phát triển và vận hành hệ thống máy học
* *Course Code*: CS317.P21
* *Year*: 2024-2025

## ACADEMIC ADVISOR
<a name="giangvien"></a>
* *Đỗ Văn Tiến* - tiendv@uit.edu.vn
* *Lê Trần Trọng Khiêm* - khiemltt@uit.edu.vn

## MEMBERS
<a name="thanhvien"></a>
* Từ Minh Phi - 22521080
* Lê Thành Tiến - 22521467
* Dương Thành Trí - 22521516
* Nguyễn Minh Thiện  - 22521391
* Nguyễn Quốc Vinh - 22521674

---
# CS317 Lab 1 – Titanic Survival Prediction Pipeline

## Overview
This repository contains the solution for **CS317 Lab 1**, where you build an end‑to‑end machine‑learning pipeline that predicts Titanic passenger survival.  The lab demonstrates data exploration, preprocessing, feature engineering, model training, evaluation, and inference — orchestrated with **Metaflow** and tracked with **MLflow**.

## Dataset
Download Kaggle’s *Titanic: Machine Learning from Disaster* dataset and place the CSVs in the `dataset/` directory:

| File | Description |
|------|-------------|
| `train.csv` | Training rows with features **and** the target `Survived`. |
| `test.csv` | Test rows with the same features but **without** `Survived`. |
| `gender_submission.csv` | (Optional) Sample submission that assumes every female passenger survived. |

## Project Structure
```text
CS317-Lab-1---Titanic-survival-prediction-pipeline/
│
├── dataset/                # Titanic CSV files (ignored by Git)
│   ├── train.csv
│   ├── test.csv
│   └── gender_submission.csv
├── source/
│   └── Task_1.py           # Metaflow FlowSpec (training + evaluation)
├── requirements.txt        # Python dependencies
├── LICENSE
└── README.md               # You are here
```

## Installation
```bash
# 1 · Clone the repository
$ git clone https://github.com/ThienNguyenMinh1311/CS317-Lab-1---Titanic-survival-prediction-pipeline.git
$ cd CS317-Lab-1---Titanic-survival-prediction-pipeline

# 2 · Create & activate a virtual environment (recommended)
$ python -m venv venv
$ source venv/bin/activate          # Windows: venv\Scripts\activate

# 3 · Install required packages
(venv)$ pip install -r requirements.txt
```

## Usage
All steps are encapsulated in **`source/Task_1.py`**, a Metaflow *FlowSpec* named `SklearnPipelineFlow`.

### 1 · Run the Full Pipeline
```bash
python source/Task_1.py run
```
This executes the flow in order: **start → check_versions → load_data → build_pipeline → train_model → evaluate → end**.  Model artifacts (`best_rf_model.pkl`, `best_knn_model.pkl`) are saved in the project root and logged to MLflow.

### 2 · Pass Custom Parameters (optional)
```bash
python source/Task_1.py run \
  --train_data_path dataset/train.csv \
  --test_data_path  dataset/test.csv \
  --gender_submission_path dataset/gender_submission.csv
```

### 3 · Inspect or Resume Runs
```bash
# List completed runs
python source/Task_1.py list

# Resume from a specific step (e.g., evaluate)
python source/Task_1.py resume <RUN_ID>/evaluate
```

### 4 · Open Metaflow & MLflow UIs
```bash
# Metaflow card/metadata UI (if installed)
python source/Task_1.py ui            # default http://localhost:8080

# MLflow experiment tracking UI
export MLFLOW_TRACKING_URI="file:./mlruns"
mlflow ui                             # http://localhost:5000
```

## Pipeline Description
1. **Data Loading** – read CSVs, drop unused columns, split features/target.
2. **Preprocessing**
   * Numerical: impute (mean/median) + standardise.
   * Categorical: impute most‑frequent + one‑hot encode.
3. **Feature Engineering** – `FamilySize` & title extraction (future work).
4. **Model Training** – grid‑search two classifiers:
   * **RandomForestClassifier**
   * **KNeighborsClassifier**
5. **Evaluation** – accuracy on the provided `gender_submission.csv` labels; metrics logged to MLflow.
6. **Artifacts** – best models persisted with `joblib` and logged to MLflow.

## Technologies
| Tool / Library | Purpose |
|----------------|---------|
| **Metaflow** | Orchestrates the ML pipeline with reproducible steps |
| **MLflow** | Tracks experiments, hyper‑parameters, metrics, artifacts |
| **scikit‑learn** | Core ML algorithms + `GridSearchCV` tuning |
| pandas / numpy | Data manipulation |
| matplotlib / seaborn | Visualisation of data & metrics |
| joblib | Serialises trained models |

## Dependencies
Key packages (see `requirements.txt` for exact versions):
- Python 3.8+
- pandas, numpy
- scikit‑learn
- matplotlib, seaborn
- metaflow
- mlflow
- joblib


## Contributing
This repository is part of a CS317 lab assignment; external contributions are not expected.  Feel free to fork and experiment.


## Acknowledgments
- Dataset courtesy of Kaggle.

## Contact
Open an issue or reach out to **[@ThienNguyenMinh1311](https://github.com/ThienNguyenMinh1311)** on GitHub.

