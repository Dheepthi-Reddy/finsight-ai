# FinSight AI

## What is FinSight AI?

FinSight AI is a production-grade machine learning platform for financial industry. This project is aimed at doing "Fraud Detection" and "Compliance Assistance" simultaneously.

- Watches live stream of financial transactions and scores each one in real time: "How likely is this transaction to be fraudulent", and it explains "why"
- Get answers for the questions posed in plain English pulled directly from SEC documents.

## Tech Stack

|       Tool       |        Why we use it       |
|------------------|----------------------------|
|       Python     |Since most of the ML and DS libraries are written in Python, it's the most standard language for AI/ML projects|
|       Git        | For version control, tracks every change to the code and allows us to go back to any previous version|
|       Docker      |To run infrastructure locally, for its ease of compatability across all the alternatives on different machines|
|       VS Code     |Coding tool used for this project, as its light weight and has built-in extensions|
|       pip         |Installs and manages all Python libraries the project depends on|



## Project Structure

finsight-ai/
├── data_pipeline/
│   └── generators/   
│   │    └── transaction_generator.py  → generates 500k fake transactions for training
│   ├── kafka/
│   │     └── producer.py  → Streams transactions to kafka at 100 events/sec
│   └── spark/
│         └── feature_engineering.py   → Computes Fraud detection(PySpark) 
├── fraud_model/
│   ├── training/ 
│   │      └── train.py → XGBoost classifier with Optuna tuning and MLflow trackig
│   ├── evaluation/    
│   │      └── shap_analysis.py  → SHAP explainability, per-transaction feature importance
│   ├── serving/
│   │      └── predictor.py →  loads model, scores the transactions and stores the results in memory
│   └── artifacts/     → 
├── rag_pipeline/
│   ├── ingestion/ 
│   │       ├── sec_loader.py    → downloads SEC filings(10-K, 10-Q, 8-K) for 10 companies
│   │       ├── chunker.py → splits filings into 512-char overlapping chunks
│   │       └── embedder.py  → converts text chunks to vectors, stores in FAISS index
│   └── generation/    → 
├── api/
│   └── routers/       → 
├── infrastructure/
│   └── docker/        → 
├── tests/
│   └── unit/          → 
└── notebooks/         →




## Status

- [x] Environment setup
- [x] Folder structure created
- [x] requirements.txt created
- [x] Docker Compose configured
- [x] Dockerfile created
- [x] Transaction data generator created
- [x] 500k synthetic transactions generated
- [x] Kafka producer created
- [x] Spark feature engineering job created
- [x] XGBoost fraud model training script screated
- [x] SHAP explainability module created
- [x] Fraud model serving layer created
- [x] SEC filing loader created
- [x] Document chunker created
- [x] FAISS embedder created

