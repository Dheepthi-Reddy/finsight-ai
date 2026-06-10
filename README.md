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
```
finsight-ai/
├── data_pipeline/
│   ├── generators/   
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
│   ├── generation/
│   │       ├── bedrock_client.py  → connects to Claude via AWS Bedrock, stub mode for local dev
│   │       ├── prompt_templates.py  → system and user prompts for compliance assistant and fraud explainer
│   │       └── response_evaluator.py  → evaluates Claude responses with BLEU, ROUGE-L and faithfulness scores
│   └── chain.py → main RAG pipeline, wires retrieval and Claude generation
├── api/
│   ├── routers/ 
│   │       ├── health.py   → to check if the service is running fine
│   │       ├── fraud.py    → endpoint to get fraud scoring
│   │       ├── query.py    → endpoint for the RAG compliance assistant.
│   │       └── explain.py  → endpoint for SHAP explanations.
│   ├── config.py     → centralized app configuration from .env
│   └── main.py        → FastAPI app entry point, registers all routers
├── infrastructure/
│   └── docker/        → 
├── tests/
│   └── unit/          → 
└── notebooks/         →

```


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
- [x] AWS Bedrock client created
- [x] Prompt templates created
- [x] Response evaluator created
- [x] RAG chain created
- [x] API config created
- [x] FastAPI main app created
- [x] Health end point created
- [x] Fraud endpoint created
- [x] Query endpoint created
- [x] Explain endpoint created
- [x] GitHub Actions CI pipeline created
