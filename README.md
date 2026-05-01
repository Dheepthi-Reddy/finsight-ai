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


finsight-ai/
├── data_pipeline/
│   ├── generators/    → 
│   ├── kafka/         → 
│   └── spark/         → 
├── fraud_model/
│   ├── training/      → 
│   ├── evaluation/    → 
│   ├── serving/       → 
│   └── artifacts/     → 
├── rag_pipeline/
│   ├── ingestion/     → 
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