# FinSight AI — Setup Journal

> **What this document is:** A step-by-step record of every decision made while setting up the FinSight AI project. Written so that anyone — coder or not — can understand what was done, why it was done, and how to reproduce it from scratch.

---

## Table of Contents

1. [What is FinSight AI?](#1-what-is-finsight-ai)
2. [The Big Picture — How the Pieces Fit Together](#2-the-big-picture)
3. [Tools We Need and Why](#3-tools-we-need-and-why)
4. [Checking What Was Already Installed](#4-checking-what-was-already-installed)
5. [Installing Docker Desktop](#5-installing-docker-desktop)
6. [Installing VS Code](#6-installing-vs-code)
7. [Making Docker Permanently Available](#7-making-docker-permanently-available)
8. [Creating the GitHub Repository](#8-creating-the-github-repository)
9. [Cloning the Repository to Your Mac](#9-cloning-the-repository-to-your-mac)
10. [Setting Up the Python Virtual Environment](#10-setting-up-the-python-virtual-environment)
11. [Creating the Project Folder Structure](#11-creating-the-project-folder-structure)
12. [What Comes Next](#12-what-comes-next)

---

## 1. What is FinSight AI?

FinSight AI is a production-grade machine learning platform for the financial industry. It does two things simultaneously:

**Thing 1 — Fraud Detection**
It watches a live stream of financial transactions (like credit card purchases) and scores each one in real time: *"How likely is this transaction to be fraudulent?"* It also explains *why* it thinks so — which specific factors (unusual hour, high-risk merchant, large amount) contributed to the suspicion.

**Thing 2 — Compliance Assistant**
Analysts can type plain-English questions like *"What compliance risks does JPMorgan disclose in their latest 10-K filing?"* and get accurate, sourced answers pulled directly from SEC documents — instead of spending hours reading filings manually.

**Why build this?**
This project is a portfolio piece demonstrating the full stack of modern AI/ML engineering: data ingestion, model training, explainability, large language models, cloud deployment, and automated testing. Every technology used here appears on a real ML engineer's resume.

---

## 2. The Big Picture

Before touching any code, it helps to understand how all the pieces connect.

```
Raw Data (transactions, SEC filings)
        │
        ▼
  Data Pipeline
  (Kafka + Spark)          ← collects and cleans data in real time
        │
        ▼
   ML Models
  ┌─────────────────────────────────┐
  │  Fraud Model (XGBoost)          │  ← decides if a transaction is fraud
  │  RAG Pipeline (LangChain)       │  ← answers compliance questions
  └─────────────────────────────────┘
        │
        ▼
   FastAPI Server                   ← exposes everything as web endpoints
        │
        ▼
   Docker Container                 ← packages everything to run anywhere
        │
        ▼
   GitHub Actions CI/CD             ← automatically tests every code change
```

Every layer depends on the one above it. This is why we build in order — data first, models second, API third, deployment last.

---

## 3. Tools We Need and Why

| Tool | What it is | Why we need it |
|---|---|---|
| **Python 3.13** | Programming language | All ML code is written in Python |
| **Git** | Version control system | Tracks every change to the code; enables collaboration |
| **pip** | Python package installer | Downloads and installs Python libraries |
| **Docker** | Container platform | Runs Kafka, PostgreSQL, and MLflow locally without complex installation |
| **VS Code** | Code editor | Where we write and navigate all the code |
| **GitHub** | Cloud Git repository | Stores the code online; runs automated tests via GitHub Actions |

---

## 4. Checking What Was Already Installed

Before installing anything, we checked what was already on the Mac. This avoids reinstalling things that already work and helps identify version conflicts early.

**Command used:**
```bash
python3 --version
git --version
pip3 --version
docker --version
code --version
```

**Why run these?**
Each command asks the tool to report its version. If the tool is installed, it prints a version number. If it is not installed, the terminal prints `command not found`.

**Results:**

| Tool | Result | Meaning |
|---|---|---|
| Python | `Python 3.13.7` | ✅ Already installed via Homebrew |
| Git | `git version 2.39.5 (Apple Git-154)` | ✅ Apple's built-in Git |
| pip | `pip 25.2 from /opt/homebrew/...` | ✅ Installed via Homebrew |
| Docker | `command not found` | ❌ Needs installation |
| VS Code | `command not found` | ❌ Needs installation |

**Note on Python version:** The project was designed for Python 3.11, but 3.13.7 is fully compatible. No changes were needed.

---

## 5. Installing Docker Desktop

**What is Docker?**
Docker is a tool that lets you run software in isolated "containers" — think of a container like a self-contained box that has everything a program needs to run (its own mini operating system, its own libraries, its own settings). This means Kafka, PostgreSQL, and MLflow can all run on your laptop without you having to install and configure each one manually.

**Why do we need it for this project?**
FinSight needs several infrastructure services running locally during development:
- **Kafka** — receives the stream of transactions
- **PostgreSQL** — stores processed data
- **MLflow** — tracks ML experiments and model versions

Without Docker, setting each of these up on a Mac is a multi-hour process. With Docker, one command starts all three.

**Installation steps:**
1. Went to `docker.com/products/docker-desktop`
2. Downloaded the **Apple Silicon** version (for M1/M2/M3 MacBook Air)
3. Opened the `.dmg` file and dragged Docker to Applications
4. Launched Docker from Applications and waited for the whale icon in the menu bar to stop animating — this means the Docker engine is running

**Important:** Docker must be running (whale icon visible in menu bar) every time you work on this project.

---

## 6. Installing VS Code

**What is VS Code?**
VS Code (Visual Studio Code) is a free code editor made by Microsoft. It has syntax highlighting, error detection, an integrated terminal, and thousands of extensions for Python, Docker, Git, and more.

**Why VS Code and not something else?**
It is the most widely used editor in the ML/data science community. It has excellent Python support, integrates directly with GitHub, and the integrated terminal means you never need to switch windows between your editor and the command line.

**Installation steps:**
1. Went to `code.visualstudio.com`
2. Downloaded for Mac
3. Opened VS Code, then enabled the `code` terminal command:
   - Pressed `Cmd + Shift + P`
   - Typed `Shell Command`
   - Clicked **Install 'code' command in PATH**

**What does "install in PATH" mean?**
The PATH is a list of folders your terminal searches when you type a command. By adding VS Code to the PATH, typing `code .` in any terminal opens that folder in VS Code. Without this, you'd have to open VS Code manually every time.

**Verified with:**
```bash
code --version
# Output: 1.118.1
```

---

## 7. Making Docker Permanently Available

**The problem:**
After installing Docker Desktop, the `docker` command only worked in new Terminal windows — not in windows that were already open. And after testing, it became clear that the command would fail in some terminals because Docker's executables were not in the system PATH.

**Why does this happen?**
When Docker installs, it places its command-line tools in `/Applications/Docker.app/Contents/Resources/bin`. The terminal only knows to look in certain folders by default. Unless Docker's folder is added to those default locations, the terminal can't find the `docker` command.

**The fix:**
We added Docker's location to `~/.zshrc` — a configuration file that runs every time a new terminal window opens on a Mac with zsh (the default shell since macOS Catalina).

**Command used:**
```bash
echo 'export PATH="$PATH:/usr/local/bin:/Applications/Docker.app/Contents/Resources/bin"' >> ~/.zshrc
source ~/.zshrc
```

**Breaking this down:**
- `echo '...'` — prints the text inside the quotes
- `>> ~/.zshrc` — appends that text to the end of the `.zshrc` file (instead of `>` which would overwrite)
- `export PATH="$PATH:..."` — tells the terminal to look in the new folder in addition to all the existing ones
- `source ~/.zshrc` — reloads the config file so changes take effect immediately without restarting Terminal

**Verified with:**
```bash
docker --version
# Output: Docker version 29.4.1, build 055a478
```

---

## 8. Creating the GitHub Repository

**What is a repository?**
A repository (repo) is a project folder that Git tracks. Every change you make to any file gets recorded with a timestamp and a message describing what changed. You can go back to any previous version at any time.

**Why host it on GitHub?**
GitHub stores your repository in the cloud so it is never lost, lets you share it with others (including employers), and provides GitHub Actions — a system that automatically runs tests every time you push new code.

**Steps taken:**
1. Went to `github.com` and signed in
2. Clicked **+** (top right) → **New repository**
3. Named it `finsight-ai`
4. Set visibility to **Public** — important for a portfolio project so employers can see it
5. Checked **Add a README file** — creates an initial file so the repo is not empty
6. Selected **Python** under Add .gitignore — this tells Git to ignore common Python files that should never be committed (like compiled `.pyc` files, virtual environment folders, etc.)
7. Clicked **Create repository**

**Why all lowercase with hyphens?**
Repository names in GitHub become part of a URL: `github.com/username/finsight-ai`. Lowercase with hyphens is the universal convention because it works reliably on all operating systems (Linux servers are case-sensitive, so `FinSight-AI` and `finsight-ai` would be different things on a server). The project can still be called "FinSight AI" everywhere else — README, resume, presentations.

---

## 9. Cloning the Repository to Your Mac

**What is cloning?**
Cloning downloads a copy of the GitHub repository to your local machine and sets up the connection between the local copy and the GitHub copy. Any changes you make locally can be "pushed" back up to GitHub.

**Steps taken:**
```bash
cd ~/Desktop
git clone https://github.com/YOUR_USERNAME/finsight-ai.git
cd finsight-ai
code .
```

**Breaking this down:**
- `cd ~/Desktop` — navigates to the Desktop folder (`~` means your home folder)
- `git clone <url>` — downloads the repository and creates a `finsight-ai` folder
- `cd finsight-ai` — enters that folder
- `code .` — opens the current folder (`.`) in VS Code

**Result:** VS Code opened with the `finsight-ai` folder. The project's location on disk is `/Users/dheepthireddy/Documents/finsight-ai`.

---

## 10. Setting Up the Python Virtual Environment

**What is a virtual environment?**
A virtual environment is an isolated Python installation just for this project. It has its own copy of Python and its own set of installed libraries that are completely separate from any other Python projects on your Mac.

**Why do we need one?**
Different projects need different versions of the same library. For example, Project A might need `pandas 1.5` and Project B might need `pandas 2.2`. Without virtual environments, installing one would break the other. With a virtual environment, each project has its own sandbox.

**What went wrong first:**
The initial attempt was interrupted mid-way through (a `KeyboardInterrupt` — pressing Ctrl+C accidentally). This left a broken virtual environment that had the folder structure but no pip installed inside it.

**The fix:**
```bash
# Step 1: Delete the broken environment
rm -rf venv

# Step 2: Create a fresh one, skipping the pip installation step
#         (--without-pip avoids the step that was failing)
python3 -m venv venv --without-pip

# Step 3: Activate the environment
source venv/bin/activate

# Step 4: Install pip manually using the official installer
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py
rm get-pip.py
```

**How to know it worked:**
The terminal prompt changes to show `(venv)` at the start:
```bash
# Before activation:
dheepthireddy@MacBookAir finsight-ai %

# After activation:
(venv) dheepthireddy@MacBookAir finsight-ai %
```

And `which python3` points inside the project:
```bash
which python3
# Output: /Users/dheepthireddy/Documents/finsight-ai/venv/bin/python3
```

**Important:** Every time you open a new terminal and want to work on this project, you must re-activate the virtual environment:
```bash
cd ~/Documents/finsight-ai
source venv/bin/activate
```

**Final state:**
```
Python: 3.13.7 (inside venv)
pip:    26.1   (inside venv)
```

---

## 11. Creating the Project Folder Structure

**Why plan the folder structure before writing code?**
A well-organised folder structure communicates the architecture of the system to anyone who opens the repo. It also prevents the common mistake of writing everything in one folder and having to reorganise later when the project grows.

Each folder corresponds to one layer of the system architecture.

**Command used:**
```bash
mkdir -p data \
  data_pipeline/generators \
  data_pipeline/kafka \
  data_pipeline/spark \
  fraud_model/training \
  fraud_model/evaluation \
  fraud_model/serving \
  fraud_model/artifacts \
  rag_pipeline/ingestion \
  rag_pipeline/generation \
  api/routers \
  infrastructure/docker \
  tests/unit \
  notebooks \
  .github/workflows
```

**What does `mkdir -p` do?**
- `mkdir` — make directory (create a folder)
- `-p` — create parent folders too if they don't exist, and don't error if the folder already exists
- The `\` at the end of each line is a line continuation — it tells the terminal the command continues on the next line

**What each folder is for:**

| Folder | Purpose |
|---|---|
| `data/` | Raw and processed datasets (not committed to Git — too large) |
| `data_pipeline/generators/` | Scripts that create synthetic training data |
| `data_pipeline/kafka/` | Kafka producer that simulates a live transaction stream |
| `data_pipeline/spark/` | PySpark jobs for feature engineering |
| `fraud_model/training/` | XGBoost model training script with Optuna tuning |
| `fraud_model/evaluation/` | SHAP explainability analysis |
| `fraud_model/serving/` | Model inference wrapper (loads model, scores transactions) |
| `fraud_model/artifacts/` | Saved model files (not committed to Git) |
| `rag_pipeline/ingestion/` | Downloads SEC filings and embeds them into a vector database |
| `rag_pipeline/generation/` | Connects to Claude via AWS Bedrock to generate answers |
| `api/routers/` | FastAPI endpoints (/score, /query, /explain, /health) |
| `infrastructure/docker/` | Dockerfile and docker-compose.yml |
| `tests/unit/` | Automated tests for every component |
| `notebooks/` | Jupyter notebooks for exploration and prototyping |
| `.github/workflows/` | GitHub Actions CI/CD pipeline definitions |

---

## 12. What Comes Next

The environment is fully set up. The next steps in order are:

1. **Create `.env` file** — stores secret keys and configuration (AWS credentials, Pinecone API key, etc.) without committing them to Git
2. **Create `requirements.txt`** — lists every Python library the project needs so anyone can reproduce the environment with one command
3. **Start infrastructure** — run `docker compose up` to start Kafka, PostgreSQL, and MLflow
4. **Generate training data** — run the synthetic transaction generator (500,000 transactions)
5. **Feature engineering** — run the PySpark job to compute derived features
6. **Train the fraud model** — XGBoost with Optuna hyperparameter search (50 trials, ~45 minutes)
7. **Build the RAG index** — download SEC filings, chunk them, embed them into FAISS
8. **Start the API** — run FastAPI server, test all endpoints
9. **Connect to AWS Bedrock** — configure credentials, enable real Claude responses
10. **Run tests** — pytest with coverage report
11. **Push to GitHub** — CI pipeline triggers automatically

---

## Appendix — Command Reference

Quick reference for the most commonly used commands in this project.

```bash
# Activate virtual environment (run this every time you open a new terminal)
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Start all infrastructure (Kafka, Postgres, MLflow)
docker compose -f infrastructure/docker/docker-compose.yml up -d

# Stop all infrastructure
docker compose -f infrastructure/docker/docker-compose.yml down

# Generate training data
python data_pipeline/generators/transaction_generator.py --n 500000

# Train fraud model
python fraud_model/training/train.py --features data/features.parquet --trials 50

# Build RAG index
python rag_pipeline/ingestion/embedder.py --mode faiss

# Start API server
uvicorn api.main:app --reload --port 8000

# Run tests
pytest tests/ -v --cov=. --cov-report=term-missing

# Check git status
git status

# Save your work to GitHub
git add .
git commit -m "description of what you changed"
git push origin main
```

---

*Last updated: Day 1 of build — environment setup complete.*
*Next session: requirements.txt, .env, Docker infrastructure, data generation.*
