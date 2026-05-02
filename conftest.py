"""
Pytest root configuration — adds the project root to sys.path so that
test files can import project modules (api, fraud_model, rag_pipeline, etc.)
without installing the project as a package.

Placing conftest.py at the project root is the standard pytest convention for
this setup: pytest detects it as the rootdir and processes it before any test
collection begins.
"""
import sys
from pathlib import Path

# Insert the project root at the front of sys.path so project imports take
# precedence over any identically-named packages that might be installed
# in the virtual environment from other projects.
sys.path.insert(0, str(Path(__file__).parent))
