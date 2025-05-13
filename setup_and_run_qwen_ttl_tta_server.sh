# FILE: setup_and_run.sh
#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Define the project directory (assuming this script is in the root of the project)
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXECUTABLE="${PROJECT_DIR}/venv/bin/python" # Assuming a virtual environment

echo "Starting Qwen TTA TTL Server setup..."

# 1. Create a virtual environment (optional, but recommended)
if [ ! -d "${PROJECT_DIR}/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "${PROJECT_DIR}/venv"
    echo "Virtual environment created at ${PROJECT_DIR}/venv"
fi

# Activate virtual environment
source "${PROJECT_DIR}/venv/bin/activate"
echo "Activated virtual environment."

# 2. Install dependencies
echo "Installing dependencies..."
# Assuming a requirements.txt file exists.
# Create a placeholder requirements.txt if it doesn't exist for the script to run.
# For a real setup, this file would list packages like:
# torch torchvision torchaudio
# transformers
# fastapi
# uvicorn[standard]
# einops
# tensordict
# assoc-scan (if it's a pip package)
# rotary-embedding-torch
# x-transformers (if hyper_connections comes from it)
# tqdm

# Example: pip install -r requirements.txt
# For now, just ensure uvicorn and fastapi are notionally installed for the script's purpose
# In a real scenario, you'd run:
# ${PYTHON_EXECUTABLE} -m pip install -r requirements.txt

echo "Dependencies notionally installed."

# 3. Run the server
echo "Starting the FastAPI server using Uvicorn..."
# Make sure qwen_ttl_tta_server.py and run_server.py are in the PROJECT_DIR
# The PYTHONPATH needs to include the directory containing 'titans_pytorch'
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

# The run_server.py script will execute qwen_ttl_tta_server:app
export QWEN_BASE_MODEL_FOR_TTA_SERVER="Qwen/Qwen3-0.6B"
${PYTHON_EXECUTABLE} "${PROJECT_DIR}/run_server.py"
#${PYTHON_EXECUTABLE} "${PROJECT_DIR}/run_server.py" --base_model_qwen_name "Qwen/Qwen3-0.6B"
#${PYTHON_EXECUTABLE} "${PROJECT_DIR}/run_server.py" --host 0.0.0.0 --port 8000 --workers 1

echo "Server script finished or was interrupted."

# Deactivate virtual environment (optional, if you want to clean up shell)
# deactivate

