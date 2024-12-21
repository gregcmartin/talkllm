#!/bin/bash

# Exit on error
set -e

echo "Setting up TalkLLM..."

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo "Error: Python 3.11 is required but not installed."
    echo "Please install Python 3.11 using:"
    echo "brew install python@3.11"
    exit 1
fi

# Create and activate virtual environment with Python 3.11
echo "Creating Python virtual environment..."
python3.11 -m venv .venv
source .venv/bin/activate

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv package installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Ensure uv is in PATH after installation
if ! command -v uv &> /dev/null; then
    echo "Error: uv installation failed or not in PATH"
    echo "Please try running: export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    exit 1
fi

# Ensure we're using the latest pip
echo "Upgrading pip..."
python3.11 -m pip install --upgrade pip

# Install dependencies one by one
echo "Installing core dependencies (this may take a few minutes)..."

echo "Installing nltk..."
uv pip install "nltk>=3.8.0"

echo "Installing torch and torchaudio..."
uv pip install "torch==2.4.0" "torchaudio==2.4.0"  # Specific versions known to work on Mac

echo "Installing sounddevice..."
uv pip install "sounddevice>=0.4.6"

echo "Installing lightning-whisper-mlx..."
uv pip install "lightning-whisper-mlx>=0.0.10"

echo "Installing mlx-lm..."
uv pip install "mlx-lm>=0.0.1"

echo "Installing ChatTTS..."
uv pip install "ChatTTS>=0.0.1"

echo "Installing remaining core dependencies..."
uv pip install "funasr>=1.0.0" "faster-whisper>=1.0.0" "modelscope>=1.0.0" "deepfilternet>=0.5.0" "openai>=1.0.0"

# Install git dependencies separately
echo "Installing git-based dependencies..."
echo "Installing parler-tts..."
uv pip install git+https://github.com/huggingface/parler-tts.git

echo "Installing moonshine..."
uv pip install git+https://github.com/andimarafioti/moonshine.git

echo "Setup complete! You can now run the pipeline with optimal Mac settings using:"
echo ""
echo "source .venv/bin/activate && python s2s_pipeline.py \\"
echo "    --local_mac_optimal_settings \\"
echo "    --device mps \\"
echo "    --stt_model_name large-v3 \\"
echo "    --language auto \\"
echo "    --mlx_lm_model_name mlx-community/Qwen2.5-72B-Instruct-bf16 \\"
echo "    --tts chatTTS"
echo ""
echo "This configuration uses:"
echo "- LightningWhisperMLX for STT"
echo "- MLX LM for language model (Qwen2.5-72B-Instruct-bf16)"
echo "- ChatTTS for TTS"
echo "- MPS for hardware acceleration"
echo "- Automatic language detection"
