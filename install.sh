#!/bin/bash

# Exit on error
set -e

echo "Setting up TalkLLM..."

# Check if Python 3.12 is installed
if ! command -v python3.12 &> /dev/null; then
    echo "Error: Python 3.12 is required but not installed."
    echo "Please install Python 3.12 using:"
    echo "brew install python@3.12"
    exit 1
fi

# Create and activate virtual environment with Python 3.12
echo "Creating Python virtual environment..."
python3.12 -m venv .venv
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
python3.12 -m pip install --upgrade pip

# Install dependencies with performance optimizations
echo "Installing core dependencies (this may take a few minutes)..."

echo "Installing nltk..."
uv pip install "nltk>=3.8.0"

echo "Installing torch and torchaudio..."
uv pip install --no-deps "torch==2.4.0" "torchaudio==2.4.0"  # Specific versions optimized for Mac

echo "Installing sounddevice..."
uv pip install "sounddevice>=0.4.6"

echo "Installing lightning-whisper-mlx..."
uv pip install "lightning-whisper-mlx>=0.0.10"

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

# Check Ollama installation and status
echo "Checking Ollama setup..."
if command -v ollama &> /dev/null; then
    echo "Ollama is installed"
    
    # Check if Ollama service is running
    if pgrep -x "ollama" > /dev/null; then
        echo "Ollama service is running"
        
        # Check if qwen2.5:7b model is pulled
        if ollama list | grep -q "qwen2.5:7b"; then
            echo "Qwen2.5:7b model is already pulled"
        else
            echo "Pulling Qwen2.5:7b model..."
            ollama pull qwen2.5:7b
        fi
    else
        echo "Ollama service is not running"
        echo "Starting Ollama service..."
        ollama serve > /dev/null 2>&1 &
        sleep 5  # Wait for service to start
        
        echo "Pulling Qwen2.5:7b model..."
        ollama pull qwen2.5:7b
    fi
else
    echo "Ollama is not installed"
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
    
    echo "Starting Ollama service..."
    ollama serve > /dev/null 2>&1 &
    sleep 5  # Wait for service to start
    
    echo "Pulling Qwen2.5:7b model..."
    ollama pull qwen2.5:7b
fi

echo "Setup complete! You can now run the optimized pipeline with:"
echo ""
echo "source .venv/bin/activate && python s2s_pipeline.py \\"
echo "    --local_mac_optimal_settings \\"
echo "    --mode local \\"
echo "    --device mps \\"
echo "    --llm ollama \\"
echo "    --ollama_model qwen2.5:7b"
echo ""
echo "This configuration uses:"
echo "- Hardware-accelerated Silero VAD with MPS"
echo "- LightningWhisperMLX for STT (optimized for Apple Silicon)"
echo "- Ollama with qwen2.5:7b model for LLM"
echo "- ChatTTS for high-quality speech synthesis"
echo "- Automatic language detection"
echo ""
echo "Performance optimizations include:"
echo "- Pre-allocated buffers for audio processing"
echo "- Efficient tensor operations"
echo "- Smart cache management"
echo "- Optimized device placement (MPS/CPU)"
echo "- Streamlined audio processing pipeline"
echo "- Language detection caching"
