#!/bin/bash

# Install dependencies for Mac
echo "Installing dependencies..."
uv pip install -r requirements_mac.txt

# Install unidic for MeloTTS
echo "Installing unidic for MeloTTS..."
python -m unidic download

echo "Setup complete! You can now run the pipeline with optimal Mac settings using:"
echo ""
echo "python s2s_pipeline.py \\"
echo "    --local_mac_optimal_settings \\"
echo "    --device mps \\"
echo "    --stt_model_name large-v3 \\"
echo "    --language auto \\"
echo "    --mlx_lm_model_name mlx-community/Qwen2.5-72B-Instruct-bf16"
echo ""
echo "This configuration uses:"
echo "- LightningWhisperMLX for STT"
echo "- MLX LM for language model"
echo "- MeloTTS for TTS"
echo "- MPS for hardware acceleration"
echo "- Automatic language detection"
