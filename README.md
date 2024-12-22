# TalkLLM

This repository is a fork and enhancement of the [Hugging Face speech-to-speech project](https://github.com/huggingface/speech-to-speech), which provides an open-source and modular approach to speech-to-speech conversation.

## Original Project

The original project by Hugging Face implements a speech-to-speech cascaded pipeline with the following components:
1. Voice Activity Detection (VAD)
2. Speech to Text (STT)
3. Language Model (LM)
4. Text to Speech (TTS)

For detailed information about the original project's features, setup instructions, and usage, please visit the [original repository](https://github.com/huggingface/speech-to-speech).

## Enhancements & Optimizations

This fork maintains the core functionality of the original project while adding significant optimizations:

### Voice Activity Detection (VAD)
- Hardware-accelerated Silero VAD with MPS
- Pre-allocated tensor buffers
- Optimized audio processing pipeline
- Improved silence detection and speech segmentation

### Speech to Text (STT)
- LightningWhisperMLX optimized for Apple Silicon
- Enhanced language detection with caching
- Efficient batch processing
- Thorough model warmup

### Language Model (LLM)
- Integrated Ollama support with qwen2.5:7b model
- Natural, concise conversation style
- Optimized generation parameters
- Efficient chat history management
- Streamlined response handling

### Text to Speech (TTS)
- ChatTTS with multiple voice types:
  - neutral: Balanced, natural speaking voice
  - warm: Friendly, approachable voice
  - professional: Clear, authoritative voice
  - casual: Relaxed, conversational voice
  - energetic: Dynamic, enthusiastic voice
- Optimized audio processing
- Smart interruption handling
- Improved speech continuity

## Quick Setup

For Mac users, we provide an install script with optimized settings:

```bash
# Make the install script executable
chmod +x install.sh

# Run the install script
./install.sh
```

This script will:
1. Install all required dependencies
2. Configure the pipeline components for optimal performance
3. Set up Ollama with the qwen2.5:7b model

## Running the Pipeline

### Basic Usage
```bash
# Run with default settings (neutral voice)
python s2s_pipeline.py --local_mac_optimal_settings --mode local --device mps --llm ollama --ollama_model qwen2.5:7b
```

### Selecting a Voice Type
```bash
# Run with a specific voice type
python s2s_pipeline.py --local_mac_optimal_settings --mode local --device mps --llm ollama --ollama_model qwen2.5:7b --tts_voice_type warm
```

Available voice types:
- neutral: Best for general conversation
- warm: Great for friendly interactions
- professional: Ideal for formal content
- casual: Perfect for informal chat
- energetic: Good for engaging presentations

## Performance Optimizations

The pipeline includes several optimizations for improved performance:

1. Memory Management
   - Pre-allocated buffers for audio processing
   - Efficient tensor operations
   - Smart cache management
   - Optimized device placement (MPS/CPU)

2. Processing Pipeline
   - Streamlined audio processing
   - Efficient text normalization
   - Optimized language detection
   - Improved response handling

3. Multi-language Support
   - Cached language detection
   - Pre-compiled language mappings
   - Efficient language switching
   - Optimized prompts for each language

## Requirements

- macOS with Apple Silicon (for MPS acceleration)
- Python 3.12+
- Ollama installed and running
- At least 16GB RAM recommended
- SSD storage recommended for model loading

## Credits

This project builds upon the excellent work done by the Hugging Face team and their speech-to-speech project. All original citations and acknowledgments from the base project are maintained and appreciated:

- Silero VAD for voice activity detection
- Distil-Whisper and LightningWhisperMLX for speech recognition
- Ollama for language model inference
- ChatTTS for speech synthesis

For the complete list of citations and acknowledgments, please refer to the original repository.
