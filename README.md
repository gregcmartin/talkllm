# TalkLLM

This repository is a fork and enhancement of the [Hugging Face speech-to-speech project](https://github.com/huggingface/speech-to-speech), which provides an open-source and modular approach to speech-to-speech conversation.

## Original Project

The original project by Hugging Face implements a speech-to-speech cascaded pipeline with the following components:
1. Voice Activity Detection (VAD)
2. Speech to Text (STT)
3. Language Model (LM)
4. Text to Speech (TTS)

For detailed information about the original project's features, setup instructions, and usage, please visit the [original repository](https://github.com/huggingface/speech-to-speech).

## Enhancements

This fork maintains the core functionality of the original project while adding:
- [List your specific enhancements or modifications here]

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
2. Set up MeloTTS with unidic
3. Provide the command to run the pipeline with optimal Mac settings:
   - LightningWhisperMLX for STT
   - MLX LM for language model
   - MeloTTS for TTS
   - MPS for hardware acceleration
   - Automatic language detection

## Alternative Setup & Usage

For other setup options and detailed configuration, please refer to the [original project documentation](https://github.com/huggingface/speech-to-speech). The pipeline can be run using:
- Server/Client approach
- Local approach
- Docker Server approach

## Credits

This project builds upon the excellent work done by the Hugging Face team and their speech-to-speech project. All original citations and acknowledgments from the base project are maintained and appreciated:

- Silero VAD
- Distil-Whisper
- Parler-TTS

For the complete list of citations and acknowledgments, please refer to the original repository.
