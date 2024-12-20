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

## Setup & Usage

Please refer to the original project's documentation for setup and usage instructions. The basic steps remain the same:

1. Install dependencies:
```bash
# For most systems
uv pip install -r requirements.txt

# For Mac users
uv pip install -r requirements_mac.txt
```

2. Run the pipeline using one of the following approaches:
- Server/Client approach
- Local approach
- Docker Server approach

For detailed instructions and configuration options, please see the [original documentation](https://github.com/huggingface/speech-to-speech).

## Credits

This project builds upon the excellent work done by the Hugging Face team and their speech-to-speech project. All original citations and acknowledgments from the base project are maintained and appreciated:

- Silero VAD
- Distil-Whisper
- Parler-TTS

For the complete list of citations and acknowledgments, please refer to the original repository.
