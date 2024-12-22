import logging
from time import perf_counter
from baseHandler import BaseHandler
from lightning_whisper_mlx import LightningWhisperMLX
import numpy as np
from rich.console import Console
from copy import copy
import torch

logger = logging.getLogger(__name__)

console = Console()

SUPPORTED_LANGUAGES = [
    "en",
    "de",
    "es",
]


class LightningWhisperSTTHandler(BaseHandler):
    """
    Handles the Speech To Text generation using a Whisper model.
    """

    def setup(
        self,
        model_name="distil-large-v3",
        device="mps",
        torch_dtype="float16",
        compile_mode=None,
        language=None,
        gen_kwargs={},
    ):
        if len(model_name.split("/")) > 1:
            model_name = model_name.split("/")[-1]
        self.device = device
        # Increase batch size for better throughput
        self.model = LightningWhisperMLX(model=model_name, batch_size=12, quant=None)
        self.start_language = language
        self.last_language = language
        # Cache for language detection
        self.language_cache = {}
        # Counter for memory management
        self.transcription_count = 0
        self.cache_clear_interval = 50  # Clear cache every 50 transcriptions

        logger.info(f"Initialized WhisperMLX with model: {model_name}, device: {device}")
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # More thorough warmup with varying input sizes
        dummy_inputs = [
            np.array([0] * size) for size in [512, 1024, 2048]
        ]
        for i, dummy_input in enumerate(dummy_inputs):
            result = self.model.transcribe(dummy_input)["text"].strip()
            logger.debug(f"Warmup {i+1}/3: Input size {len(dummy_input)}, Output: '{result}'")

    def process(self, spoken_prompt):
        logger.debug("Starting WhisperMLX inference...")
        logger.debug(f"Input audio length: {len(spoken_prompt)} samples")

        global pipeline_start
        pipeline_start = perf_counter()

        # Use cached language if available
        audio_hash = hash(spoken_prompt.tobytes())
        
        if self.start_language != 'auto':
            logger.debug(f"Using fixed language: {self.start_language}")
            transcription_dict = self.model.transcribe(spoken_prompt, language=self.start_language)
        else:
            if audio_hash in self.language_cache:
                language_code = self.language_cache[audio_hash]
                logger.debug(f"Using cached language: {language_code}")
                transcription_dict = self.model.transcribe(spoken_prompt, language=language_code)
            else:
                logger.debug("Detecting language...")
                transcription_dict = self.model.transcribe(spoken_prompt)
                language_code = transcription_dict["language"]
                logger.debug(f"Detected language: {language_code}")
                
                if language_code in SUPPORTED_LANGUAGES:
                    self.last_language = language_code
                    self.language_cache[audio_hash] = language_code
                    logger.debug(f"Language {language_code} is supported, caching")
                else:
                    logger.warning(f"Whisper detected unsupported language: {language_code}")
                    language_code = self.last_language if self.last_language in SUPPORTED_LANGUAGES else "en"
                    logger.debug(f"Falling back to language: {language_code}")
                    transcription_dict = self.model.transcribe(spoken_prompt, language=language_code)

        # Manage memory more efficiently
        self.transcription_count += 1
        if self.transcription_count >= self.cache_clear_interval:
            logger.debug("Clearing memory cache")
            torch.mps.empty_cache()
            self.language_cache.clear()
            self.transcription_count = 0
            
        pred_text = transcription_dict["text"].strip()
        language_code = transcription_dict["language"]

        # Calculate inference time
        inference_time = perf_counter() - pipeline_start
        logger.debug(f"WhisperMLX inference completed in {inference_time:.2f}s")
        
        # Print transcription details
        console.print("[blue]Speech Recognition Details:[/blue]")
        console.print(f"[yellow]Transcribed Text: {pred_text}")
        console.print(f"[yellow]Detected Language: {language_code}")
        console.print(f"[yellow]Inference Time: {inference_time:.2f}s")
        console.print(f"[yellow]Audio Length: {len(spoken_prompt)/16000:.2f}s")  # Assuming 16kHz sample rate

        logger.debug(f"Language Code: {language_code}")
        logger.debug(f"Transcription: '{pred_text}'")

        if self.start_language == "auto":
            language_code += "-auto"
                    
        yield (pred_text, language_code)
