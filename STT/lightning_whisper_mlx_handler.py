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

        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")

        # More thorough warmup with varying input sizes
        dummy_inputs = [
            np.array([0] * size) for size in [512, 1024, 2048]
        ]
        for dummy_input in dummy_inputs:
            _ = self.model.transcribe(dummy_input)["text"].strip()

    def process(self, spoken_prompt):
        logger.debug("infering whisper...")

        global pipeline_start
        pipeline_start = perf_counter()

        # Use cached language if available
        audio_hash = hash(spoken_prompt.tobytes())
        
        if self.start_language != 'auto':
            transcription_dict = self.model.transcribe(spoken_prompt, language=self.start_language)
        else:
            if audio_hash in self.language_cache:
                language_code = self.language_cache[audio_hash]
                transcription_dict = self.model.transcribe(spoken_prompt, language=language_code)
            else:
                transcription_dict = self.model.transcribe(spoken_prompt)
                language_code = transcription_dict["language"]
                
                if language_code in SUPPORTED_LANGUAGES:
                    self.last_language = language_code
                    self.language_cache[audio_hash] = language_code
                else:
                    logger.warning(f"Whisper detected unsupported language: {language_code}")
                    language_code = self.last_language if self.last_language in SUPPORTED_LANGUAGES else "en"
                    transcription_dict = self.model.transcribe(spoken_prompt, language=language_code)

        # Manage memory more efficiently
        self.transcription_count += 1
        if self.transcription_count >= self.cache_clear_interval:
            torch.mps.empty_cache()
            self.language_cache.clear()
            self.transcription_count = 0
            
        pred_text = transcription_dict["text"].strip()
        language_code = transcription_dict["language"]

        logger.debug("finished whisper inference")
        console.print(f"[yellow]USER: {pred_text}")
        logger.debug(f"Language Code Whisper: {language_code}")

        if self.start_language == "auto":
            language_code += "-auto"
                    
        yield (pred_text, language_code)
