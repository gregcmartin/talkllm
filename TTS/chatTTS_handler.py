import ChatTTS
import logging
import os
import pickle
import re
from pathlib import Path
from baseHandler import BaseHandler
import librosa
import numpy as np
from rich.console import Console
import torch

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

console = Console()

# Pre-compile regex patterns for text cleaning
PUNCTUATION_PATTERNS = {
    r'[?!]': '.',  # Replace ? and ! with periods
    r'[,:]': '',   # Remove commas and colons
    r';': '.',     # Convert semicolons to periods
    r'[-/]': ' ',  # Convert hyphens and slashes to spaces
    r'[()]': '',   # Remove parentheses
    r'&': ' and ', # Convert ampersands to 'and'
    r'\'': '',     # Remove apostrophes
    r'\s+': ' ',   # Normalize spaces
}

class ChatTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cpu",  # ChatTTS uses CPU for stability with certain operations
        gen_kwargs={},  # Unused
        stream=True,
        chunk_size=512,
        speaker_id=None,  # New parameter for speaker configuration
    ):
        self.should_listen = should_listen
        self.device = device
        self.model = ChatTTS.Chat()
        self.model.load(compile=False)  # Doesn't work for me with True
        self.chunk_size = chunk_size
        self.stream = stream
        
        # Pre-allocate reusable buffers
        self.audio_buffer = np.zeros(chunk_size, dtype=np.int16)
        self.padding_buffer = np.zeros(chunk_size, dtype=np.int16)
        
        # Cache for cleaned text
        self.text_cache = {}
        self.max_cache_size = 1000
        
        # Load or create speaker embedding
        self._setup_speaker_embedding()
            
        # Initialize inference parameters with the speaker embedding
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=self.speaker_embedding,
        )

    def _setup_speaker_embedding(self):
        """Setup speaker embedding with efficient file handling."""
        self.speaker_file = Path("speaker_embedding.pkl")
        try:
            if self.speaker_file.exists():
                with open(self.speaker_file, 'rb') as f:
                    self.speaker_embedding = pickle.load(f)
                logger.info("Loaded existing speaker voice")
            else:
                self._create_new_speaker()
        except Exception as e:
            logger.warning(f"Failed to load speaker embedding: {e}")
            self._create_new_speaker()

    def _create_new_speaker(self):
        """Create and save new speaker embedding."""
        self.speaker_embedding = self.model.sample_random_speaker()
        self._save_speaker_embedding()
        logger.info("Generated new speaker voice")

    def _save_speaker_embedding(self):
        """Save speaker embedding with atomic write."""
        temp_file = self.speaker_file.with_suffix('.tmp')
        try:
            with open(temp_file, 'wb') as f:
                pickle.dump(self.speaker_embedding, f)
            temp_file.replace(self.speaker_file)  # Atomic replace
        except Exception as e:
            logger.warning(f"Failed to save speaker embedding: {e}")
            if temp_file.exists():
                temp_file.unlink()  # Clean up temp file
        
    def set_speaker(self, speaker_id=None):
        """Change the speaker voice."""
        if speaker_id is not None:
            logger.warning("Specific speaker selection not yet supported")
            
        self._create_new_speaker()
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=self.speaker_embedding,
        )
        self.warmup()

    def warmup(self):
        """Warm up the model with a short text."""
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.infer("text")

    def clean_text(self, text):
        """Clean text using pre-compiled patterns and caching."""
        # Check cache first
        if text in self.text_cache:
            return self.text_cache[text]
        
        # Apply all regex patterns
        cleaned = text
        for pattern, replacement in PUNCTUATION_PATTERNS.items():
            cleaned = re.sub(pattern, replacement, cleaned)
        
        # Only allow letters, numbers, spaces, and periods
        cleaned = ''.join(c for c in cleaned if c.isalnum() or c.isspace() or c == '.')
        cleaned = cleaned.strip()
        
        # Update cache with LRU-style eviction
        if len(self.text_cache) >= self.max_cache_size:
            self.text_cache.pop(next(iter(self.text_cache)))
        self.text_cache[text] = cleaned
        
        return cleaned

    def _process_audio_chunk(self, chunk, orig_sr=24000, target_sr=16000):
        """Process audio chunk with efficient resampling and conversion."""
        # Resample using pre-allocated buffer when possible
        resampled = librosa.resample(chunk, orig_sr=orig_sr, target_sr=target_sr)
        
        # Convert to int16 efficiently
        audio_int16 = np.clip(resampled * 32768, -32768, 32767).astype(np.int16)
        
        return audio_int16

    def process(self, llm_input):
        # Extract and clean text
        llm_sentence = llm_input[0] if isinstance(llm_input, tuple) else llm_input
        llm_sentence = self.clean_text(llm_sentence)
        console.print(f"[green]ASSISTANT: {llm_sentence}")

        # Get audio generator
        wavs_gen = self.model.infer(
            llm_sentence, params_infer_code=self.params_infer_code, stream=self.stream
        )

        if self.stream:
            for gen in wavs_gen:
                if gen[0] is None or len(gen[0]) == 0:
                    self.should_listen.set()
                    return
                
                # Process audio chunk efficiently
                audio_chunk = self._process_audio_chunk(gen[0])
                
                # Handle multi-channel audio
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk[0]
                
                # Yield chunks efficiently
                pos = 0
                while pos + self.chunk_size <= len(audio_chunk):
                    yield audio_chunk[pos:pos + self.chunk_size]
                    pos += self.chunk_size
                
                # Handle remaining samples
                if pos < len(audio_chunk):
                    remaining = len(audio_chunk) - pos
                    self.padding_buffer[:remaining] = audio_chunk[pos:]
                    self.padding_buffer[remaining:] = 0
                    yield self.padding_buffer
        else:
            wavs = wavs_gen
            if len(wavs[0]) == 0:
                self.should_listen.set()
                return
                
            # Process entire audio at once
            audio_chunk = self._process_audio_chunk(wavs[0])
            
            # Yield fixed-size chunks
            for i in range(0, len(audio_chunk), self.chunk_size):
                chunk = audio_chunk[i:i + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    self.padding_buffer[:len(chunk)] = chunk
                    self.padding_buffer[len(chunk):] = 0
                    yield self.padding_buffer
                else:
                    yield chunk
                    
        self.should_listen.set()
