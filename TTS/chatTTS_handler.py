import ChatTTS
import logging
import os
import pickle
import re
import time
import hashlib
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

# Voice types with their characteristics
VOICE_TYPES = {
    "neutral": {"description": "Balanced, natural speaking voice", "seed": 42},
    "warm": {"description": "Friendly, approachable voice", "seed": 137},
    "professional": {"description": "Clear, authoritative voice", "seed": 271},
    "casual": {"description": "Relaxed, conversational voice", "seed": 314},
    "energetic": {"description": "Dynamic, enthusiastic voice", "seed": 628},
}

class ChatTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cpu",  # ChatTTS uses CPU for stability with certain operations
        gen_kwargs={},  # Unused
        stream=True,
        chunk_size=1024,  # Larger chunks for smoother audio
        voice_type="neutral",  # Voice type selection
        speaker_id=None,  # New parameter for speaker configuration
    ):
        self.should_listen = should_listen
        self.device = device
        self.model = ChatTTS.Chat()
        self.model.load(compile=False)  # Doesn't work for me with True
        self.chunk_size = chunk_size
        self.stream = stream
        self.voice_type = voice_type
        
        # Pre-allocate reusable buffers with larger sizes
        self.audio_buffer = np.zeros(chunk_size * 2, dtype=np.int16)  # Double buffer
        self.padding_buffer = np.zeros(chunk_size * 2, dtype=np.int16)
        
        # Add cooldown to prevent interruptions
        self.last_speech_time = 0
        self.speech_cooldown = 0.5  # seconds
        
        # Cache for cleaned text
        self.text_cache = {}
        self.max_cache_size = 1000
        
        # Voice embeddings cache directory
        self.voice_cache_dir = Path("voice_embeddings")
        self.voice_cache_dir.mkdir(exist_ok=True)
        
        # Load or create speaker embedding
        self._setup_speaker_embedding()
            
        # Initialize inference parameters with the speaker embedding
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=self.speaker_embedding,
        )
        
        logger.info(f"ChatTTS setup complete with {voice_type} voice")

    def _setup_speaker_embedding(self):
        """Setup speaker embedding with efficient file handling."""
        self.speaker_file = self.voice_cache_dir / f"speaker_embedding_{self.voice_type}.pkl"
        try:
            if self.speaker_file.exists():
                with open(self.speaker_file, 'rb') as f:
                    self.speaker_embedding = pickle.load(f)
                logger.info(f"Loaded existing {self.voice_type} voice from {self.speaker_file}")
            else:
                self._create_new_speaker()
        except Exception as e:
            logger.warning(f"Failed to load speaker embedding: {e}")
            self._create_new_speaker()

    def _create_new_speaker(self):
        """Create and save new speaker embedding with deterministic selection."""
        # Set fixed seed for voice type
        seed = VOICE_TYPES[self.voice_type]["seed"]
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        logger.info(f"Generating new {self.voice_type} voice with seed {seed}")
        
        # Generate a fixed number of candidates
        candidates = []
        for i in range(20):  # Generate more candidates for better selection
            embedding = self.model.sample_random_speaker()
            # Convert embedding to bytes for hashing
            if isinstance(embedding, str):
                embedding_bytes = embedding.encode('utf-8')
            elif isinstance(embedding, np.ndarray):
                embedding_bytes = embedding.tobytes()
            elif isinstance(embedding, torch.Tensor):
                embedding_bytes = embedding.cpu().numpy().tobytes()
            else:
                embedding_bytes = str(embedding).encode('utf-8')
            
            # Create a deterministic hash
            embedding_hash = hashlib.sha256(embedding_bytes).hexdigest()
            candidates.append((embedding, embedding_hash))
            logger.debug(f"Generated candidate {i+1}/20 with hash {embedding_hash[:8]}")
        
        # Sort candidates by hash to ensure deterministic selection
        candidates.sort(key=lambda x: x[1])
        
        # Always select the same embedding for this voice type
        self.speaker_embedding = candidates[0][0]
        logger.info(f"Selected voice embedding with hash {candidates[0][1][:8]}")
        
        # Reset random seeds
        torch.manual_seed(torch.initial_seed())
        np.random.seed(None)
        
        self._save_speaker_embedding()
        logger.info(f"Generated deterministic {self.voice_type} voice")

    def _save_speaker_embedding(self):
        """Save speaker embedding with atomic write."""
        temp_file = self.speaker_file.with_suffix('.tmp')
        try:
            # Ensure directory exists
            self.voice_cache_dir.mkdir(exist_ok=True)
            
            with open(temp_file, 'wb') as f:
                pickle.dump(self.speaker_embedding, f)
            temp_file.replace(self.speaker_file)  # Atomic replace
            logger.info(f"Saved voice embedding to {self.speaker_file}")
        except Exception as e:
            logger.warning(f"Failed to save speaker embedding: {e}")
            if temp_file.exists():
                temp_file.unlink()  # Clean up temp file
        
    def set_speaker(self, voice_type=None):
        """Change the speaker voice."""
        if voice_type is not None and voice_type in VOICE_TYPES:
            self.voice_type = voice_type
            self._setup_speaker_embedding()  # Load existing or create new voice
            self.params_infer_code = ChatTTS.Chat.InferCodeParams(
                spk_emb=self.speaker_embedding,
            )
            self.warmup()
        else:
            available_voices = "\n".join(f"- {v}: {VOICE_TYPES[v]['description']}" 
                                       for v in VOICE_TYPES)
            logger.warning(f"Invalid voice type. Available voices:\n{available_voices}")

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
        
        # Track speech timing
        self.last_speech_time = time.time()
        console.print(f"[green]ASSISTANT: {llm_sentence}")
        logger.info(f"Generating speech for: {llm_sentence}")

        # Get audio generator
        wavs_gen = self.model.infer(
            llm_sentence, params_infer_code=self.params_infer_code, stream=self.stream
        )
        logger.debug("Got audio generator")

        if self.stream:
            for gen in wavs_gen:
                if gen[0] is None or len(gen[0]) == 0:
                    logger.warning("Empty audio chunk received")
                    # Only allow interruption after cooldown
                    if time.time() - self.last_speech_time > self.speech_cooldown:
                        self.should_listen.set()
                    return
                
                # Process audio chunk efficiently
                audio_chunk = self._process_audio_chunk(gen[0])
                logger.debug(f"Processed audio chunk of size {len(audio_chunk)}")
                
                # Handle multi-channel audio
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk[0]
                
                # Yield chunks efficiently
                pos = 0
                while pos + self.chunk_size <= len(audio_chunk):
                    yield audio_chunk[pos:pos + self.chunk_size]
                    pos += self.chunk_size
                    logger.debug(f"Yielded audio chunk at position {pos}")
                
                # Handle remaining samples
                if pos < len(audio_chunk):
                    remaining = len(audio_chunk) - pos
                    self.padding_buffer[:remaining] = audio_chunk[pos:]
                    self.padding_buffer[remaining:] = 0
                    yield self.padding_buffer
                    logger.debug(f"Yielded final chunk with {remaining} samples")
        else:
            wavs = wavs_gen
            if len(wavs[0]) == 0:
                logger.warning("Empty audio received in non-stream mode")
                # Only allow interruption after cooldown
                if time.time() - self.last_speech_time > self.speech_cooldown:
                    self.should_listen.set()
                return
                
            # Process entire audio at once
            audio_chunk = self._process_audio_chunk(wavs[0])
            logger.debug(f"Processed complete audio of size {len(audio_chunk)}")
            
            # Yield fixed-size chunks
            for i in range(0, len(audio_chunk), self.chunk_size):
                chunk = audio_chunk[i:i + self.chunk_size]
                if len(chunk) < self.chunk_size:
                    self.padding_buffer[:len(chunk)] = chunk
                    self.padding_buffer[len(chunk):] = 0
                    yield self.padding_buffer
                else:
                    yield chunk
                logger.debug(f"Yielded chunk {i//self.chunk_size + 1}")
        
        # Update speech timing and respect cooldown
        self.last_speech_time = time.time()
        if time.time() - self.last_speech_time > self.speech_cooldown:
            self.should_listen.set()
        logger.info("Finished generating speech")

    @staticmethod
    def list_voice_types():
        """List available voice types and their descriptions."""
        return "\n".join(f"{v}: {VOICE_TYPES[v]['description']}" for v in VOICE_TYPES)
