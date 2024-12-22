import torchaudio
from VAD.vad_iterator import VADIterator
from baseHandler import BaseHandler
import numpy as np
import torch
from rich.console import Console

from utils.utils import int2float
from df.enhance import enhance, init_df
import logging

logger = logging.getLogger(__name__)

console = Console()


class VADHandler(BaseHandler):
    """
    Handles voice activity detection. When voice activity is detected, audio will be accumulated until the end of speech is detected and then passed
    to the following part.
    """

    def setup(
        self,
        should_listen,
        thresh=0.3,  # Lower threshold for better sensitivity
        sample_rate=16000,
        min_silence_ms=500,  # Shorter silence for more responsive detection
        min_speech_ms=100,  # Much shorter minimum speech duration
        max_speech_ms=float("inf"),
        speech_pad_ms=200,  # More padding for smoother transitions
        audio_enhancement=False,
    ):
        logger.debug(f"Initializing VAD with threshold={thresh}, sample_rate={sample_rate}")
        self.should_listen = should_listen
        self.sample_rate = sample_rate
        self.min_silence_ms = min_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_speech_ms = max_speech_ms
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        self.model = self.model.to(self.device)
        
        # Pre-allocate reusable buffer
        self.buffer = torch.zeros(1024, dtype=torch.float32, device=self.device)
        
        self.iterator = VADIterator(
            self.model,
            threshold=thresh,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.audio_enhancement = audio_enhancement
        if audio_enhancement:
            self.enhanced_model, self.df_state, _ = init_df()

    def process(self, audio_chunk):
        try:
            # Efficient conversion to tensor
            if isinstance(audio_chunk, bytes):
                audio_tensor = torch.frombuffer(audio_chunk, dtype=torch.int16).float()
            else:
                audio_tensor = torch.from_numpy(audio_chunk.flatten()).float()
            
            # Convert to float32 in [-1, 1] range in one operation
            audio_tensor = audio_tensor / 32768.0
            
            # Resize buffer if needed
            if len(audio_tensor) > len(self.buffer):
                self.buffer = torch.zeros(len(audio_tensor), dtype=torch.float32, device=self.device)
            
            # Copy to pre-allocated buffer and move to device
            self.buffer[:len(audio_tensor)] = audio_tensor
            audio_tensor = self.buffer[:len(audio_tensor)]
            
            # Process through VAD
            vad_output = self.iterator(audio_tensor.view(1, -1))
            
            # If we have output, check its duration
            if vad_output is not None and len(vad_output) != 0:
                logger.debug("VAD: end of speech detected")
                array = torch.cat(vad_output).cpu()
                # Keep as tensor until absolutely necessary to convert to numpy
                duration_ms = array.shape[-1] / self.sample_rate * 1000
                logger.debug(f"Speech duration: {duration_ms:.2f}ms")
                
                # Only process if duration meets requirements
                if duration_ms < self.min_speech_ms:
                    logger.debug(f"Speech too short ({duration_ms:.2f}ms < {self.min_speech_ms}ms), skipping")
                    return
                elif duration_ms > self.max_speech_ms:
                    logger.debug(f"Speech too long ({duration_ms:.2f}ms > {self.max_speech_ms}ms), skipping")
                    return
                
                # Process valid speech segment
                self.should_listen.clear()
                logger.debug("Stop listening")
                
                # Apply audio enhancement if enabled
                if self.audio_enhancement:
                    array = self._enhance_audio(array)
                
                # Convert to numpy only at the final step
                yield array.numpy()
                
        except Exception as e:
            logger.warning(f"Error processing audio chunk: {e}")
            return
            
    def _enhance_audio(self, audio_array):
        """Apply audio enhancement if enabled."""
        if self.sample_rate != self.df_state.sr():
            audio_float32 = torchaudio.functional.resample(
                torch.from_numpy(audio_array),
                orig_freq=self.sample_rate,
                new_freq=self.df_state.sr(),
            )
            enhanced = enhance(
                self.enhanced_model,
                self.df_state,
                audio_float32.unsqueeze(0),
            )
            enhanced = torchaudio.functional.resample(
                enhanced,
                orig_freq=self.df_state.sr(),
                new_freq=self.sample_rate,
            )
        else:
            enhanced = enhance(
                self.enhanced_model,
                self.df_state,
                torch.from_numpy(audio_array),
            )
        return enhanced.numpy().squeeze()

    @property
    def min_time_to_debug(self):
        return 0.00001
