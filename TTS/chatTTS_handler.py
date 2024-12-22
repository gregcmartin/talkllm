import ChatTTS
import logging
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


class ChatTTSHandler(BaseHandler):
    def setup(
        self,
        should_listen,
        device="cuda",
        gen_kwargs={},  # Unused
        stream=True,
        chunk_size=512,
    ):
        self.should_listen = should_listen
        self.device = device
        self.model = ChatTTS.Chat()
        self.model.load(compile=False)  # Doesn't work for me with True
        self.chunk_size = chunk_size
        self.stream = stream
        rnd_spk_emb = self.model.sample_random_speaker()
        self.params_infer_code = ChatTTS.Chat.InferCodeParams(
            spk_emb=rnd_spk_emb,
        )
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        _ = self.model.infer("text")

    def clean_text(self, text):
        """Clean text by removing problematic characters while preserving meaning."""
        # Replace common punctuation with periods which work well with TTS
        text = text.replace('?', '.')
        text = text.replace('!', '.')
        text = text.replace("'", '')
        # Remove any remaining special characters but preserve sentence structure
        text = text.replace(',', '')  # Remove commas
        text = text.replace(':', '')  # Remove colons
        text = text.replace(';', '.')  # Convert semicolons to periods
        text = text.replace('-', ' ')  # Convert hyphens to spaces
        text = text.replace('/', ' or ')  # Convert slashes to 'or'
        text = text.replace('(', '')  # Remove parentheses
        text = text.replace(')', '')
        text = text.replace('&', ' and ')  # Convert ampersands to 'and'
        # Only allow letters, numbers, spaces, and periods
        text = ''.join(c for c in text if c.isalnum() or c.isspace() or c == '.')
        # Ensure proper spacing
        text = ' '.join(text.split())
        return text.strip()

    def process(self, llm_input):
        # Extract text from LLM input (which could be a tuple of (text, language_code))
        llm_sentence = llm_input[0] if isinstance(llm_input, tuple) else llm_input
        
        # Clean the text before displaying and processing
        llm_sentence = self.clean_text(llm_sentence)
        
        console.print(f"[green]ASSISTANT: {llm_sentence}")
        if self.device == "mps":
            import time

            start = time.time()
            torch.mps.synchronize()  # Waits for all kernels in all streams on the MPS device to complete.
            torch.mps.empty_cache()  # Frees all memory allocated by the MPS device.
            _ = (
                time.time() - start
            )  # Removing this line makes it fail more often. I'm looking into it.

        wavs_gen = self.model.infer(
            llm_sentence, params_infer_code=self.params_infer_code, stream=self.stream
        )

        if self.stream:
            wavs = [np.array([])]
            for gen in wavs_gen:
                if gen[0] is None or len(gen[0]) == 0:
                    self.should_listen.set()
                    return
                audio_chunk = librosa.resample(gen[0], orig_sr=24000, target_sr=16000)
                audio_chunk = (audio_chunk * 32768).astype(np.int16)
                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk[0]  # Take first channel if multi-channel
                while len(audio_chunk) > self.chunk_size:
                    yield audio_chunk[: self.chunk_size]  # Return the first chunk_size samples of the audio data
                    audio_chunk = audio_chunk[self.chunk_size :]  # Remove the samples that have already been returned
                if len(audio_chunk) > 0:
                    yield np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
        else:
            wavs = wavs_gen
            if len(wavs[0]) == 0:
                self.should_listen.set()
                return
            audio_chunk = librosa.resample(wavs[0], orig_sr=24000, target_sr=16000)
            audio_chunk = (audio_chunk * 32768).astype(np.int16)
            for i in range(0, len(audio_chunk), self.chunk_size):
                yield np.pad(
                    audio_chunk[i : i + self.chunk_size],
                    (0, self.chunk_size - len(audio_chunk[i : i + self.chunk_size])),
                )
        self.should_listen.set()
