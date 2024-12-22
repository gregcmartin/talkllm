import torch


class VADIterator:
    def __init__(
        self,
        model,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """
        Mainly taken from https://github.com/snakers4/silero-vad
        Class for stream imitation

        Parameters
        ----------
        model: preloaded .jit/.onnx silero VAD model

        threshold: float (default - 0.5)
            Speech threshold. Silero VAD outputs speech probabilities for each audio chunk, probabilities ABOVE this value are considered as SPEECH.
            It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.

        sampling_rate: int (default - 16000)
            Currently silero VAD models support 8000 and 16000 sample rates

        min_silence_duration_ms: int (default - 100 milliseconds)
            In the end of each speech chunk wait for min_silence_duration_ms before separating it

        speech_pad_ms: int (default - 30 milliseconds)
            Final speech chunks are padded by speech_pad_ms each side
        """

        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.is_speaking = False

        if sampling_rate not in [8000, 16000]:
            raise ValueError(
                "VADIterator does not support sampling rates other than [8000, 16000]"
            )

        # Pre-allocate buffer as tensor to avoid list operations
        self.max_buffer_size = 30 * sampling_rate  # 30 seconds max
        self.buffer = torch.zeros(self.max_buffer_size, dtype=torch.float32, device=next(model.parameters()).device)
        self.buffer_idx = 0

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self):
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0
        self.buffer_idx = 0

    @torch.no_grad()
    def __call__(self, x):
        """
        x: torch.Tensor
            audio chunk (see examples in repo)

        return_seconds: bool (default - False)
            whether return timestamps in seconds (default - samples)
        """
        # Input should already be tensor from VADHandler
        if not torch.is_tensor(x):
            raise TypeError("Input must be a tensor")

        # Ensure input is on the correct device
        if x.device != self.buffer.device:
            x = x.to(self.buffer.device)

        window_size_samples = x.shape[-1]  # More efficient than len()
        self.current_sample += window_size_samples

        # Resize input if needed
        if window_size_samples > 512:
            # Split into 512-sample chunks
            chunks = x.view(-1, 512)
            speech_probs = []
            for chunk in chunks:
                prob = self.model(chunk.unsqueeze(0), self.sampling_rate).item()
                speech_probs.append(prob)
            speech_prob = sum(speech_probs) / len(speech_probs)
        else:
            speech_prob = self.model(x, self.sampling_rate).item()

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            return None

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            else:
                # End of speech detected
                self.temp_end = 0
                self.triggered = False
                # Return only the used portion of buffer
                spoken_utterance = self.buffer[:self.buffer_idx].unsqueeze(0)
                self.buffer_idx = 0  # Reset buffer index
                return [spoken_utterance]  # Keep list format for compatibility

        if self.triggered:
            # Add to pre-allocated buffer
            if self.buffer_idx + window_size_samples <= self.max_buffer_size:
                self.buffer[self.buffer_idx:self.buffer_idx + window_size_samples] = x.squeeze()
                self.buffer_idx += window_size_samples

        return None
