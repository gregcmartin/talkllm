import logging
import json
import requests
from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console
import torch

logger = logging.getLogger(__name__)
console = Console()

WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "fr": "french",
    "es": "spanish",
    "zh": "chinese",
    "ja": "japanese",
    "ko": "korean",
    "hi": "hindi",
    "de": "german",
    "pt": "portuguese",
    "pl": "polish",
    "it": "italian",
    "nl": "dutch",
}

class OllamaLanguageModelHandler(BaseHandler):
    """
    Handles the language model part using Ollama.
    """

    def setup(
        self,
        model_name="llama2",
        base_url="http://localhost:11434",
        gen_kwargs={},
        user_role="user",
        chat_size=1,
        init_chat_role=None,
        init_chat_prompt="You are a helpful AI assistant.",
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.gen_kwargs = gen_kwargs
        self.chat = Chat(chat_size)
        
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial prompt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
        
        self.user_role = user_role
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_input_text = "Repeat the word 'hello'."
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": dummy_input_text,
                    "stream": False
                }
            )
            response.raise_for_status()
            logger.info("Ollama warmup successful")
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama warmup failed: {str(e)}")
            logger.error("Please ensure Ollama is running and accessible")
            raise

    def process(self, prompt):
        logger.debug("inferring with Ollama language model...")
        language_code = None

        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                prompt = f"Please reply to my message in {WHISPER_LANGUAGE_TO_LLM_LANGUAGE[language_code]}. " + prompt

        self.chat.append({"role": self.user_role, "content": prompt})
        
        # Format chat history for Ollama
        messages = []
        for msg in self.chat.to_list():
            role = "system" if msg["role"] == "assistant" else "user"
            messages.append({"role": role, "content": msg["content"]})

        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": True,
                    **self.gen_kwargs
                },
                stream=True
            )
            response.raise_for_status()

            generated_text = ""
            current_chunk = ""
            
            for line in response.iter_lines():
                if line:
                    try:
                        chunk = json.loads(line)
                        if chunk.get("done", False):
                            break
                        if "message" in chunk and "content" in chunk["message"]:
                            text = chunk["message"]["content"]
                            generated_text += text
                            current_chunk += text
                            
                            if any(current_chunk.endswith(p) for p in [".", "?", "!", "\n"]):
                                yield (current_chunk, language_code)
                                current_chunk = ""
                    except json.JSONDecodeError as e:
                        logger.error(f"Error decoding JSON: {str(e)}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing chunk: {str(e)}")
                        continue

            if current_chunk:
                yield (current_chunk, language_code)

            if generated_text:
                self.chat.append({"role": "assistant", "content": generated_text})
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API error: {str(e)}")
            yield (f"I apologize, but I encountered an error: {str(e)}", language_code)
