import logging
import ollama
from typing import Generator, Optional, Tuple, Union, List, Dict
from LLM.chat import Chat
from baseHandler import BaseHandler
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

# Pre-compile language mappings for faster lookup
WHISPER_LANGUAGE_TO_LLM_LANGUAGE = {
    "en": "english",
    "es": "spanish",
    "de": "german",
}

# Pre-compile sentence ending patterns
SENTENCE_ENDINGS = frozenset([".", "?", "!", "\n"])

# Concise system prompt for natural conversation
SYSTEM_PROMPT = """You are a friendly conversational AI. Keep responses natural, concise, and to the point. 
Avoid repetition and unnecessary questions. Respond in a casual, human-like manner.
Focus on providing direct, helpful information without being overly formal."""

class OllamaLanguageModelHandler(BaseHandler):
    """
    Handles the language model part using Ollama Python client.
    """

    def setup(
        self,
        model_name="llama2",
        base_url="http://localhost:11434",
        gen_kwargs={
            "temperature": 0.7,  # Balanced between creative and focused
            "top_k": 40,        # More focused response selection
            "top_p": 0.9,       # Slightly reduced randomness
            "repeat_penalty": 1.1,  # Mild penalty for repetition
        },
        user_role="user",
        chat_size=3,  # Keep conversation context small
        init_chat_role="system",
        init_chat_prompt=SYSTEM_PROMPT,
    ):
        self.model_name = model_name
        self.client = ollama.Client(host=base_url)
        self.gen_kwargs = gen_kwargs
        self.chat = Chat(chat_size)
        self.user_role = user_role
        
        # Pre-allocate messages list with system prompt
        self.messages: List[Dict[str, str]] = []
        if init_chat_role:
            if not init_chat_prompt:
                raise ValueError(
                    "An initial prompt needs to be specified when setting init_chat_role."
                )
            self.chat.init_chat({"role": init_chat_role, "content": init_chat_prompt})
            self.messages.append({"role": "system", "content": init_chat_prompt})
        
        # Cache for language prompts
        self.language_prompts = {
            lang: f"Respond in {llm_lang}. "  # Simplified language prompt
            for lang, llm_lang in WHISPER_LANGUAGE_TO_LLM_LANGUAGE.items()
        }
        
        self.warmup()

    def warmup(self):
        logger.info(f"Warming up {self.__class__.__name__}")
        dummy_input_text = "Hi"  # Simple warmup prompt
        try:
            # Test both generate and chat endpoints
            self.client.generate(model=self.model_name, prompt=dummy_input_text)
            self.client.chat(model=self.model_name, messages=[{"role": "user", "content": dummy_input_text}])
            logger.info("Ollama warmup successful")
        except Exception as e:
            logger.error(f"Ollama warmup failed: {str(e)}")
            logger.error("Please ensure Ollama is running and accessible")
            raise

    def _format_messages(self, new_prompt: str) -> List[Dict[str, str]]:
        """Efficiently format chat history for Ollama."""
        # Start with existing messages
        messages = self.messages.copy()
        
        # Add new user message
        messages.append({"role": "user", "content": new_prompt})
        
        return messages

    def process(self, prompt: Union[str, Tuple[str, str]]) -> Generator[Tuple[str, Optional[str]], None, None]:
        logger.debug("inferring with Ollama language model...")
        language_code = None

        # Handle language-specific prompts
        if isinstance(prompt, tuple):
            prompt, language_code = prompt
            if language_code[-5:] == "-auto":
                language_code = language_code[:-5]
                # Use cached language prompt
                if language_code in self.language_prompts:
                    prompt = self.language_prompts[language_code] + prompt

        # Update chat history
        self.chat.append({"role": self.user_role, "content": prompt})
        
        # Get formatted messages for Ollama
        messages = self._format_messages(prompt)

        try:
            generated_text = []  # Use list for efficient string building
            current_chunk = []
            
            # Stream response with optimized chunk handling
            for chunk in self.client.chat(
                model=self.model_name,
                messages=messages,
                stream=True,
                **self.gen_kwargs
            ):
                if "message" in chunk and "content" in chunk["message"]:
                    text = chunk["message"]["content"]
                    generated_text.append(text)
                    current_chunk.append(text)
                    
                    # Check for sentence endings
                    if any(text.endswith(end) for end in SENTENCE_ENDINGS):
                        chunk_text = "".join(current_chunk)
                        yield (chunk_text, language_code)
                        current_chunk = []

            # Handle any remaining text
            if current_chunk:
                chunk_text = "".join(current_chunk)
                yield (chunk_text, language_code)

            # Update chat history with complete response
            complete_response = "".join(generated_text)
            if complete_response:
                # Trim chat history to maintain context size
                if len(self.messages) >= self.chat.size * 2:
                    self.messages = self.messages[-self.chat.size:]
                
                self.chat.append({"role": "assistant", "content": complete_response})
                self.messages.append({"role": "assistant", "content": complete_response})
            
        except Exception as e:
            logger.error(f"Ollama API error: {str(e)}")
            yield (f"I apologize, but I encountered an error: {str(e)}", language_code)
