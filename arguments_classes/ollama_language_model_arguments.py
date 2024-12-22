from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class OllamaLanguageModelHandlerArguments:
    """
    Arguments for the Ollama Language Model Handler
    """
    ollama_model_name: Optional[str] = field(
        default="qwen2.5:7b",
        metadata={"help": "Name of the Ollama model to use"}
    )
    ollama_base_url: Optional[str] = field(
        default="http://localhost:11434",
        metadata={"help": "Base URL for the Ollama API"}
    )
    ollama_user_role: Optional[str] = field(
        default="user",
        metadata={"help": "Role to use for user messages"}
    )
    ollama_chat_size: Optional[int] = field(
        default=1,
        metadata={"help": "Number of previous exchanges to keep in chat history"}
    )
    ollama_init_chat_role: Optional[str] = field(
        default="system",
        metadata={"help": "Role for the initial chat message"}
    )
    ollama_init_chat_prompt: Optional[str] = field(
        default="You are a helpful AI assistant.",
        metadata={"help": "Initial system prompt for the chat"}
    )
    ollama_gen_kwargs: Dict = field(
        default_factory=lambda: {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 1024
        },
        metadata={"help": "Generation parameters for Ollama API"}
    )
