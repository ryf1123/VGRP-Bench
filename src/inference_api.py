import base64
import logging
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Optional, List, Any
from PIL import Image
import argparse
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AIModelConfig:
    """Configuration class for AI models
    
    Attributes:
        model_name: Name of the model to use
        provider: Provider of the model (e.g., 'openai', 'gemini')
        api_url: Optional API endpoint URL for the model
        api_key: Optional API key for authentication
        max_tokens: Maximum number of tokens in the response
        temperature: Sampling temperature for generation
        retries: Number of retry attempts for failed requests
        retry_interval: Base interval between retries in seconds
    """
    model_name: str 
    provider: str
    api_url: Optional[str] = None
    api_key: Optional[str] = None
    max_tokens: int = 8192
    temperature: float = 0.0
    retries: int = 3
    retry_interval: int = 1

@dataclass
class Prompt:
    """Prompt class for model input
    
    Attributes:
        text: The text content of the prompt
        role: The role of the message sender (e.g., 'user', 'system')
        images: Optional list of images to include with the prompt
    """
    text: str
    role: str = "user"
    images: Optional[List[Image.Image]] = None

class ModelOutput:
    """Class for model generation output
    
    Attributes:
        text: The generated text response
        usage: Dictionary containing token usage statistics
        finish_reason: Reason why the model stopped generating
    """
    def __init__(self, text: str, usage: dict = None, finish_reason: str = None):
        self.text = text
        self.usage = usage or {}
        self.finish_reason = finish_reason

class BaseModelInterface:
    """Base interface for AI model implementations
    
    This class provides common functionality for all model interfaces
    and defines the interface that specific model implementations must follow.
    
    Attributes:
        config: Configuration for the model
        _client: The underlying client for the model API
    """
    
    def __init__(self, config: AIModelConfig):
        """Initialize the model interface
        
        Args:
            config: Configuration for the model
        """
        self.config = config
        self._client = None
        
    def _retry_operation(self, operation):
        """Execute operation with retry logic
        
        Args:
            operation: Function to execute with retries
            
        Returns:
            The result of the operation
            
        Raises:
            Exception: If all retry attempts fail
        """
        attempts = 0
        while attempts < self.config.retries:
            try:
                return operation()
            except Exception as e:
                attempts += 1
                delay = self.config.retry_interval * (2 ** attempts)
                logger.error(f"Operation failed: {e}. Attempt {attempts}/{self.config.retries}")
                if attempts < self.config.retries:
                    time.sleep(delay)
                else:
                    raise
                    
    def _process_image(self, image: Image.Image) -> str:
        """Convert image to base64 string
        
        Args:
            image: PIL Image to convert
            
        Returns:
            Base64 encoded string of the image
        """
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()

    def generate(self, prompt: Prompt) -> ModelOutput:
        """Generate response from model
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            ModelOutput containing the generated response
            
        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError

class GeminiInterface(BaseModelInterface):
    """Interface for Google's Gemini models
    
    Implements the BaseModelInterface for Google's Gemini models.
    """
    
    def __init__(self, config: AIModelConfig):
        """Initialize the Gemini interface
        
        Args:
            config: Configuration for the model
        """
        super().__init__(config)
        import google.generativeai as genai
        self._client = genai.GenerativeModel(config.model_name)
        
    def generate(self, prompt: Prompt) -> ModelOutput:
        """Generate response from Gemini model
        
        Args:
            prompt: The prompt to send to the model
            
        Returns:
            ModelOutput containing the generated response
        """
        def _generate():
            content = [prompt.text]
            if prompt.images:
                content.extend(prompt.images)
            response = self._client.generate_content(content)
            return ModelOutput(
                text=response.text,
                usage={"prompt_tokens": 0, "completion_tokens": 0},
                finish_reason="stop"
            )
        return self._retry_operation(_generate)

class DeepseekInterface(BaseModelInterface):
    """Interface for Deepseek models"""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        from openai import OpenAI
        self._client = OpenAI(base_url=config.api_url, api_key=config.api_key)
        
    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            completion = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": prompt.role, "content": prompt.text}],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            return ModelOutput(
                text=completion.choices[0].message.content,
                usage=completion.usage,
                finish_reason=completion.choices[0].finish_reason
            )
        return self._retry_operation(_generate)

class OpenAIInterface(BaseModelInterface):
    """Interface for OpenAI models"""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        from openai import OpenAI
        self._client = OpenAI(api_key=config.api_key)
        
    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            messages = [{"role": prompt.role, "content": []}]
            
            # Add text content
            if prompt.text:
                messages[0]["content"].append({"type": "text", "text": prompt.text})
            
            # Add image content if available
            if prompt.images:
                for image in prompt.images:
                    image_content = {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{self._process_image(image)}"}
                    }
                    messages[0]["content"].append(image_content)
            
            completion = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return ModelOutput(
                text=completion.choices[0].message.content,
                usage=completion.usage.model_dump(),
                finish_reason=completion.choices[0].finish_reason
            )
        return self._retry_operation(_generate)

class VLLMInterface(BaseModelInterface):
    """Interface for VLLM models (locally deployed with OpenAI-compatible API)"""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        # We'll use requests directly instead of the OpenAI client
        self._requests = requests
        self.api_url = config.api_url
        
    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            # Format messages with proper content structure for VLLM
            content = []
            
            # Add text content
            if prompt.text:
                content.append({"type": "text", "text": prompt.text})
            
            # Add image content if available
            if prompt.images:
                for image in prompt.images:
                    image_content = {
                        "type": "image_url", 
                        "image_url": {"url": f"data:image/png;base64,{self._process_image(image)}"}
                    }
                    content.append(image_content)
            
            # Prepare the request payload
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": prompt.role, "content": content}],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "encoding_format": "float"
            }
            
            # Make the direct API call
            response = self._requests.post(
                f"{self.api_url}/chat/completions",
                json=payload
            )
            response.raise_for_status()
            response_json = response.json()
            
            # Extract the response content
            return ModelOutput(
                text=response_json['choices'][0]['message']['content'],
                usage=response_json.get('usage', {}),
                finish_reason=response_json['choices'][0].get('finish_reason', None)
            )
        return self._retry_operation(_generate)

class ClaudeInterface(BaseModelInterface):
    """Interface for Anthropic Claude models"""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        from anthropic import Anthropic
        self._client = Anthropic(api_key=config.api_key)
        
    def generate(self, prompt: Prompt) -> ModelOutput:
        def _generate():
            messages = []
            
            # Add text content
            content = [{"type": "text", "text": prompt.text}]
            
            # Add image content if available
            if prompt.images:
                for image in prompt.images:
                    image_content = {
                        "type": "image",
                        "source": {
                            "type": "base64", 
                            "media_type": "image/png", 
                            "data": self._process_image(image)
                        }
                    }
                    content.append(image_content)
            
            # Claude expects messages in this format
            messages.append({"role": prompt.role, "content": content})
            
            completion = self._client.messages.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            
            return ModelOutput(
                text=completion.content[0].text,
                usage={"input_tokens": completion.usage.input_tokens, 
                       "output_tokens": completion.usage.output_tokens},
                finish_reason=completion.stop_reason
            )
        return self._retry_operation(_generate)

def create_model(config: AIModelConfig) -> BaseModelInterface:
    """Factory function to create model interface
    
    Args:
        config: Configuration for the model
        
    Returns:
        An instance of the appropriate model interface
        
    Raises:
        ValueError: If the provider is not supported
    """
    provider_lower = config.provider.lower()
    if "gemini" in provider_lower:
        return GeminiInterface(config)
    elif "deepseek" in provider_lower:
        return DeepseekInterface(config)
    elif "vllm" in provider_lower:
        return VLLMInterface(config)
    elif "claude" in provider_lower:
        return ClaudeInterface(config)
    elif "gpt" in provider_lower or "openai" in provider_lower:
        return OpenAIInterface(config)
    else:
        raise ValueError(f"Unsupported model provider: {config.provider}")

def generate_response(model_name: str, prompt_text: str, image_path: str = None, api_url: str = None) -> str:
    """High-level function to generate response from model
    
    Args:
        model_name: Name of the model to use
        prompt_text: Text prompt to send to the model
        image_path: Optional path to an image file to include with the prompt
        api_url: Optional API endpoint URL
        
    Returns:
        The generated text response
    """
    
    # Get provider from model_client_mapping instead of splitting model name
    client_type = get_client_type(model_name)
    if not client_type:
        # Fallback to old method if not found in mapping
        client_type = model_name.split("-")[0]
    
    # Prepare configuration
    config = AIModelConfig(
        model_name=model_name,
        provider=client_type,
        api_url=api_url
    )
    
    # Create model interface
    model = create_model(config)
    
    # Process image if provided
    images = None
    if image_path:
        try:
            images = [Image.open(image_path)]
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
    
    # Create prompt
    prompt = Prompt(text=prompt_text, images=images)
    
    # Generate response
    response = model.generate(prompt)
    return response.text

def load_model_client_mapping(file_path="configs/baseline_method_client_name.txt"):
    """
    Load the mapping between model names and their client types.
    
    Args:
        file_path: Path to the baseline_method_client_name.txt file
        
    Returns:
        dict: A dictionary mapping model names to their client types
    """
    model_to_client = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                parts = line.split(', ')
                if len(parts) == 2:
                    model_name, client_type = parts
                    model_to_client[model_name] = client_type
    except Exception as e:
        logger.error(f"Failed to load model client mapping: {e}")
                
    return model_to_client

# Load the mapping at module level
model_client_mapping = load_model_client_mapping()

# To get the client type for a specific model:
def get_client_type(model_name, mapping=None):

    if mapping is None:
        mapping = model_client_mapping
    
    return mapping.get(model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Model Interface")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--image", help="Optional image path")
    parser.add_argument("--api_url", help="Optional API endpoint", default="http://localhost:8000/v1")
    
    args = parser.parse_args()
    
    response = generate_response(
        args.model,
        args.prompt,
        args.image,
        args.api_url
    )
    
    print("\nModel Response:")
    print("=" * 50)
    print(response)
    print("=" * 50)
