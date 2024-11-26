from openai import OpenAI
import time
from typing import Optional

class GroqAnalyzer:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = "llama-3.1-8b-instant"
        self.max_retries = 5
        self.base_delay = 1  # Base delay in seconds

    def _make_api_call(self, prompt: str, retry_count: int = 0) -> Optional[str]:
        """
        Make API call with exponential backoff retry logic
        """
        try:
            if retry_count > 0:
                delay = min(self.base_delay * (2 ** (retry_count - 1)), 8)  # Cap at 8 seconds
                time.sleep(delay)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=50
            )
            
            if not response or not response.choices or not response.choices[0].message:
                raise ValueError("Invalid response format from API")
            
            content = response.choices[0].message.content
            if not content:
                raise ValueError("Empty content in API response")
                
            return content.strip().lower()

        except Exception as e:
            error_msg = str(e)
            if "rate_limit_exceeded" in error_msg and retry_count < self.max_retries:
                print(f"Rate limit exceeded, retrying in {delay} seconds...")
                return self._make_api_call(prompt, retry_count + 1)
            elif retry_count < self.max_retries:
                print(f"API error: {error_msg}, retrying...")
                return self._make_api_call(prompt, retry_count + 1)
            else:
                print(f"Max retries ({self.max_retries}) exceeded. Error: {error_msg}")
                return None

    def analyze(self, prompt: str) -> str:
        """
        Analyze sentiment using Groq API with retry logic
        """
        result = self._make_api_call(prompt)
        if result is None:
            return "unknown"
            
        # Normalize response
        if "positive" in result:
            return "positive"
        elif "negative" in result:
            return "negative"
        else:
            return "unknown"
