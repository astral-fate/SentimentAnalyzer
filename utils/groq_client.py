from openai import OpenAI

class GroqAnalyzer:
    def __init__(self, api_key):
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        self.model = "llama-3.1-8b-instant"

    def analyze(self, prompt):
        """
        Analyze sentiment using Groq API
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=50
            )
            
            result = response.choices[0].message.content.strip().lower()
            
            # Normalize response
            if "positive" in result:
                return "positive"
            elif "negative" in result:
                return "negative"
            else:
                return "unknown"
                
        except Exception as e:
            raise Exception(f"Error calling Groq API: {str(e)}")
