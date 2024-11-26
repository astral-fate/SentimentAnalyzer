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
            
            # Add null check and proper response handling
            if not response or not response.choices or not response.choices[0].message:
                return "unknown"
            
            content = response.choices[0].message.content
            if not content:
                return "unknown"
                
            result = content.strip().lower()
            
            # Normalize response
            if "positive" in result:
                return "positive"
            elif "negative" in result:
                return "negative"
            else:
                return "unknown"
                
        except Exception as e:
            print(f"Error in Groq API call: {str(e)}")
            return "unknown"  # Graceful error handling
