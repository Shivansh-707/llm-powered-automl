"""Quick test to verify Groq API connectivity."""
from groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {"role": "user", "content": "Say 'Setup complete!' if you can read this."}
    ],
    max_tokens=50,
)

print(response.choices[0].message.content)
