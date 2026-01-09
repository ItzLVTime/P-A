from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_BASE = os.getenv("LITELLM_API_BASE")
API_KEY = os.getenv("LITELLM_API_KEY")

if not API_BASE:
    raise RuntimeError("LITELLM_API_BASE not set")
if not API_KEY:
    raise RuntimeError("LITELLM_API_KEY not set")

client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE
)

def ask_llm(prompt, model="ollama/qwen2.5:7b-instruct-q4_k_m"):
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return resp.choices[0].message.content
