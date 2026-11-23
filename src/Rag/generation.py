import getpass
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv() 

groq_client = Groq() 
GROQ_MODEL  = "openai/gpt-oss-20b"

SYSTEM_PROMPT = """You are a helpful HR assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information, say so — do not make things up.
Always cite the section name when referencing specific information."""

def generation(user_message: str)-> str:
     res = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,   # Low = factual;  High = creative
    )
     content = res.choices[0].message.content
     return content
