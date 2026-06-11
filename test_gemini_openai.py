import os
import time
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

start = time.time()
print("Initializing OpenAI compat for Gemini...")
embeddings = OpenAIEmbeddings(
    api_key=key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="gemini-embedding-2", # or models/gemini-embedding-2
    check_embedding_ctx_length=False
)

try:
    print("Sending request...")
    res = embeddings.embed_query("Hello world")
    print(f"Success! Length: {len(res)} in {time.time()-start:.2f} seconds")
except Exception as e:
    print("Error:", e)
