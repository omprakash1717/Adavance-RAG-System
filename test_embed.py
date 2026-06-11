import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

embeddings = OpenAIEmbeddings(
    api_key=key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    model="embedding-001",
    check_embedding_ctx_length=False
)

try:
    print("Sending request...")
    res = embeddings.embed_query("Hello world")
    print("Success! Length:", len(res))
except Exception as e:
    print("Error:", e)
