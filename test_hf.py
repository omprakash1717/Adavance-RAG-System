import os
import time
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

start = time.time()
print("Initializing HF API...")
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=hf_token,
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

try:
    print("Sending request...")
    res = embeddings.embed_query("Hello world")
    print(f"Success! Length: {len(res)} in {time.time()-start:.2f} seconds")
except Exception as e:
    print("Error:", e)
