import os
import time
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()
hf_token = os.getenv("HUGGINGFACE_API_TOKEN")

start = time.time()
print("Initializing HF Endpoint...")
embeddings = HuggingFaceEndpointEmbeddings(
    huggingfacehub_api_token=hf_token,
    model="sentence-transformers/all-MiniLM-L6-v2"
)

try:
    print("Sending request...")
    res = embeddings.embed_query("Hello world")
    print(f"Success! Length: {len(res)} in {time.time()-start:.2f} seconds")
except Exception as e:
    print("Error:", e)
