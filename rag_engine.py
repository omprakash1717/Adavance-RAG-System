import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from endee import Endee, Precision

load_dotenv()

HF_TOKEN       = os.getenv("HUGGINGFACE_API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ENDEE_TOKEN    = os.getenv("ENDEE_API_KEY")
ENDEE_BASE_URL = os.getenv("ENDEE_BASE_URL")
COLLECTION     = os.getenv("ENDEE_COLLECTION", "RAG_system")

# Global cache for models
_embeddings_model = None
_chat_model = None

# Monkey-patch VectorItem for Python 3.14 bug
from endee.schema import VectorItem
if not hasattr(VectorItem, "get"):
    VectorItem.get = lambda self, key, default=None: getattr(self, key, default)

client = Endee(ENDEE_TOKEN)
client.set_base_url(ENDEE_BASE_URL)

def get_embeddings_model():
    global _embeddings_model
    if _embeddings_model is None:
        if GEMINI_API_KEY:
            print("Loading Google Generative AI Embeddings (API-based)...")
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _embeddings_model = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GEMINI_API_KEY
            )
        else:
            print("Loading HuggingFace Embeddings (Local)...")
            from langchain_huggingface import HuggingFaceEmbeddings
            _embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _embeddings_model

def get_chat_model():
    global _chat_model
    if _chat_model is None:
        print("Loading Gemini Chat Model (first time)...")
        _chat_model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY,
            temperature=0.3
        )
    return _chat_model

def _ensure_index(dimension=384):
    """Delete any stale index and (re)create one with the right dimension."""
    from endee.exceptions import ConflictException

    # Try to create the index; if it already exists, verify it is usable
    try:
        client.create_index(
            name=COLLECTION,
            dimension=dimension,
            space_type="cosine",
            precision=Precision.INT8,
        )
        print(f"Index '{COLLECTION}' created (dim={dimension}).")
        return client.get_index(name=COLLECTION)
    except ConflictException:
        # Index already exists — try to get it
        pass
    except Exception as e:
        err = str(e)
        if "Missing or incompatible" in err or "metadata" in err.lower():
            print(f"Incompatible index detected, deleting '{COLLECTION}'...")
            try:
                client.delete_index(name=COLLECTION)
            except Exception:
                pass
            client.create_index(
                name=COLLECTION,
                dimension=dimension,
                space_type="cosine",
                precision=Precision.INT8,
            )
            print(f"Index '{COLLECTION}' re-created (dim={dimension}).")
            return client.get_index(name=COLLECTION)
        raise

    # Index already existed — try to open it
    try:
        index = client.get_index(name=COLLECTION)
        return index
    except Exception as e:
        err = str(e)
        if "Missing or incompatible" in err or "metadata" in err.lower():
            print(f"Stale index detected, deleting '{COLLECTION}'...")
            try:
                client.delete_index(name=COLLECTION)
            except Exception:
                pass
            client.create_index(
                name=COLLECTION,
                dimension=dimension,
                space_type="cosine",
                precision=Precision.INT8,
            )
            print(f"Index '{COLLECTION}' re-created (dim={dimension}).")
            return client.get_index(name=COLLECTION)
        raise


def process_pdf(pdf_path: str):
    """Loads a PDF, splits into chunks, and upserts dense vectors to Endee."""
    print(f"Loading '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings_model = get_embeddings_model()

    # Build vectors first so we can detect dimension
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = embeddings_model.embed_query(chunk.page_content)
        vectors.append({
            "id": f"{os.path.basename(pdf_path)}_{i}",
            "vector": embedding,
            "meta": {
                "text": chunk.page_content,
                "page": str(chunk.metadata.get("page", "N/A"))
            }
        })

    dim = len(vectors[0]["vector"])  # should be 384
    print(f"Embedding dimension: {dim}")

    # Get (or recreate) the index with the correct dimension
    index = _ensure_index(dimension=dim)

    # Upsert — if it still fails due to dimension mismatch, delete & retry once
    try:
        index.upsert(vectors)
    except BaseException as e:
        err_msg = str(e)
        print(f"[DEBUG] Upsert error type={type(e).__name__}, msg={err_msg}")
        if "Expected shape" in err_msg or "shape" in err_msg or "dimension" in err_msg.lower() or "3072" in err_msg:
            print(f"Dimension mismatch on upsert, recreating index...")
            try:
                client.delete_index(name=COLLECTION)
                print(f"Old index '{COLLECTION}' deleted.")
            except Exception as del_e:
                print(f"[DEBUG] Delete failed: {del_e}")
            client.create_index(
                name=COLLECTION,
                dimension=dim,
                space_type="cosine",
                precision=Precision.INT8,
            )
            print(f"New index '{COLLECTION}' created with dim={dim}.")
            index = client.get_index(name=COLLECTION)
            index.upsert(vectors)
        else:
            raise

    print(f"{len(chunks)} chunks stored in Endee Cloud!")
    return len(chunks)


def query_pdf(user_query: str):
    """Queries the Endee DB and passes context to Gemini for an answer."""
    try:
        index = client.get_index(name=COLLECTION)
    except Exception as e:
        raise Exception(
            "No PDF has been indexed yet. Please upload a PDF first."
        )

    embeddings_model = get_embeddings_model()
    query_vector = embeddings_model.embed_query(user_query)
    results = index.query(vector=query_vector, top_k=3)

    context = "\n\n".join(
        f"Page Content: {r['meta'].get('text', '')}\nPage Number: {r['meta'].get('page', 'N/A')}"
        for r in results
    )

    prompt = f"""
You are a helpful and detailed AI assistant answering questions about an uploaded PDF document.
Use the context below to answer accurately. Always state what page numbers your answer is referencing if applicable.

Context from PDF:
{context}

Question:
{user_query}
"""
    chat_model = get_chat_model()
    response = chat_model.invoke(prompt)
    return response.content
