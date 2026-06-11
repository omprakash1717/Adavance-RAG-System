import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

HF_TOKEN       = os.getenv("HUGGINGFACE_API_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
COLLECTION     = "RAG_system_v3"
CHROMA_PERSIST_DIR = "./chroma_db"

# Global cache
_embeddings_model = None
_vectorstore = None


def get_embeddings_model():
    global _embeddings_model
    if _embeddings_model is None:
        if GEMINI_API_KEY:
            print("Loading Google Generative AI Embeddings (OpenAI Compat)...")
            from langchain_openai import OpenAIEmbeddings
            _embeddings_model = OpenAIEmbeddings(
                api_key=GEMINI_API_KEY,
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                model="gemini-embedding-2",
                check_embedding_ctx_length=False
            )
        else:
            raise Exception("GEMINI_API_KEY is missing! Please set it in your Render Environment Variables.")
    return _embeddings_model


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        from langchain_chroma import Chroma
        if not os.path.exists(CHROMA_PERSIST_DIR):
            raise Exception("No PDF has been indexed yet. Please upload a PDF first.")
        _vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=get_embeddings_model(),
            collection_name=COLLECTION
        )
    return _vectorstore


def _call_llm(prompt: str) -> str:
    """Call LLM with automatic fallback: DeepSeek -> Gemini OpenAI-compat."""
    providers = []

    if GEMINI_API_KEY:
        providers.append(("Gemini", GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.0-flash-lite"))
        providers.append(("Gemini", GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.0-flash"))
        providers.append(("Gemini", GEMINI_API_KEY, "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-2.5-flash"))
    if DEEPSEEK_API_KEY:
        providers.append(("DeepSeek", DEEPSEEK_API_KEY, "https://api.deepseek.com", "deepseek-chat"))

    last_error = None
    for name, key, base_url, model in providers:
        try:
            client = OpenAI(api_key=key, base_url=base_url)
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=2000,
                timeout=30,
            )
            return response.choices[0].message.content
        except Exception as e:
            last_error = e
            print(f"  {name}/{model} failed: {str(e)[:80]}... trying next")
            continue

    raise Exception(
        f"All AI providers are currently unavailable. "
        f"Please wait a minute and try again. Last error: {str(last_error)[:150]}"
    )


def process_pdf(pdf_path: str):
    """Loads a PDF, splits into chunks, and saves to ChromaDB."""
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_chroma import Chroma
    
    print(f"Loading '{pdf_path}'...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    if not chunks:
        raise Exception("Could not extract any text from the uploaded PDF. The file might be empty, or it's a scanned/image-based PDF with no selectable text.")

    embeddings_model = get_embeddings_model()

    global _vectorstore
    print("Embedding and storing in ChromaDB...")
    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings_model,
        persist_directory=CHROMA_PERSIST_DIR,
        collection_name=COLLECTION
    )

    print(f"{len(chunks)} chunks stored in local ChromaDB!")
    return len(chunks)


def query_pdf(user_query: str):
    """Queries the Chroma DB and passes context to LLM for an answer."""
    vectorstore = get_vectorstore()

    results = vectorstore.similarity_search(user_query, k=3)

    context = "\n\n".join(
        f"Page Content: {r.page_content}\nPage Number: {r.metadata.get('page', 'N/A')}"
        for r in results
    )

    prompt = f"""You are a helpful AI assistant answering questions about an uploaded PDF document.
Use the context below to answer accurately. State page numbers if applicable.
Keep your response concise.

Context from PDF:
{context}

Question:
{user_query}"""

    return _call_llm(prompt)
