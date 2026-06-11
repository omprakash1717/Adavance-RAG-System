import sys
import time
from rag_engine import process_pdf

if __name__ == "__main__":
    start = time.time()
    try:
        chunks = process_pdf("PDF-Guide-Node-Andrew-Mead-v3.pdf")
        print(f"\nSUCCESS! Processed {chunks} chunks in {time.time() - start:.2f} seconds.")
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
