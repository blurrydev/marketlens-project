**MarketLens: Enterprise Grade Financial RAG Engine**
MarketLens is a full-stack Retrieval-Augmented Generation (RAG) system designed to securely query complex financial documents (like Annual Reports).

Unlike standard AI wrappers, this project implements a custom, low level data ingestion and retrieval pipeline. It utilizes native Hugging Face models for privacy first, on device vector embeddings, and stores them in a local ChromaDB instance. The REST API is built with FastAPI, featuring strict multi-tenant architecture that safely injects user-provided Gemini API keys at runtime without compromising server state.

**Key Engineering Highlights:**

* **Privacy First Embeddings:** Uses `SentenceTransformers` (`all-MiniLM-L6-v2`) to run vector math completely locally, preventing sensitive document leaks to third party APIs.
* **Manual Vector Orchestration:** Bypasses high level framework abstractions to manually handle document chunking, metadata schemas, and distance based neighbor retrieval.
* **Stateless Multi-Tenancy:** The FastAPI backend securely handles per-request authentication, instantly garbage collecting LLM instances to prevent memory leaks and token cross contamination.
* **Fully Typed & Documented:** Utilizes Pydantic for strict schema validation and automatically generates OpenAPI (Swagger) documentation.
