"""
RAG Diagnosis Module
Owner: Yug Bhalodia

This module retrieves relevant medical guidelines from a vector database
and generates evidence-based diagnosis suggestions with citations.

Steps to complete:
    1. ✅ Skeleton ready
    2. ⬜ Collect medical documents (PubMed / WHO guidelines)
    3. ⬜ Run build_knowledge_base.py to create FAISS index
    4. ⬜ Connect to Groq/OpenAI API in .env
    5. ⬜ Replace placeholder generate() with real RAG pipeline
"""

import os
import pickle
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────
FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "./data/processed/medical_index.faiss")
DOCUMENTS_PATH   = os.getenv("DOCUMENTS_PATH",   "./data/processed/documents.pkl")
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K_DOCS       = 5
GROQ_API_KEY     = os.getenv("GROQ_API_KEY", "")


class RAGDiagnosisModule:

    def __init__(self):
        self.embedder  = None
        self.index     = None
        self.documents = []
        self.is_loaded = False

    def load(self):
        """Load embedding model and FAISS index."""
        print("[RAG Module] Loading sentence embedder...")
        from sentence_transformers import SentenceTransformer
        self.embedder = SentenceTransformer(EMBED_MODEL_NAME)

        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(DOCUMENTS_PATH):
            import faiss
            print("[RAG Module] Loading FAISS index and documents...")
            self.index     = faiss.read_index(FAISS_INDEX_PATH)
            with open(DOCUMENTS_PATH, "rb") as f:
                self.documents = pickle.load(f)
        else:
            print("[RAG Module] ⚠️  Knowledge base not found.")
            print("[RAG Module] Run build_knowledge_base.py first to index medical documents.")

        self.is_loaded = True
        print("[RAG Module] ✅ Loaded successfully.")

    def retrieve(self, query: str, top_k: int = TOP_K_DOCS) -> list[str]:
        """Find the most relevant medical documents for a query."""
        if not self.index or not self.documents:
            return ["[Knowledge base not built yet — run build_knowledge_base.py]"]

        query_vec = self.embedder.encode([query]).astype("float32")
        _, indices = self.index.search(query_vec, top_k)
        return [self.documents[i] for i in indices[0] if i < len(self.documents)]

    def generate_diagnosis(self, clinical_note: str) -> dict:
        """
        Retrieve relevant docs and generate evidence-based diagnosis.

        Args:
            clinical_note: Raw clinical text

        Returns:
            dict with diagnosis, citations, and confidence
        """
        if not self.is_loaded:
            self.load()

        retrieved_docs = self.retrieve(clinical_note)
        context        = "\n\n".join(retrieved_docs)

        # ── Call LLM (Groq is free and fast for students) ──
        if GROQ_API_KEY:
            diagnosis_text = self._call_groq(clinical_note, context)
        else:
            diagnosis_text = (
                "⚠️  No LLM API key configured. "
                "Add GROQ_API_KEY to your .env file. "
                "Get a free key at https://console.groq.com"
            )

        return {
            "diagnosis":    diagnosis_text,
            "citations":    retrieved_docs[:3],     # Top 3 source docs shown
            "sources_used": len(retrieved_docs),
            "note":         "Citations are from retrieved medical guidelines."
        }

    def _call_groq(self, clinical_note: str, context: str) -> str:
        """Call Groq API with clinical note and retrieved context."""
        from groq import Groq

        client = Groq(api_key=GROQ_API_KEY)
        prompt = f"""You are a clinical decision support assistant.

Based ONLY on the following medical guidelines and context, provide a 
possible diagnosis for the clinical note below. Always cite which guideline 
supports each point. If information is insufficient, say so clearly.

--- Medical Guidelines ---
{context}

--- Clinical Note ---
{clinical_note}

--- Instructions ---
1. List the most likely diagnoses (ranked by probability)
2. For each diagnosis, cite the relevant guideline
3. Suggest further investigations if needed
4. Do NOT make up information not present in the guidelines
"""
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
        )
        return response.choices[0].message.content


# ── Singleton instance used by the API ───────────────
rag_module = RAGDiagnosisModule()
