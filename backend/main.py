"""
MediCore - Unified Clinical Intelligence & Triage System
Main FastAPI application entry point
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Import routers (uncomment as each module is built)
# from api.icd_routes import router as icd_router
# from api.rag_routes import router as rag_router
# from api.mental_routes import router as mental_router

# ── App Setup ────────────────────────────────────────
app = FastAPI(
    title="MediCore API",
    description="Unified Clinical Intelligence & Triage System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],   # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Include Routers ───────────────────────────────────
# app.include_router(icd_router,    prefix="/api/v1", tags=["ICD-10 Coding"])
# app.include_router(rag_router,    prefix="/api/v1", tags=["RAG Diagnosis"])
# app.include_router(mental_router, prefix="/api/v1", tags=["Mental Health"])


# ── Request / Response Models ─────────────────────────
class ClinicalNoteRequest(BaseModel):
    clinical_note: str
    patient_id: str | None = None    # Optional, for session tracking


class FullAnalysisResponse(BaseModel):
    patient_id: str | None
    icd_codes: dict
    diagnosis: dict
    mental_health: dict


# ── Health Check ──────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "MediCore API is running ✅",
        "version": "1.0.0",
        "modules": ["ICD-10 Coding", "RAG Diagnosis", "Mental Health Triage"]
    }

@app.get("/health")
async def health():
    return {"status": "healthy"}


# ── Full Analysis Endpoint ────────────────────────────
@app.post("/api/v1/analyze", response_model=FullAnalysisResponse)
async def full_analysis(request: ClinicalNoteRequest):
    """
    Master endpoint — runs all 3 modules and returns unified report.
    Each sub-module will be plugged in as it is built.
    """
    note = request.clinical_note

    if not note or len(note.strip()) < 10:
        raise HTTPException(status_code=400, detail="Clinical note is too short.")

    # ── Placeholder results (replace with real modules) ──
    icd_result = {
        "status": "module_not_ready",
        "message": "ICD-10 module coming soon — Shravani's task"
    }
    rag_result = {
        "status": "module_not_ready",
        "message": "RAG Diagnosis module coming soon — Yug's task"
    }
    mental_result = {
        "status": "module_not_ready",
        "message": "Mental Health module coming soon — Krishna's task"
    }

    return FullAnalysisResponse(
        patient_id=request.patient_id,
        icd_codes=icd_result,
        diagnosis=rag_result,
        mental_health=mental_result,
    )


# ── Run Directly ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", 8000)),
        reload=os.getenv("DEBUG", "True") == "True",
    )
