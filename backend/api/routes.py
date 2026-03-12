"""
MediCore API Routes
All three module endpoints defined here.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from modules.icd_coding.icd_module     import icd_module
from modules.rag_diagnosis.rag_module  import rag_module
from modules.mental_health.mental_module import mental_module


router = APIRouter()


# ── Shared request model ──────────────────────────────
class NoteRequest(BaseModel):
    clinical_note: str
    patient_id: str | None = None


# ─────────────────────────────────────────────────────
# 1. ICD-10 CODING  (Shravani)
# ─────────────────────────────────────────────────────
@router.post("/icd-predict", tags=["ICD-10 Coding"])
async def predict_icd(request: NoteRequest):
    """
    Predict top-5 ICD-10 codes from clinical notes.
    Returns codes with confidence scores.
    """
    if len(request.clinical_note.strip()) < 10:
        raise HTTPException(status_code=400, detail="Clinical note is too short.")

    result = icd_module.predict(request.clinical_note)
    return {"patient_id": request.patient_id, **result}


# ─────────────────────────────────────────────────────
# 2. RAG DIAGNOSIS  (Yug)
# ─────────────────────────────────────────────────────
@router.post("/diagnose", tags=["RAG Diagnosis"])
async def diagnose(request: NoteRequest):
    """
    Generate evidence-based diagnosis with citations from medical guidelines.
    """
    if len(request.clinical_note.strip()) < 10:
        raise HTTPException(status_code=400, detail="Clinical note is too short.")

    result = rag_module.generate_diagnosis(request.clinical_note)
    return {"patient_id": request.patient_id, **result}


# ─────────────────────────────────────────────────────
# 3. MENTAL HEALTH TRIAGE  (Krishna)
# ─────────────────────────────────────────────────────
@router.post("/mental-health-triage", tags=["Mental Health"])
async def mental_triage(request: NoteRequest):
    """
    Assess mental health risk level (Low / Moderate / High) from clinical note.
    """
    if len(request.clinical_note.strip()) < 10:
        raise HTTPException(status_code=400, detail="Clinical note is too short.")

    result = mental_module.triage(request.clinical_note)
    return {"patient_id": request.patient_id, **result}


# ─────────────────────────────────────────────────────
# 4. FULL ANALYSIS  (All modules combined — Harsh's UI)
# ─────────────────────────────────────────────────────
@router.post("/analyze", tags=["Full Analysis"])
async def full_analysis(request: NoteRequest):
    """
    Master endpoint — runs all 3 modules and returns a unified clinical report.
    """
    note = request.clinical_note
    if len(note.strip()) < 10:
        raise HTTPException(status_code=400, detail="Clinical note is too short.")

    icd_result    = icd_module.predict(note)
    rag_result    = rag_module.generate_diagnosis(note)
    mental_result = mental_module.triage(note)

    return {
        "patient_id":   request.patient_id,
        "icd_codes":    icd_result,
        "diagnosis":    rag_result,
        "mental_health": mental_result,
    }
