# 🏥 MediCore — Unified Clinical Intelligence & Triage System

> BTech CSE Capstone Project | MIT-WPU | Group P29

---

## 👥 Team

| Name | PRN | Module |
|---|---|---|
| Harsh Halwai | 1032233383 | UI & Integration |
| Shravani Shinde | 1032230482 | ICD-10 Coding |
| Krishna Kumar | 1032222505 | Mental Health Triage |
| Yug Bhalodia | 1032221643 | RAG Diagnosis |

**Guide:** Dr. Dipali Baviskar

---

## 🚀 Getting Started (Do This First — Everyone)

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/medicore.git
cd medicore
```

### 2. Create Python virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Open .env and add your Groq API key (free at https://console.groq.com)
```

### 5. Run the backend
```bash
cd backend
uvicorn main:app --reload --port 8000
```

### 6. Open API docs in browser
```
http://localhost:8000/docs
```

---

## 📁 Project Structure

```
medicore/
├── backend/
│   ├── main.py                         ← FastAPI app entry point
│   ├── api/
│   │   └── routes.py                   ← All API endpoints
│   ├── modules/
│   │   ├── icd_coding/
│   │   │   ├── icd_module.py           ← Shravani: BioBERT ICD predictor
│   │   │   ├── data_prep.py            ← Shravani: Dataset preprocessing
│   │   │   └── train.py                ← Shravani: Model training script
│   │   ├── rag_diagnosis/
│   │   │   ├── rag_module.py           ← Yug: RAG pipeline
│   │   │   └── build_knowledge_base.py ← Yug: Index medical documents
│   │   └── mental_health/
│   │       ├── mental_module.py        ← Krishna: Risk triage
│   │       └── train.py                ← Krishna: MentalBERT training
│   └── utils/
│       └── text_cleaner.py             ← Shared text preprocessing
├── frontend/                           ← Harsh: React/Next.js app
├── data/
│   ├── raw/                            ← Raw datasets (never commit to git)
│   └── processed/                      ← FAISS index, cleaned data
├── models/                             ← Saved trained models
├── tests/                              ← Unit tests
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🔌 API Endpoints

| Method | Endpoint | Owner | Description |
|---|---|---|---|
| GET | `/health` | — | Health check |
| POST | `/api/v1/icd-predict` | Shravani | Predict ICD-10 codes |
| POST | `/api/v1/diagnose` | Yug | RAG-based diagnosis |
| POST | `/api/v1/mental-health-triage` | Krishna | Risk classification |
| POST | `/api/v1/analyze` | All | Full unified analysis |

### Example request body
```json
{
  "clinical_note": "Patient presents with persistent cough, high fever (103°F), and difficulty breathing for 5 days. History of smoking.",
  "patient_id": "P001"
}
```

---

## 🗺️ Development Roadmap

- [x] Project skeleton
- [ ] ICD-10 model training (Shravani)
- [ ] RAG knowledge base (Yug)
- [ ] MentalBERT training (Krishna)
- [ ] Frontend dashboard (Harsh)
- [ ] Integration testing
- [ ] Deployment

---

## 📊 Evaluation Targets

| Module | Metric | Target |
|---|---|---|
| ICD-10 Coding | F1@5 | > 0.70 |
| RAG Diagnosis | Relevance | > 0.75 |
| Mental Health | Recall (High risk) | > 0.85 |
| API Response | Latency | < 3 seconds |