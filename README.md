# 🤖 AI Chatbot RAG — Luật Hôn nhân & Kinh tế Việt Nam

Chatbot AI sử dụng kỹ thuật **RAG** (Retrieval-Augmented Generation) chuyên trả lời câu hỏi về **Luật Hôn nhân & Gia đình** và **Luật Kinh tế** Việt Nam.

| Stack | Công nghệ |
|---|---|
| Backend | FastAPI · Python 3.11+ |
| Frontend | React · Vite |
| Database | PostgreSQL 16 |
| Vector DB | ChromaDB |
| Cache | Redis 7 |
| LLM | Google Gemini 2.0 Flash |
| Container | Docker Compose |

---

## 📋 Yêu cầu

- **Docker Desktop** ≥ 4.x (bao gồm Docker Compose)
- **Node.js** ≥ 18 (nếu chạy frontend local)
- **Python** ≥ 3.11 (nếu chạy backend local)
- **Google API Key** (Gemini) — lấy tại [aistudio.google.com](https://aistudio.google.com)

---

## 🚀 Cách 1: Chạy bằng Docker (Khuyến nghị)

### Bước 1 — Cấu hình môi trường

```bash
# Copy file env mẫu
copy .env.example .env

# Mở .env và điền API key
# GOOGLE_API_KEY=your_key_here
```

### Bước 2 — Đặt file PDF luật

Đặt file PDF luật vào thư mục `data/raw_pdfs/`:
```
data/raw_pdfs/
  ├── luat_hon_nhan_gia_dinh_2014.pdf
  └── luat_kinh_te.pdf
```

### Bước 3 — Khởi chạy toàn bộ hệ thống

```bash
# Build và chạy tất cả services (backend + frontend + postgres + redis)
docker compose up --build

# Hoặc chạy nền
docker compose up --build -d
```

Hệ thống sẽ tự động:
1. Khởi tạo **PostgreSQL** (port `5432`) — chờ healthcheck trước khi chạy backend
2. Khởi tạo **Redis** (port `6379`)
3. Build và chạy **FastAPI Backend** (port `8000`)
4. Build và chạy **React Frontend** (port `3000`)

### Bước 4 — Truy cập

| Service | URL |
|---|---|
| 🌐 **Frontend (React)** | [http://localhost:3000](http://localhost:3000) |
| ⚙️ **Backend API** | [http://localhost:8000](http://localhost:8000) |
| 📖 **API Docs (Swagger)** | [http://localhost:8000/docs](http://localhost:8000/docs) |

### Bước 5 — Nạp dữ liệu luật (Ingestion)

```bash
# Chạy pipeline nạp PDF vào vector database
docker compose exec backend python -m src.ingestion.run_ingestion
```

### Dừng hệ thống

```bash
# Dừng tất cả
docker compose down

# Dừng và xóa dữ liệu database (reset)
docker compose down -v
```

---

## 🛠️ Cách 2: Chạy Local (Development)

### Backend

```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# Cài dependencies
pip install -r requirements.txt

# Copy và cấu hình .env
copy .env.example .env
# Sửa POSTGRES_HOST=localhost trong .env (thay vì postgres)
# Sửa REDIS_URL=redis://localhost:6379/0

# Chạy database migration
alembic upgrade head

# Nạp dữ liệu luật
python -m src.ingestion.run_ingestion

# Chạy backend
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

> ⚠️ Khi chạy local cần có PostgreSQL và Redis chạy sẵn trên máy, hoặc chỉ chạy 2 service đó bằng Docker:
> ```bash
> docker compose up postgres redis -d
> ```

### Frontend

```bash
cd frontend

# Cài dependencies
npm install

# Chạy dev server
npm run dev
```

Frontend sẽ chạy tại [http://localhost:5173](http://localhost:5173) (Vite dev server).

---

## 📁 Cấu trúc dự án

```
CHATBOT RAG/
├── src/                    # Backend (FastAPI + RAG Pipeline)
│   ├── api/                #   API endpoints, middleware
│   ├── ingestion/          #   PDF → Chunks → Embeddings → ChromaDB
│   ├── retrieval/          #   Hybrid Search + Reranking
│   ├── generation/         #   LLM prompting + response
│   ├── validation/         #   Anti-hallucination (7 layers)
│   ├── utils/              #   Config, DB, Cache, Logger
│   └── pipeline.py         #   RAG Pipeline orchestrator
├── frontend/               # Frontend (React + Vite)
│   └── src/
│       ├── components/     #   ChatMessage, CitationCard, ...
│       ├── pages/          #   ChatPage, UploadPage
│       ├── services/       #   API calls
│       └── hooks/          #   useChat
├── data/raw_pdfs/          # ← Đặt file PDF luật vào đây
├── tests/                  # Unit & Integration tests
├── evaluation/             # RAGAS evaluation scripts
├── scripts/                # Setup, seed, evaluate
├── docs/                   # API reference, deployment guide
├── config.yaml             # Cấu hình RAG pipeline
├── docker-compose.yml      # Docker stack
└── requirements.txt        # Python dependencies
```

---

## 📖 Tài liệu

- **[ke_hoach_chatbot_rag.md](ke_hoach_chatbot_rag.md)** — Kế hoạch chi tiết: kiến trúc, pipeline RAG, kỹ thuật chống hallucination, lộ trình 8 tuần
- **[docs/api_reference.md](docs/api_reference.md)** — API endpoints cho frontend
- **[docs/deployment.md](docs/deployment.md)** — Hướng dẫn triển khai production

---

## ⚡ Các lệnh thường dùng

```bash
# Docker
docker compose up --build          # Chạy toàn bộ
docker compose up --build -d       # Chạy nền
docker compose down                # Dừng
docker compose down -v             # Dừng + xóa data
docker compose logs backend -f     # Xem log backend
docker compose logs postgres -f    # Xem log database
docker compose exec backend bash   # Truy cập container backend

# Backend (local)
uvicorn src.api.main:app --reload  # Chạy dev server
alembic upgrade head               # Chạy migration
alembic revision --autogenerate -m "message"  # Tạo migration mới
pytest                             # Chạy tests

# Frontend (local)
cd frontend && npm run dev         # Chạy dev server
cd frontend && npm run build       # Build production
```
