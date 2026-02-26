# 🤖 AI Chatbot RAG - Luật Hôn nhân & Kinh tế Việt Nam

Chatbot AI sử dụng kỹ thuật RAG (Retrieval-Augmented Generation) chuyên trả lời câu hỏi về Luật Hôn nhân & Gia đình và Luật Kinh tế Việt Nam.

## Cài đặt

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Điền API keys
```

## Chạy

```bash
# Backend API
uvicorn src.api.main:app --reload

# Frontend
streamlit run frontend/streamlit_app.py
```

## Tài liệu
Xem [ke_hoach_chatbot_rag.md](ke_hoach_chatbot_rag.md) để biết chi tiết kiến trúc và pipeline.
