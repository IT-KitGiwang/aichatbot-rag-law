# src/utils/config.py
"""
Module quản lý cấu hình hệ thống.
Đọc từ config.yaml và .env
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Đường dẫn gốc dự án
BASE_DIR = Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = None) -> dict:
    """Đọc cấu hình từ config.yaml"""
    if config_path is None:
        config_path = BASE_DIR / "config.yaml"
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


class Settings:
    """Singleton class chứa toàn bộ cấu hình"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load()
        return cls._instance
    
    def _load(self):
        self.config = load_config()
        
        # API Keys từ .env
        self.google_api_key = os.getenv("GOOGLE_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        
        # PDF Processing
        self.raw_pdf_dir = BASE_DIR / self.config["pdf_processing"]["raw_pdf_dir"]
        self.processed_dir = BASE_DIR / self.config["pdf_processing"]["processed_dir"]
        
        # Chunking
        self.chunk_size = self.config["chunking"]["chunk_size"]
        self.chunk_overlap = self.config["chunking"]["chunk_overlap"]
        
        # Embedding
        self.embedding_model = self.config["embedding"]["model"]
        self.embedding_dimension = self.config["embedding"]["dimension"]
        
        # Vector Store
        self.chroma_persist_dir = str(BASE_DIR / self.config["vector_store"]["persist_directory"])
        self.chroma_collection = self.config["vector_store"]["collection_name"]
        
        # Retrieval
        self.dense_top_k = self.config["retrieval"]["dense_top_k"]
        self.sparse_top_k = self.config["retrieval"]["sparse_top_k"]
        self.rerank_top_k = self.config["retrieval"]["rerank_top_k"]
        self.relevance_threshold = self.config["retrieval"]["relevance_threshold"]
        
        # Reranker
        self.reranker_model = self.config["reranker"]["model"]
        
        # LLM
        self.llm_provider = self.config["llm"]["provider"]
        self.llm_model = self.config["llm"]["model"]
        self.llm_temperature = self.config["llm"]["temperature"]
        self.llm_max_tokens = self.config["llm"]["max_tokens"]
        
        # Validation
        self.validation = self.config["validation"]
        
        # API
        self.api_host = self.config["api"]["host"]
        self.api_port = self.config["api"]["port"]


# Singleton instance
settings = Settings()
