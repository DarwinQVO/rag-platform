import os
import json
import time
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
from supabase import create_client, Client
from datetime import datetime
import uuid

load_dotenv()

app = FastAPI(title="RAG Platform API", version="2.0")

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://*.vercel.app",
    "*"  # Allow all for now, restrict later
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Initialize Supabase client if credentials available
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
else:
    print("⚠️ Supabase not configured - using in-memory storage")

# Fallback in-memory storage
memory_storage = {}

@app.get("/")
async def root():
    return {"message": "RAG Platform API v2.0", "storage": "supabase" if supabase else "memory"}

@app.get("/health")
async def health_check():
    doc_count = 0
    
    if supabase:
        try:
            result = supabase.table('documents').select('doc_id', count='exact').execute()
            doc_count = result.count if hasattr(result, 'count') else 0
        except:
            pass
    else:
        doc_count = len(memory_storage)
    
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "supabase_configured": bool(supabase),
        "documents_in_storage": doc_count,
        "storage_type": "supabase" if supabase else "memory"
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    # Read PDF
    pdf_content = await file.read()
    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
    
    # Extract text
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    
    # Create document
    doc_id = str(uuid.uuid4())
    doc_data = {
        "doc_id": doc_id,
        "filename": file.filename,
        "text": text[:50000],  # Limit text size for now
        "total_pages": len(pdf_reader.pages),
        "total_characters": len(text),
        "uploaded_at": datetime.utcnow().isoformat()
    }
    
    # Store document
    if supabase:
        try:
            # Store in Supabase
            result = supabase.table('documents').insert(doc_data).execute()
            
            # Simple text chunks for embeddings (without vector embeddings for now)
            chunks = [text[i:i+1000] for i in range(0, len(text), 800)][:10]  # Max 10 chunks
            
            for i, chunk in enumerate(chunks):
                chunk_data = {
                    "doc_id": doc_id,
                    "content": chunk,
                    "metadata": {"chunk_index": i, "filename": file.filename}
                }
                supabase.table('document_embeddings').insert(chunk_data).execute()
            
        except Exception as e:
            print(f"Supabase error: {e}")
            # Fallback to memory
            memory_storage[doc_id] = doc_data
    else:
        # Use memory storage
        memory_storage[doc_id] = doc_data
    
    return {
        "message": "Document uploaded successfully",
        "doc_id": doc_id,
        "filename": file.filename,
        "pages": len(pdf_reader.pages),
        "chunks": len(text) // 800
    }

@app.get("/documents")
async def list_documents():
    documents = []
    
    if supabase:
        try:
            result = supabase.table('documents').select('*').order('uploaded_at', desc=True).execute()
            documents = result.data if hasattr(result, 'data') else []
        except:
            documents = list(memory_storage.values())
    else:
        documents = list(memory_storage.values())
    
    return {"documents": documents}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    if supabase:
        try:
            # Delete from Supabase
            result = supabase.table('documents').delete().eq('doc_id', doc_id).execute()
            return {"message": "Document deleted", "doc_id": doc_id}
        except Exception as e:
            if doc_id in memory_storage:
                del memory_storage[doc_id]
                return {"message": "Document deleted from memory", "doc_id": doc_id}
            raise HTTPException(status_code=404, detail="Document not found")
    else:
        if doc_id in memory_storage:
            del memory_storage[doc_id]
            return {"message": "Document deleted", "doc_id": doc_id}
        raise HTTPException(status_code=404, detail="Document not found")

@app.get("/ask")
async def ask_question(doc_id: str, q: str):
    # Simplified response for now
    if not OPENAI_API_KEY:
        return {
            "answer": "OpenAI API key not configured. This is a demo response.",
            "sources": [],
            "confidence": 0.5
        }
    
    # Get document
    doc_text = ""
    if supabase:
        try:
            result = supabase.table('documents').select('text').eq('doc_id', doc_id).single().execute()
            doc_text = result.data['text'] if result.data else ""
        except:
            if doc_id in memory_storage:
                doc_text = memory_storage[doc_id]['text']
    else:
        if doc_id in memory_storage:
            doc_text = memory_storage[doc_id]['text']
    
    if not doc_text:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Simple response for now
    return {
        "answer": f"Based on the document, here's a response to '{q}'. (Note: Full AI integration pending)",
        "sources": [{"text": doc_text[:200], "page": 1}],
        "confidence": 0.8
    }

@app.get("/extract")
async def extract_knowledge(doc_id: str):
    # Simplified extraction for now
    return {
        "extraction_stats": {
            "totals": {
                "quotes": 5,
                "entities": 8,
                "metrics": 3,
                "relations": 2
            }
        },
        "quotes": [
            {"id": "q1", "text": "Sample quote from document", "page": 1, "importance": "high"}
        ],
        "entities": [
            {"id": "e1", "name": "Sample Entity", "type": "organization", "importance": "high"}
        ],
        "metrics": [
            {"id": "m1", "value": "100", "unit": "%", "type": "percentage", "context": "Sample metric"}
        ],
        "confidence": 0.85
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)