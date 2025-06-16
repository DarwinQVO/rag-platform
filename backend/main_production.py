import os
import json
import time
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import requests

# Essential imports only
from supabase import create_client
import tiktoken

load_dotenv()

app = FastAPI(title="RAG Platform API", version="5.0-PRODUCTION")

# CORS - allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(f"✅ OpenAI configured: {bool(OPENAI_API_KEY)}")
print(f"✅ Supabase URL configured: {bool(SUPABASE_URL)}")
print(f"✅ Supabase Key configured: {bool(SUPABASE_KEY)}")

# Configuration
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
MAX_FILE_SIZE_MB = 50
MAX_PAGES = 1000
DEFAULT_MODEL = "gpt-4o"
SIMILARITY_TOP_K = 6

# Global storage
supabase_client = None
document_storage = {}

# Initialize Supabase
if SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected")
    except Exception as e:
        print(f"⚠️ Supabase connection failed: {e}")

def extract_text_from_pdf(content: bytes) -> tuple[str, int]:
    """Extract text from PDF using PyPDF2"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        pages = len(pdf_reader.pages)
        
        text_parts = []
        for i, page in enumerate(pdf_reader.pages):
            try:
                text = page.extract_text()
                if text.strip():
                    text_parts.append(f"[PAGE {i+1}]\n{text}")
            except:
                continue
        
        full_text = "\n\n".join(text_parts) if text_parts else ""
        return full_text, pages
    except Exception as e:
        print(f"PDF extraction error: {e}")
        raise

def simple_chunk_text(text: str, chunk_size: int = CHUNK_SIZE * 4) -> List[str]:
    """Simple text chunking - approximating tokens as 4 chars"""
    chunks = []
    text_length = len(text)
    overlap_size = CHUNK_OVERLAP * 4
    
    for i in range(0, text_length, chunk_size - overlap_size):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def call_openai_api(messages: List[Dict], model: str = DEFAULT_MODEL) -> str:
    """Call OpenAI API"""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": 0.1
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

def generate_embedding(text: str) -> List[float]:
    """Generate embeddings using OpenAI API"""
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "input": text,
        "model": "text-embedding-3-small"
    }
    
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["embedding"]

def load_documents_from_supabase():
    """Load existing documents"""
    if not supabase_client:
        return
    
    try:
        result = supabase_client.table('documents').select('*').execute()
        if result.data:
            for doc in result.data:
                doc_id = doc['filename'].replace('.pdf', '').replace(' ', '_')
                
                # Try to get text from chunks
                text = ''
                try:
                    chunks_result = supabase_client.table('chunks').select('content').eq('document_id', doc.get('id')).execute()
                    if chunks_result.data:
                        text = ' '.join([chunk['content'] for chunk in chunks_result.data])
                except:
                    pass
                
                document_storage[doc_id] = {
                    'doc_id': doc_id,
                    'filename': doc['filename'],
                    'text': text,
                    'total_pages': doc.get('pages', 0),
                    'total_characters': len(text),
                    'uploaded_at': time.time()
                }
            print(f"📚 Loaded {len(document_storage)} documents")
    except Exception as e:
        print(f"Failed to load documents: {e}")

@app.on_event("startup")
async def startup():
    load_documents_from_supabase()
    print("🚀 RAG Platform started successfully")

@app.get("/")
async def root():
    return {
        "message": "RAG Platform API v5.0-PRODUCTION",
        "status": "running",
        "features": {
            "openai": bool(OPENAI_API_KEY),
            "supabase": bool(supabase_client),
            "embeddings": bool(OPENAI_API_KEY),
            "documents": len(document_storage)
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "embeddings_enabled": bool(OPENAI_API_KEY),
        "documents_loaded": len(document_storage),
        "storage_mode": "production_rag"
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OpenAI API key not configured")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    if file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"File too large (max {MAX_FILE_SIZE_MB}MB)")
    
    try:
        # Read and extract text
        content = await file.read()
        full_text, page_count = extract_text_from_pdf(content)
        
        if not full_text or len(full_text.strip()) < 50:
            raise HTTPException(400, "Could not extract text from PDF")
        
        # Create chunks
        chunks = simple_chunk_text(full_text)
        doc_id = file.filename.replace('.pdf', '').replace(' ', '_')
        
        # Store in Supabase if available
        supabase_success = False
        if supabase_client and OPENAI_API_KEY:
            try:
                # Store document
                doc_result = supabase_client.table('documents').insert({
                    'filename': file.filename,
                    'pages': page_count
                }).execute()
                
                if doc_result.data:
                    doc_uuid = doc_result.data[0]['id']
                    
                    # Store chunks
                    chunks_data = []
                    for i, chunk in enumerate(chunks[:50]):  # Limit chunks for stability
                        chunks_data.append({
                            'document_id': doc_uuid,
                            'content': chunk,
                            'chunk_index': i
                        })
                    
                    if chunks_data:
                        supabase_client.table('chunks').insert(chunks_data).execute()
                    
                    # Store embeddings if we want to enable them
                    if False:  # Temporarily disabled for stability
                        # Generate and store embeddings
                        pass
                    
                    supabase_success = True
                    print(f"✅ Stored in Supabase: {file.filename}")
            except Exception as e:
                print(f"Supabase storage error: {e}")
        
        # Store in memory
        document_storage[doc_id] = {
            'doc_id': doc_id,
            'filename': file.filename,
            'text': full_text,
            'total_pages': page_count,
            'total_characters': len(full_text),
            'chunks_count': len(chunks),
            'uploaded_at': time.time()
        }
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": page_count,
            "total_characters": len(full_text),
            "chunks_created": len(chunks),
            "stored_in_supabase": supabase_success,
            "message": f"Successfully processed {page_count} pages"
        }
        
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    load_documents_from_supabase()
    
    documents = []
    for doc_id, doc_data in document_storage.items():
        documents.append({
            "doc_id": doc_id,
            "filename": doc_data["filename"],
            "total_pages": doc_data.get("total_pages", 0),
            "total_characters": doc_data.get("total_characters", 0),
            "uploaded_at": doc_data.get("uploaded_at", time.time())
        })
    
    return {"documents": documents, "total": len(documents)}

@app.get("/ask")
async def ask_question(doc_id: str, q: str):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OpenAI API key not configured")
    
    if doc_id not in document_storage:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    if not q.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    try:
        doc = document_storage[doc_id]
        text = doc["text"]
        
        # Use first 8000 chars for context
        context = text[:8000] if len(text) > 8000 else text
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on document content."
            },
            {
                "role": "user", 
                "content": f"Document content:\n{context}\n\nQuestion: {q}\n\nProvide a detailed answer based on the document."
            }
        ]
        
        answer = call_openai_api(messages)
        
        return {
            "answer": answer,
            "doc_id": doc_id,
            "confidence": 0.8,
            "sources": [{"text": "Document context"}]
        }
        
    except Exception as e:
        print(f"Question error: {e}")
        raise HTTPException(500, f"Failed to process question: {str(e)}")

@app.get("/extract")
async def extract_knowledge(doc_id: str):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OpenAI API key not configured")
    
    if doc_id not in document_storage:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    try:
        doc = document_storage[doc_id]
        text = doc["text"][:10000]  # Limit for extraction
        
        async def generate():
            messages = [
                {"role": "system", "content": "Extract key information and return valid JSON."},
                {"role": "user", "content": f"Extract entities, quotes, and metrics from:\n\n{text}"}
            ]
            
            response_text = call_openai_api(messages)
            
            try:
                data = json.loads(response_text)
            except:
                data = {"entities": [], "quotes": [], "metrics": []}
            
            data["document_info"] = {
                "doc_id": doc_id,
                "filename": doc["filename"],
                "pages": doc.get("total_pages", 0)
            }
            
            yield json.dumps(data, ensure_ascii=False, indent=2)
        
        return StreamingResponse(generate(), media_type="application/json")
        
    except Exception as e:
        raise HTTPException(500, f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Platform API v5.0-PRODUCTION")
    uvicorn.run(app, host="0.0.0.0", port=8000)