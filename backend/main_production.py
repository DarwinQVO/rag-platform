import os
import json
import time
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import PyPDF2
import fitz  # PyMuPDF - much better for text extraction
from io import BytesIO
import requests

# Essential imports only
from supabase import create_client
import tiktoken

# Check if PyMuPDF is available
try:
    import fitz
    PYMUPDF_AVAILABLE = True
    print("✅ PyMuPDF (fitz) loaded successfully")
except ImportError as e:
    PYMUPDF_AVAILABLE = False
    print(f"⚠️ PyMuPDF not available: {e}")
    print("⚠️ Will use PyPDF2 only")

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
    """Extract text from PDF using PyMuPDF (superior) and PyPDF2 fallback"""
    
    print(f"🔍 Starting PDF extraction with {len(content)} bytes")
    
    # Method 1: Try PyMuPDF first (best for complex PDFs)
    if PYMUPDF_AVAILABLE:
        try:
            print("🚀 Attempting PyMuPDF extraction...")
            doc = fitz.open(stream=content, filetype="pdf")
            page_count = len(doc)
            print(f"📄 PyMuPDF: Found {page_count} pages")
            
            text_parts = []
            total_chars = 0
            
            for page_num in range(min(page_count, 50)):  # Limit pages for stability
                try:
                    page = doc.load_page(page_num)
                    
                    # Extract text with PyMuPDF (handles complex layouts better)
                    page_text = page.get_text()
                    
                    # Try different extraction methods if standard fails
                    if not page_text or len(page_text.strip()) < 10:
                        page_text = page.get_text("text")
                        
                    if not page_text or len(page_text.strip()) < 10:
                        blocks = page.get_text("blocks")
                        page_text = "\n".join([block[4] for block in blocks if len(block) > 4 and isinstance(block[4], str)])
                    
                    if page_text and page_text.strip():
                        clean_text = page_text.strip()
                        # Clean up excessive whitespace but preserve structure
                        import re
                        clean_text = re.sub(r'\n\s*\n', '\n\n', clean_text)
                        clean_text = re.sub(r' +', ' ', clean_text)
                        
                        text_parts.append(f"[PAGE {page_num + 1}]\n{clean_text}")
                        total_chars += len(clean_text)
                        print(f"📄 Page {page_num + 1}: {len(clean_text)} characters extracted")
                    else:
                        print(f"⚠️ Page {page_num + 1}: No text found")
                except Exception as e:
                    print(f"❌ Error processing page {page_num + 1}: {e}")
                    continue
            
            doc.close()
            
            if text_parts and total_chars > 100:
                full_text = "\n\n".join(text_parts)
                print(f"✅ PyMuPDF SUCCESS: extracted {total_chars} characters from {len(text_parts)} pages")
                return full_text, page_count
            else:
                print(f"⚠️ PyMuPDF found insufficient text ({total_chars} chars), trying PyPDF2...")
        
        except Exception as e:
            print(f"❌ PyMuPDF completely failed: {e}")
    else:
        print("⚠️ PyMuPDF not available, using PyPDF2...")
    
    # Method 2: Fallback to PyPDF2 with enhanced extraction
    try:
        print("🚀 Attempting PyPDF2 extraction...")
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        pages = len(pdf_reader.pages)
        print(f"📄 PyPDF2: Found {pages} pages")
        
        text_parts = []
        total_chars = 0
        
        for i, page in enumerate(pdf_reader.pages[:50]):  # Limit pages
            try:
                # Try multiple PyPDF2 extraction methods
                page_text = page.extract_text()
                
                # Try alternative extraction if first fails
                if not page_text or len(page_text.strip()) < 10:
                    try:
                        # Alternative extraction method
                        if hasattr(page, 'extractText'):
                            page_text = page.extractText()
                    except:
                        pass
                
                if page_text and page_text.strip():
                    clean_text = page_text.strip()
                    text_parts.append(f"[PAGE {i+1}]\n{clean_text}")
                    total_chars += len(clean_text)
                    print(f"📄 Page {i+1}: {len(clean_text)} characters")
                else:
                    print(f"⚠️ Page {i+1}: No text extracted")
                    
            except Exception as e:
                print(f"❌ Page {i+1} error: {e}")
                continue
        
        if text_parts and total_chars > 20:  # Lower threshold for PyPDF2
            full_text = "\n\n".join(text_parts)
            print(f"✅ PyPDF2 SUCCESS: extracted {total_chars} characters from {len(text_parts)} pages")
            return full_text, pages
        else:
            print(f"❌ PyPDF2 insufficient text: {total_chars} chars from {len(text_parts)} pages")
    
    except Exception as e:
        print(f"❌ PyPDF2 completely failed: {e}")
    
    # If both methods fail completely
    print("❌ BOTH extraction methods failed completely")
    raise Exception(f"Could not extract any text from PDF using PyMuPDF or PyPDF2. PDF may be image-based, corrupted, or password-protected.")

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
        
        print(f"📊 Extracted text preview: {full_text[:200]}...")
        print(f"📊 Total characters extracted: {len(full_text)}")
        
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
                    
                    # Store embeddings 
                    try:
                        # Check if embeddings table exists
                        embeddings_result = supabase_client.table('document_embeddings').select('id').limit(1).execute()
                        
                        # Generate and store embeddings for each chunk
                        embeddings_data = []
                        for i, chunk in enumerate(chunks[:20]):  # Limit to 20 chunks for now
                            try:
                                embedding = generate_embedding(chunk[:2000])  # Limit text size
                                embeddings_data.append({
                                    'doc_id': doc_uuid,
                                    'content': chunk,
                                    'embedding': embedding,
                                    'metadata': {
                                        'doc_id': doc_id,
                                        'chunk_id': i,
                                        'filename': file.filename
                                    }
                                })
                            except Exception as e:
                                print(f"Embedding generation error: {e}")
                                continue
                        
                        if embeddings_data:
                            supabase_client.table('document_embeddings').insert(embeddings_data).execute()
                            print(f"✅ Stored {len(embeddings_data)} embeddings")
                    except Exception as e:
                        print(f"Embeddings storage error: {e}")
                    
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
        
        # Try vector search first if available
        context = ""
        sources = []
        
        if supabase_client and OPENAI_API_KEY:
            try:
                # Generate query embedding
                query_embedding = generate_embedding(q)
                
                # Search similar chunks using RPC function
                result = supabase_client.rpc('match_documents', {
                    'query_embedding': query_embedding,
                    'filter_doc_id': doc_id,
                    'match_threshold': 0.3,
                    'match_count': 6
                }).execute()
                
                if result.data and len(result.data) > 0:
                    # Use similar chunks as context
                    context_parts = []
                    for chunk in result.data:
                        context_parts.append(chunk['content'])
                        sources.append({
                            'chunk_id': chunk['metadata'].get('chunk_id', 0),
                            'similarity': chunk.get('similarity', 0)
                        })
                    context = "\n\n".join(context_parts)
                    print(f"✅ Found {len(result.data)} similar chunks")
            except Exception as e:
                print(f"Vector search error: {e}")
        
        # Fallback to simple text if vector search fails
        if not context:
            context = text[:8000] if len(text) > 8000 else text
            sources = [{"text": "Full document context"}]
        
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
            "confidence": min(0.95, 0.7 + len(sources) * 0.05) if sources else 0.7,
            "sources": sources,
            "vector_search_used": len(sources) > 1
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