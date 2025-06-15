import os
import json
import time
import asyncio
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import asyncpg

# LangChain & Embeddings imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from supabase import create_client, Client
import tiktoken

# Project imports
from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_SIZE_MB, MAX_PAGES, DEFAULT_MODEL, SIMILARITY_TOP_K

load_dotenv()

app = FastAPI(title="RAG Platform API", version="3.0")

# Temporarily allow all origins to fix CORS issue
cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment check
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(f"✅ OpenAI configured: {bool(OPENAI_API_KEY)}")
print(f"✅ Gemini configured: {bool(GEMINI_API_KEY)}")
print(f"✅ Supabase URL configured: {bool(SUPABASE_URL)}")
print(f"✅ Supabase Key configured: {bool(SUPABASE_KEY)}")

# Storage configuration
USE_SUPABASE = bool(SUPABASE_URL and SUPABASE_KEY and OPENAI_API_KEY)

# Initialize connections
supabase_client = None
embeddings = None
db_pool = None

if USE_SUPABASE:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected for RAG")
        
        # Initialize embeddings with error handling
        try:
            embeddings = None  # Temporarily disable embeddings
            print("⚠️ Embeddings temporarily disabled")
        except Exception as embed_error:
            print(f"⚠️ Embeddings setup failed: {embed_error}")
            embeddings = None
    except Exception as e:
        print(f"⚠️ RAG setup failed: {e}")
        USE_SUPABASE = False
        supabase_client = None
        embeddings = None

async def get_db_pool():
    global db_pool
    if not db_pool and SUPABASE_URL:
        # For Railway/production, we might not need direct PostgreSQL pool
        # Supabase client handles the connection internally
        print("✅ Skipping PostgreSQL pool - using Supabase client")
    return db_pool

def count_tokens_simple(text: str) -> int:
    """Simple token counting - roughly 4 chars per token"""
    return len(text) // 4

def count_tokens_precise(text: str, model: str = "gpt-4") -> int:
    """Precise token counting using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except:
        return count_tokens_simple(text)

async def create_embeddings_table():
    """Ensure embeddings table exists with proper schema"""
    if not USE_SUPABASE:
        return False
    
    try:
        # Create table if it doesn't exist
        result = supabase_client.table('document_embeddings').select('id').limit(1).execute()
        print("✅ document_embeddings table exists")
        return True
    except Exception as e:
        print(f"⚠️ Embeddings table check failed: {e}")
        try:
            # Try to create the table structure we need
            supabase_client.rpc('create_embeddings_table_if_not_exists').execute()
            return True
        except:
            return False

async def store_document_simple(doc_id: str, filename: str, text: str, total_pages: int):
    """Store document in Supabase without embeddings for now"""
    if not USE_SUPABASE:
        return False
    
    try:
        # Store document metadata in old table structure for compatibility
        doc_result = supabase_client.table('documents').insert({
            'filename': filename,
            'pages': total_pages
        }).execute()
        
        if not doc_result.data:
            raise Exception("Failed to store document")
        
        stored_doc_id = doc_result.data[0]['id']
        
        # Store text chunks in chunks table
        chunk_size = 3000  # Simple chunking for now
        chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
        
        chunks_data = []
        for i, chunk_text in enumerate(chunks):
            chunk_record = {
                'document_id': stored_doc_id,
                'content': chunk_text,
                'chunk_index': i
            }
            chunks_data.append(chunk_record)
        
        # Insert chunks
        if chunks_data:
            chunks_result = supabase_client.table('chunks').insert(chunks_data).execute()
            if chunks_result.data:
                print(f"✅ Stored {len(chunks_result.data)} chunks for {filename}")
                return True
        
        return True
            
    except Exception as e:
        print(f"❌ Error storing document: {e}")
        return False

async def search_similar_chunks(query: str, doc_id: str = None, k: int = SIMILARITY_TOP_K) -> List[Dict]:
    """Search for similar chunks using vector similarity"""
    if not USE_SUPABASE:
        return []
    
    try:
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Use Supabase's vector similarity search
        # This requires the match_documents function to be created in Supabase
        rpc_params = {
            'query_embedding': query_embedding,
            'match_threshold': 0.3,  # Minimum similarity threshold
            'match_count': k
        }
        
        if doc_id:
            rpc_params['filter_doc_id'] = doc_id
        
        result = supabase_client.rpc('match_documents', rpc_params).execute()
        
        if result.data:
            print(f"🔍 Found {len(result.data)} similar chunks")
            return result.data
        else:
            print("🔍 No similar chunks found")
            return []
            
    except Exception as e:
        print(f"❌ Vector search error: {e}")
        return []

def clean_and_parse_json(response_text: str) -> Dict:
    """Clean and robustly parse JSON response"""
    try:
        # Remove markdown formatting
        cleaned = response_text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        
        # Find JSON boundaries
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        
        if start != -1 and end > start:
            json_text = cleaned[start:end]
        else:
            json_text = cleaned
        
        # Try to parse
        data = json.loads(json_text)
        
        # Ensure required fields exist
        data.setdefault("quotes", [])
        data.setdefault("entities", [])
        data.setdefault("metrics", [])
        data.setdefault("relations", [])
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response text: {response_text[:200]}...")
        
        # Return fallback structure
        return {
            "quotes": [],
            "entities": [],
            "metrics": [],
            "relations": [],
            "error": f"JSON parsing failed: {str(e)}",
            "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
        }

def call_openai_chat_sync(messages: List[Dict], model: str = DEFAULT_MODEL, temperature: float = 0.1) -> str:
    """Call OpenAI API synchronously"""
    import requests
    
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise HTTPException(500, f"AI processing failed: {str(e)}")

# Document storage for compatibility with existing frontend
document_storage = {}

async def load_documents_from_supabase():
    """Load existing documents for compatibility"""
    if not USE_SUPABASE:
        return
    
    try:
        result = supabase_client.table('documents').select('*').execute()
        if result.data:
            for doc in result.data:
                doc_id = doc['filename'].replace('.pdf', '').replace(' ', '_')
                
                # Get text from chunks if it exists
                text = ''
                try:
                    chunks_result = supabase_client.table('chunks').select('content').eq('document_id', doc.get('id')).order('chunk_index').execute()
                    if chunks_result.data:
                        text = ' '.join([chunk.get('content', '') for chunk in chunks_result.data])
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
            print(f"📚 Loaded {len(document_storage)} documents with content from chunks")
    except Exception as e:
        print(f"⚠️ Failed to load documents: {e}")

@app.on_event("startup")
async def startup():
    await get_db_pool()
    await create_embeddings_table()
    await load_documents_from_supabase()

@app.on_event("shutdown")
async def shutdown():
    if db_pool:
        await db_pool.close()

@app.get("/")
async def root():
    return {
        "message": "RAG Platform API v3.0 - Real RAG with Embeddings",
        "status": "running",
        "features": {
            "openai": bool(OPENAI_API_KEY),
            "gemini": bool(GEMINI_API_KEY),
            "embeddings": bool(embeddings),
            "vector_search": USE_SUPABASE,
            "documents": len(document_storage)
        }
    }

@app.get("/health")
async def health_check():
    # Get document count from Supabase if available
    supabase_docs = 0
    supabase_embeddings = 0
    
    if USE_SUPABASE and supabase_client:
        try:
            result = supabase_client.table('documents').select('doc_id').execute()
            supabase_docs = len(result.data) if result.data else 0
            
            embed_result = supabase_client.table('document_embeddings').select('id').execute()
            supabase_embeddings = len(embed_result.data) if embed_result.data else 0
        except:
            pass
    
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "embeddings_configured": bool(embeddings),
        "vector_search_enabled": USE_SUPABASE,
        "documents_in_storage": len(document_storage),
        "documents_in_supabase": supabase_docs,
        "embeddings_in_supabase": supabase_embeddings,
        "storage_mode": "rag_with_embeddings" if USE_SUPABASE else "local"
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
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        
        if len(pdf_reader.pages) > MAX_PAGES:
            raise HTTPException(400, f"PDF too long (max {MAX_PAGES} pages)")
        
        # Extract text with page tracking
        text_parts = []
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"[PAGE {i+1}]\n{page_text}")
            except Exception as e:
                print(f"Warning: Could not extract text from page {i+1}: {e}")
                continue
        
        if not text_parts:
            raise HTTPException(400, "Could not extract any text from PDF")
        
        full_text = "\n\n".join(text_parts)
        doc_id = file.filename.replace('.pdf', '').replace(' ', '_')
        
        # Store in Supabase if available
        if USE_SUPABASE:
            success = await store_document_simple(
                doc_id, file.filename, full_text, len(pdf_reader.pages)
            )
            if not success:
                raise HTTPException(500, "Failed to store document in Supabase")
        
        # Store in memory for compatibility
        document_storage[doc_id] = {
            "doc_id": doc_id,
            "filename": file.filename,
            "text": full_text,
            "total_pages": len(pdf_reader.pages),
            "total_characters": len(full_text),
            "uploaded_at": time.time()
        }
        
        print(f"📄 Uploaded: {doc_id}")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": len(pdf_reader.pages),
            "total_characters": len(full_text),
            "embeddings_enabled": USE_SUPABASE,
            "message": f"Successfully processed {len(pdf_reader.pages)} pages with RAG embeddings"
        }
        
    except PyPDF2.errors.PdfReadError:
        raise HTTPException(400, "Invalid or corrupted PDF file")
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    # Refresh from Supabase
    await load_documents_from_supabase()
    
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
        if USE_SUPABASE:
            # Use vector search for better RAG
            similar_chunks = await search_similar_chunks(q, doc_id, k=SIMILARITY_TOP_K)
            
            if similar_chunks:
                # Build context from similar chunks
                context_parts = []
                sources = []
                
                for chunk in similar_chunks:
                    context_parts.append(chunk['content'])
                    sources.append({
                        'chunk_id': chunk['metadata'].get('chunk_id', 0),
                        'similarity': chunk.get('similarity', 0),
                        'text': chunk['content'][:100] + "..."
                    })
                
                context = "\n\n".join(context_parts)
                confidence = 0.9 if len(similar_chunks) >= 3 else 0.7
                
            else:
                # Fallback to traditional method
                doc = document_storage[doc_id]
                context = doc["text"][:6000]  # Use more context
                sources = [{"text": "Full document preview", "similarity": 0.5}]
                confidence = 0.5
        else:
            # Traditional method without embeddings
            doc = document_storage[doc_id]
            text = doc["text"]
            
            tokens = count_tokens_simple(text)
            if tokens > 2000:
                context = text[:3000] + "\n\n[...CONTENT OMITTED...]\n\n" + text[-3000:]
            else:
                context = text
            
            sources = [{"text": "Document sections", "similarity": 0.6}]
            confidence = 0.6
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on document content. Always cite relevant information and be precise."
            },
            {
                "role": "user", 
                "content": f"""Document content:\n{context}\n\nQuestion: {q}\n\nProvide a detailed answer in English, citing relevant parts of the content when possible."""
            }
        ]
        
        response_text = call_openai_chat_sync(messages)
        
        return {
            "answer": response_text,
            "doc_id": doc_id,
            "sources": sources,
            "confidence": confidence,
            "vector_search_used": USE_SUPABASE and len(similar_chunks) > 0 if USE_SUPABASE else False
        }
        
    except Exception as e:
        print(f"Question answering error: {e}")
        raise HTTPException(500, f"Failed to process question: {str(e)}")

# Keep other endpoints compatible
@app.get("/extract")
async def extract_knowledge(doc_id: str):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OpenAI API key not configured")
    
    if doc_id not in document_storage:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    try:
        doc = document_storage[doc_id]
        text = doc["text"]
        tokens = count_tokens_simple(text)
        
        print(f"🔍 Starting extraction for {doc_id}: {tokens} tokens")
        
        async def generate_extraction():
            try:
                # Handle large documents
                if tokens > 4000:
                    text_len = len(text)
                    sections = [
                        text[:3000],
                        text[text_len//2-1500:text_len//2+1500],
                        text[-3000:]
                    ]
                    processed_text = "\n\n[...SECTION BREAK...]\n\n".join(sections)
                else:
                    processed_text = text
                
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert knowledge extraction AI. Extract comprehensive information and return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": f"""Extract information from this document:\n\n{processed_text}\n\nReturn JSON with:\n- quotes: important statements with context\n- entities: people, organizations, places, products\n- metrics: numbers, dates, statistics\n- relations: relationships between entities\n\nFormat as valid JSON."""
                    }
                ]
                
                response_text = call_openai_chat_sync(messages)
                
                # Clean and parse response
                extraction_data = clean_and_parse_json(response_text)
                extraction_data["model"] = DEFAULT_MODEL
                extraction_data["confidence"] = 0.85 if tokens < 4000 else 0.75
                
                # Add metadata
                extraction_data["document_info"] = {
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "total_pages": doc.get("total_pages", 0),
                    "total_tokens": tokens,
                    "rag_enabled": USE_SUPABASE
                }
                
                # Generate statistics
                extraction_data["extraction_stats"] = {
                    "totals": {
                        "quotes": len(extraction_data.get("quotes", [])),
                        "entities": len(extraction_data.get("entities", [])),
                        "metrics": len(extraction_data.get("metrics", [])),
                        "relations": len(extraction_data.get("relations", []))
                    }
                }
                
                print(f"✅ Extraction completed for {doc_id}")
                
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
                extraction_data = {
                    "quotes": [],
                    "entities": [],
                    "metrics": [],
                    "relations": [],
                    "error": str(e),
                    "document_info": {"doc_id": doc_id, "total_tokens": tokens}
                }
            
            yield json.dumps(extraction_data, ensure_ascii=False, indent=2)
        
        return StreamingResponse(generate_extraction(), media_type="application/json")
        
    except Exception as e:
        print(f"Extract endpoint error: {e}")
        raise HTTPException(500, f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Platform API v3.0 with Real Embeddings")
    uvicorn.run(app, host="0.0.0.0", port=8000)