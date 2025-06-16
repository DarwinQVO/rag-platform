import os
import json
import time
import asyncio
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import requests
from io import BytesIO
import hashlib

# PDF Processing imports
import PyPDF2
import fitz  # PyMuPDF for better text extraction

# LangChain & Embeddings imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document
from supabase import create_client, Client
import tiktoken

# Project imports
from config import CHUNK_SIZE, CHUNK_OVERLAP, MAX_FILE_SIZE_MB, MAX_PAGES, DEFAULT_MODEL, SIMILARITY_TOP_K

load_dotenv()

app = FastAPI(title="RAG Platform API - COMPLETE", version="4.0")

# CORS configuration
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

# Global variables
supabase_client = None
embeddings = None
document_storage = {}

# Initialize Supabase and embeddings
USE_EMBEDDINGS = bool(OPENAI_API_KEY and SUPABASE_URL and SUPABASE_KEY)

if USE_EMBEDDINGS:
    try:
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small"
        )
        print("✅ Supabase connected")
        print("✅ OpenAI embeddings ready")
    except Exception as e:
        print(f"⚠️ RAG setup failed: {e}")
        USE_EMBEDDINGS = False

def extract_text_advanced(file_content: bytes, filename: str) -> tuple[str, int]:
    """
    Advanced text extraction using multiple methods
    Returns: (text, page_count)
    """
    text_parts = []
    page_count = 0
    
    # Method 1: Try PyMuPDF first (best for most PDFs)
    try:
        doc = fitz.open(stream=file_content, filetype="pdf")
        page_count = len(doc)
        print(f"📄 PyMuPDF: Found {page_count} pages")
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text()
            
            if text.strip():
                text_parts.append(f"[PAGE {page_num + 1}]\n{text.strip()}")
                print(f"📄 Page {page_num + 1}: {len(text)} characters")
        
        doc.close()
        
        if text_parts:
            full_text = "\n\n".join(text_parts)
            print(f"✅ PyMuPDF extracted {len(full_text)} characters")
            return full_text, page_count
            
    except Exception as e:
        print(f"⚠️ PyMuPDF failed: {e}")
    
    # Method 2: Fallback to PyPDF2
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(file_content))
        page_count = len(pdf_reader.pages)
        print(f"📄 PyPDF2: Found {page_count} pages")
        
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"[PAGE {i + 1}]\n{page_text.strip()}")
                    print(f"📄 Page {i + 1}: {len(page_text)} characters")
            except Exception as e:
                print(f"⚠️ Error on page {i + 1}: {e}")
                continue
        
        if text_parts:
            full_text = "\n\n".join(text_parts)
            print(f"✅ PyPDF2 extracted {len(full_text)} characters")
            return full_text, page_count
            
    except Exception as e:
        print(f"⚠️ PyPDF2 failed: {e}")
    
    # Method 3: If no text extracted, provide informative message
    if page_count > 0:
        placeholder_text = f"""[DOCUMENT: {filename}]
[PAGES: {page_count}]
[STATUS: This appears to be an image-based or scanned PDF]
[NOTE: Text extraction was not possible with available methods]
[RECOMMENDATION: For scanned documents, OCR processing would be required]"""
        return placeholder_text, page_count
    
    raise Exception("Could not process PDF file")

def chunk_text_intelligently(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Intelligent text chunking using tiktoken
    """
    try:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            encoding_name="cl100k_base",
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_text(text)
        print(f"📝 Created {len(chunks)} chunks (~{chunk_size} tokens each)")
        return chunks
    except Exception as e:
        print(f"⚠️ Chunking failed: {e}")
        # Fallback to simple chunking
        chunk_length = chunk_size * 4  # Approximate tokens to characters
        return [text[i:i+chunk_length] for i in range(0, len(text), chunk_length - overlap * 4)]

def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts
    """
    if not embeddings:
        raise Exception("Embeddings not configured")
    
    try:
        # Generate embeddings in batch
        embedding_vectors = embeddings.embed_documents(texts)
        print(f"✅ Generated {len(embedding_vectors)} embeddings")
        return embedding_vectors
    except Exception as e:
        print(f"❌ Embedding generation failed: {e}")
        raise

def store_in_supabase(doc_id: str, filename: str, text: str, chunks: List[str], embeddings_list: List[List[float]], pages: int) -> bool:
    """
    Store document and embeddings in Supabase
    """
    if not supabase_client:
        return False
    
    try:
        # 1. Store document metadata
        doc_data = {
            'filename': filename,
            'text': text[:10000],  # Store preview
            'total_pages': pages,
            'total_characters': len(text),
            'created_at': 'now()'
        }
        
        doc_result = supabase_client.table('documents').insert(doc_data).execute()
        if not doc_result.data:
            raise Exception("Failed to insert document")
        
        document_uuid = doc_result.data[0]['doc_id']
        print(f"✅ Document stored with UUID: {document_uuid}")
        
        # 2. Store chunks with embeddings
        chunks_data = []
        for i, (chunk, embedding_vector) in enumerate(zip(chunks, embeddings_list)):
            chunk_data = {
                'doc_id': document_uuid,
                'content': chunk,
                'embedding': embedding_vector,
                'metadata': {
                    'doc_id': doc_id,
                    'chunk_id': i,
                    'filename': filename,
                    'total_chunks': len(chunks),
                    'tokens': len(chunk) // 4  # Approximate
                }
            }
            chunks_data.append(chunk_data)
        
        # Insert in batches to avoid timeouts
        batch_size = 50
        for i in range(0, len(chunks_data), batch_size):
            batch = chunks_data[i:i+batch_size]
            result = supabase_client.table('document_embeddings').insert(batch).execute()
            if not result.data:
                raise Exception(f"Failed to insert batch {i//batch_size + 1}")
            print(f"✅ Stored batch {i//batch_size + 1}: {len(result.data)} chunks")
        
        print(f"✅ Successfully stored {len(chunks)} chunks with embeddings")
        return True
        
    except Exception as e:
        print(f"❌ Supabase storage failed: {e}")
        return False

def call_openai_api(messages: List[Dict], model: str = DEFAULT_MODEL, temperature: float = 0.1) -> str:
    """
    Call OpenAI API for chat completions
    """
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
        response = requests.post(url, headers=headers, json=data, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"OpenAI API error: {e}")
        raise HTTPException(500, f"AI processing failed: {str(e)}")

def search_similar_embeddings(query: str, doc_id: str = None, k: int = SIMILARITY_TOP_K) -> List[Dict]:
    """
    Search for similar chunks using vector similarity
    """
    if not (embeddings and supabase_client):
        return []
    
    try:
        # Generate query embedding
        query_embedding = embeddings.embed_query(query)
        
        # Use Supabase RPC for vector similarity search
        rpc_params = {
            'query_embedding': query_embedding,
            'match_threshold': 0.3,
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

def load_documents_from_storage():
    """
    Load documents from Supabase into memory for compatibility
    """
    if not supabase_client:
        return
    
    try:
        result = supabase_client.table('documents').select('*').execute()
        if result.data:
            for doc in result.data:
                doc_id = doc['filename'].replace('.pdf', '').replace(' ', '_')
                document_storage[doc_id] = {
                    'doc_id': doc_id,
                    'filename': doc['filename'],
                    'text': doc.get('text', ''),
                    'total_pages': doc.get('total_pages', 0),
                    'total_characters': doc.get('total_characters', 0),
                    'uploaded_at': time.time()
                }
            print(f"📚 Loaded {len(document_storage)} documents")
    except Exception as e:
        print(f"⚠️ Failed to load documents: {e}")

@app.on_event("startup")
async def startup():
    load_documents_from_storage()

@app.get("/")
async def root():
    return {
        "message": "RAG Platform API v4.0 - COMPLETE IMPLEMENTATION",
        "status": "running",
        "features": {
            "openai": bool(OPENAI_API_KEY),
            "supabase": bool(supabase_client),
            "embeddings": bool(embeddings),
            "vector_search": USE_EMBEDDINGS,
            "advanced_pdf_extraction": True,
            "intelligent_chunking": True,
            "documents": len(document_storage)
        }
    }

@app.get("/health")
async def health_check():
    # Get real counts from Supabase
    supabase_docs = 0
    supabase_embeddings = 0
    
    if supabase_client:
        try:
            docs_result = supabase_client.table('documents').select('doc_id').execute()
            supabase_docs = len(docs_result.data) if docs_result.data else 0
            
            embeddings_result = supabase_client.table('document_embeddings').select('id').execute()
            supabase_embeddings = len(embeddings_result.data) if embeddings_result.data else 0
        except:
            pass
    
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "embeddings_configured": bool(embeddings),
        "vector_search_enabled": USE_EMBEDDINGS,
        "documents_in_storage": len(document_storage),
        "documents_in_supabase": supabase_docs,
        "embeddings_in_supabase": supabase_embeddings,
        "storage_mode": "complete_rag" if USE_EMBEDDINGS else "local_only",
        "pdf_extraction": "advanced_pymupdf",
        "chunking_strategy": f"{CHUNK_SIZE}_tokens_with_{CHUNK_OVERLAP}_overlap"
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
        print(f"🚀 Processing {file.filename}...")
        
        # Read file content
        content = await file.read()
        
        # Extract text using advanced methods
        full_text, page_count = extract_text_advanced(content, file.filename)
        
        if len(full_text.strip()) < 50:
            raise HTTPException(400, "PDF contains insufficient text content")
        
        print(f"📄 Extracted {len(full_text)} characters from {page_count} pages")
        
        # Create document ID
        doc_id = file.filename.replace('.pdf', '').replace(' ', '_')
        
        # Chunk text intelligently
        chunks = chunk_text_intelligently(full_text)
        
        # Generate embeddings if enabled
        embeddings_success = False
        if USE_EMBEDDINGS:
            try:
                embedding_vectors = generate_embeddings(chunks)
                embeddings_success = store_in_supabase(
                    doc_id, file.filename, full_text, chunks, embedding_vectors, page_count
                )
                print(f"✅ Embeddings stored: {embeddings_success}")
            except Exception as e:
                print(f"⚠️ Embeddings failed: {e}")
        
        # Store in memory for immediate access
        document_storage[doc_id] = {
            "doc_id": doc_id,
            "filename": file.filename,
            "text": full_text,
            "total_pages": page_count,
            "total_characters": len(full_text),
            "uploaded_at": time.time(),
            "chunks_count": len(chunks),
            "embeddings_stored": embeddings_success
        }
        
        print(f"✅ Upload complete: {doc_id}")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": page_count,
            "total_characters": len(full_text),
            "chunks_created": len(chunks),
            "embeddings_enabled": USE_EMBEDDINGS,
            "embeddings_stored": embeddings_success,
            "extraction_method": "pymupdf_advanced",
            "message": f"Successfully processed {page_count} pages with {len(chunks)} chunks and {'✅' if embeddings_success else '❌'} embeddings"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    # Refresh from Supabase
    load_documents_from_storage()
    
    documents = []
    for doc_id, doc_data in document_storage.items():
        documents.append({
            "doc_id": doc_id,
            "filename": doc_data["filename"],
            "total_pages": doc_data.get("total_pages", 0),
            "total_characters": doc_data.get("total_characters", 0),
            "chunks_count": doc_data.get("chunks_count", 0),
            "embeddings_stored": doc_data.get("embeddings_stored", False),
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
        
        if USE_EMBEDDINGS:
            # Use vector search for the best context
            similar_chunks = search_similar_embeddings(q, doc_id, k=SIMILARITY_TOP_K)
            
            if similar_chunks:
                # Build context from similar chunks
                context_parts = []
                sources = []
                
                for chunk in similar_chunks:
                    context_parts.append(chunk['content'])
                    sources.append({
                        'chunk_id': chunk['metadata'].get('chunk_id', 0),
                        'similarity': chunk.get('similarity', 0),
                        'preview': chunk['content'][:150] + "..."
                    })
                
                context = "\n\n".join(context_parts)
                confidence = min(0.95, 0.7 + (len(similar_chunks) * 0.05))
                vector_search_used = True
                
            else:
                # Fallback to full document
                context = doc["text"][:8000]
                sources = [{"chunk_id": 0, "preview": "Full document fallback"}]
                confidence = 0.6
                vector_search_used = False
        else:
            # Traditional approach without vector search
            text = doc["text"]
            tokens = len(text) // 4  # Approximate
            if tokens > 3000:
                context = text[:6000] + "\n\n[...content truncated...]\n\n" + text[-2000:]
            else:
                context = text
            
            sources = [{"chunk_id": 0, "preview": "Traditional text search"}]
            confidence = 0.7
            vector_search_used = False
        
        # Generate answer using OpenAI
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on document content. Provide detailed, accurate answers and cite relevant information when possible."
            },
            {
                "role": "user", 
                "content": f"""Document content:\n{context}\n\nQuestion: {q}\n\nProvide a comprehensive answer based on the document content. If the information is not available in the provided context, please say so clearly."""
            }
        ]
        
        response_text = call_openai_api(messages)
        
        return {
            "answer": response_text,
            "doc_id": doc_id,
            "sources": sources,
            "confidence": confidence,
            "vector_search_used": vector_search_used,
            "context_length": len(context),
            "chunks_used": len(sources) if vector_search_used else 1
        }
        
    except Exception as e:
        print(f"❌ Question answering error: {e}")
        raise HTTPException(500, f"Failed to process question: {str(e)}")

@app.get("/extract")
async def extract_knowledge(doc_id: str):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OpenAI API key not configured")
    
    if doc_id not in document_storage:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    try:
        doc = document_storage[doc_id]
        text = doc["text"]
        
        # Use manageable text size for extraction
        if len(text) > 20000:
            # Use first 15k characters for extraction
            text_for_extraction = text[:15000]
        else:
            text_for_extraction = text
        
        async def generate_extraction():
            try:
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert knowledge extraction AI. Extract comprehensive structured information from documents. Return only valid JSON."
                    },
                    {
                        "role": "user",
                        "content": f"""Extract structured information from this document:\n\n{text_for_extraction}\n\nReturn JSON with:\n- quotes: important statements with context and page references\n- entities: people, organizations, places, products mentioned\n- metrics: numbers, dates, statistics with context\n- relations: relationships between entities\n\nFormat as valid JSON with proper structure."""
                    }
                ]
                
                response_text = call_openai_api(messages, temperature=0.0)
                
                # Parse and validate JSON
                try:
                    extraction_data = json.loads(response_text)
                except:
                    extraction_data = {
                        "quotes": [],
                        "entities": [],
                        "metrics": [],
                        "relations": [],
                        "error": "JSON parsing failed",
                        "raw_response": response_text[:1000]
                    }
                
                # Add metadata
                extraction_data["document_info"] = {
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "total_pages": doc.get("total_pages", 0),
                    "total_characters": doc.get("total_characters", 0),
                    "chunks_count": doc.get("chunks_count", 0),
                    "embeddings_enabled": USE_EMBEDDINGS
                }
                
                extraction_data["extraction_stats"] = {
                    "quotes_found": len(extraction_data.get("quotes", [])),
                    "entities_found": len(extraction_data.get("entities", [])),
                    "metrics_found": len(extraction_data.get("metrics", [])),
                    "relations_found": len(extraction_data.get("relations", []))
                }
                
                print(f"✅ Knowledge extraction completed for {doc_id}")
                
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
                extraction_data = {
                    "quotes": [],
                    "entities": [],
                    "metrics": [],
                    "relations": [],
                    "error": str(e),
                    "document_info": {"doc_id": doc_id}
                }
            
            yield json.dumps(extraction_data, ensure_ascii=False, indent=2)
        
        return StreamingResponse(generate_extraction(), media_type="application/json")
        
    except Exception as e:
        print(f"❌ Extract endpoint error: {e}")
        raise HTTPException(500, f"Extraction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Platform API v4.0 - COMPLETE IMPLEMENTATION")
    uvicorn.run(app, host="0.0.0.0", port=8000)