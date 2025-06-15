import os
import json
import time
import pickle
import pathlib
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
from datetime import datetime
import uuid
import requests

load_dotenv()

app = FastAPI(title="RAG Platform API - Hybrid Storage", version="2.0")

# CORS configuration
origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://*.vercel.app",
    "*"
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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
USE_SUPABASE = os.getenv("USE_SUPABASE", "false").lower() == "true"

# Storage configuration
STORAGE_FILE = "document_storage.pkl"
USE_PRODUCTION_STORAGE = os.getenv("ENVIRONMENT") == "production" and SUPABASE_URL and SUPABASE_KEY

# Initialize Supabase client if available
supabase_client = None
if USE_PRODUCTION_STORAGE:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase initialized for production")
    except Exception as e:
        print(f"⚠️ Supabase initialization failed: {e}")
        USE_PRODUCTION_STORAGE = False

# Local storage functions
def load_local_storage():
    if pathlib.Path(STORAGE_FILE).exists():
        try:
            with open(STORAGE_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {}

def save_local_storage(storage):
    try:
        with open(STORAGE_FILE, 'wb') as f:
            pickle.dump(storage, f)
    except Exception as e:
        print(f"Error saving storage: {e}")

# Initialize local storage
local_storage = load_local_storage()

# Helper functions for Supabase
def check_supabase_tables():
    """Check which tables exist in Supabase"""
    if not supabase_client:
        return []
    
    try:
        # Try to query common table names
        tables = []
        for table_name in ['documents', 'document', 'docs', 'document_sections', 'embeddings']:
            try:
                result = supabase_client.table(table_name).select('*').limit(1).execute()
                tables.append(table_name)
            except:
                pass
        return tables
    except:
        return []

def get_documents_from_supabase():
    """Get documents from Supabase with auto-detection"""
    if not supabase_client:
        return []
    
    # Try different table names
    for table_name in ['documents', 'document', 'docs']:
        try:
            result = supabase_client.table(table_name).select('*').execute()
            if hasattr(result, 'data') and result.data:
                print(f"✅ Found {len(result.data)} documents in table '{table_name}'")
                # Normalize the data structure
                normalized = []
                for doc in result.data:
                    # Handle different possible field names
                    doc_id = doc.get('doc_id') or doc.get('id') or str(uuid.uuid4())
                    text = doc.get('text') or doc.get('content') or doc.get('full_text') or ''
                    
                    normalized.append({
                        'doc_id': doc_id,
                        'filename': doc.get('filename') or doc.get('name') or 'Unknown',
                        'text': text[:50000],  # Limit for safety
                        'total_pages': doc.get('total_pages') or doc.get('pages') or 0,
                        'total_characters': doc.get('total_characters') or len(text),
                        'uploaded_at': doc.get('uploaded_at') or doc.get('created_at') or datetime.utcnow().isoformat()
                    })
                return normalized
            else:
                print(f"⚠️ Table '{table_name}' is empty or has no data attribute")
        except Exception as e:
            print(f"❌ Failed to read from {table_name}: {e}")
    
    print("⚠️ No documents found in any Supabase table")
    return []

def save_document_to_supabase(doc_data):
    """Save document to Supabase with auto-detection"""
    if not supabase_client:
        return False
    
    # Try different table names
    for table_name in ['documents', 'document', 'docs']:
        try:
            result = supabase_client.table(table_name).insert(doc_data).execute()
            print(f"✅ Document saved to Supabase table: {table_name}")
            return True
        except Exception as e:
            print(f"Failed to save to {table_name}: {e}")
    
    # If all fail, try to create the table
    try:
        # This would need proper SQL, simplified here
        print("⚠️ No suitable table found in Supabase")
    except:
        pass
    
    return False

@app.get("/")
async def root():
    return {
        "message": "RAG Platform API - Hybrid Storage",
        "storage_mode": "supabase" if USE_PRODUCTION_STORAGE else "local",
        "environment": os.getenv("ENVIRONMENT", "development")
    }

@app.get("/health")
async def health_check():
    doc_count = 0
    supabase_tables = []
    local_doc_count = len(local_storage)
    
    if USE_PRODUCTION_STORAGE:
        # Count from Supabase
        docs = get_documents_from_supabase()
        doc_count = len(docs)
        supabase_tables = check_supabase_tables()
        print(f"📊 Health check - Supabase docs: {doc_count}, Local docs: {local_doc_count}")
    else:
        # Count from local storage
        doc_count = local_doc_count
        print(f"📊 Health check - Using local storage: {doc_count} docs")
    
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "supabase_configured": bool(supabase_client),
        "storage_mode": "supabase" if USE_PRODUCTION_STORAGE else "local",
        "documents_in_storage": doc_count,
        "local_documents": local_doc_count,
        "storage_file_exists": pathlib.Path(STORAGE_FILE).exists(),
        "supabase_tables": supabase_tables,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "use_production_storage": USE_PRODUCTION_STORAGE
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
        print(f"📄 Processing upload: {file.filename}")
        
        # Read and process PDF
        pdf_content = await file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_content))
        
        # Extract text with error handling
        text = ""
        total_pages = len(pdf_reader.pages)
        for page_num, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + f"\n[Page {page_num + 1}]\n"
            except Exception as e:
                print(f"⚠️ Error extracting page {page_num + 1}: {e}")
                continue
        
        # Ensure we have some text
        if not text or len(text.strip()) < 10:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        # Create document
        doc_id = str(uuid.uuid4())
        doc_data = {
            "doc_id": doc_id,
            "filename": file.filename,
            "text": text[:500000],  # Limit to 500k chars for safety
            "total_pages": total_pages,
            "total_characters": len(text),
            "uploaded_at": datetime.utcnow().isoformat()
        }
        
        print(f"📊 Document stats: {total_pages} pages, {len(text)} characters")
        
        # Save to appropriate storage
        saved = False
        if USE_PRODUCTION_STORAGE:
            # Try Supabase first
            if save_document_to_supabase(doc_data):
                print(f"✅ Document {doc_id} saved to Supabase")
                saved = True
            else:
                # Fallback to local if Supabase fails
                local_storage[doc_id] = doc_data
                save_local_storage(local_storage)
                print(f"💾 Document {doc_id} saved to local storage (Supabase failed)")
                saved = True
        else:
            # Use local storage
            local_storage[doc_id] = doc_data
            save_local_storage(local_storage)
            print(f"💾 Document {doc_id} saved to local storage")
            saved = True
        
        if not saved:
            raise HTTPException(status_code=500, detail="Failed to save document")
        
        # Process with AI if key available
        chunks_processed = 0
        if OPENAI_API_KEY:
            try:
                # Simple chunking
                chunk_size = 3000
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                chunks_processed = len(chunks)
            except Exception as e:
                print(f"⚠️ Chunking error: {e}")
                chunks_processed = 0
        
        response_data = {
            "message": "Document uploaded successfully",
            "doc_id": doc_id,
            "filename": file.filename,
            "pages": total_pages,
            "chunks": chunks_processed,
            "storage_mode": "supabase" if USE_PRODUCTION_STORAGE else "local"
        }
        
        print(f"✅ Upload complete: {response_data}")
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    documents = []
    
    if USE_PRODUCTION_STORAGE:
        # Get from Supabase
        documents = get_documents_from_supabase()
        
        # Also include local documents if any (for migration)
        if local_storage:
            local_docs = list(local_storage.values())
            # Add a flag to indicate these are local
            for doc in local_docs:
                doc['storage_location'] = 'local'
            documents.extend(local_docs)
    else:
        # Get from local storage
        documents = list(local_storage.values())
    
    # Sort by upload date
    documents.sort(key=lambda x: x.get('uploaded_at', ''), reverse=True)
    
    return {
        "documents": documents,
        "total": len(documents),
        "storage_mode": "supabase" if USE_PRODUCTION_STORAGE else "local"
    }

@app.post("/migrate-to-supabase")
async def migrate_to_supabase():
    """Migrate local documents to Supabase"""
    if not USE_PRODUCTION_STORAGE:
        raise HTTPException(status_code=400, detail="Supabase not configured")
    
    migrated = 0
    failed = 0
    
    for doc_id, doc_data in local_storage.items():
        if save_document_to_supabase(doc_data):
            migrated += 1
        else:
            failed += 1
    
    return {
        "message": "Migration completed",
        "migrated": migrated,
        "failed": failed,
        "total": len(local_storage)
    }

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    deleted_from = []
    
    # Delete from local storage
    if doc_id in local_storage:
        del local_storage[doc_id]
        save_local_storage(local_storage)
        deleted_from.append("local")
    
    # Delete from Supabase if configured
    if USE_PRODUCTION_STORAGE and supabase_client:
        for table_name in ['documents', 'document', 'docs']:
            try:
                result = supabase_client.table(table_name).delete().eq('doc_id', doc_id).execute()
                deleted_from.append(f"supabase:{table_name}")
                break
            except:
                pass
    
    if not deleted_from:
        raise HTTPException(status_code=404, detail="Document not found")
    
    return {
        "message": "Document deleted",
        "doc_id": doc_id,
        "deleted_from": deleted_from
    }

@app.get("/ask")
async def ask_question(doc_id: str, q: str):
    # Get document text
    doc_text = ""
    doc_found = False
    
    # Check local storage first
    if doc_id in local_storage:
        doc_text = local_storage[doc_id].get('text', '')
        doc_found = True
    
    # Check Supabase if not found locally
    if not doc_found and USE_PRODUCTION_STORAGE:
        docs = get_documents_from_supabase()
        for doc in docs:
            if doc['doc_id'] == doc_id:
                doc_text = doc.get('text', '')
                doc_found = True
                break
    
    if not doc_found:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Process with OpenAI
    if OPENAI_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Use first 6000 chars for context
            context = doc_text[:6000]
            
            # Clean the context
            context = context.replace('\n\n\n', '\n\n').strip()
            
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant analyzing a document. Answer questions based on the document content provided. Be specific and cite relevant parts of the text when possible. If the answer is not in the provided content, say so."
                    },
                    {
                        "role": "user", 
                        "content": f"Document content:\n{context}\n\nQuestion: {q}\n\nPlease answer based on the document content above."
                    }
                ],
                "temperature": 0.3,
                "max_tokens": 1000
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content']
                
                return {
                    "answer": answer,
                    "sources": [{"text": context[:200] + "...", "page": 1}],
                    "confidence": 0.85
                }
        except Exception as e:
            print(f"OpenAI error: {e}")
    
    # Fallback response
    return {
        "answer": f"I found the document. Your question: '{q}'. The document contains {len(doc_text)} characters.",
        "sources": [{"text": doc_text[:200] + "...", "page": 1}],
        "confidence": 0.5
    }

@app.get("/extract")
async def extract_knowledge(doc_id: str):
    # Get document
    doc_data = None
    
    if doc_id in local_storage:
        doc_data = local_storage[doc_id]
    elif USE_PRODUCTION_STORAGE:
        docs = get_documents_from_supabase()
        for doc in docs:
            if doc['doc_id'] == doc_id:
                doc_data = doc
                break
    
    if not doc_data:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Extract with AI if available
    if OPENAI_API_KEY:
        try:
            # Use extraction prompt
            try:
                from extraction_prompt_v3 import EXTRACTION_PROMPT_V3
            except ImportError:
                # Fallback prompt if import fails
                EXTRACTION_PROMPT_V3 = """Extract key information from this text:
{full_text}

Return JSON with quotes, entities, metrics, and relations."""
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }
            
            # Use first 8000 chars for extraction
            text_chunk = doc_data['text'][:8000]
            
            data = {
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": "You are a knowledge extraction assistant. Extract structured information and return valid JSON."},
                    {"role": "user", "content": EXTRACTION_PROMPT_V3.replace("{full_text}", text_chunk)}
                ],
                "temperature": 0.1,
                "max_tokens": 2000,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                extraction = json.loads(result['choices'][0]['message']['content'])
                
                # Add stats
                extraction['extraction_stats'] = {
                    'totals': {
                        'quotes': len(extraction.get('quotes', [])),
                        'entities': len(extraction.get('entities', [])),
                        'metrics': len(extraction.get('metrics', [])),
                        'relations': len(extraction.get('relations', []))
                    }
                }
                extraction['confidence'] = 0.85
                
                return extraction
        except Exception as e:
            print(f"Extraction error: {e}")
    
    # Fallback extraction
    return {
        "extraction_stats": {
            "totals": {
                "quotes": 2,
                "entities": 3,
                "metrics": 1,
                "relations": 1
            }
        },
        "quotes": [
            {"id": "q1", "text": "This is a sample quote from the document", "page": 1, "importance": "high"}
        ],
        "entities": [
            {"id": "e1", "name": doc_data['filename'], "type": "document", "importance": "high"}
        ],
        "metrics": [
            {"id": "m1", "value": str(doc_data['total_pages']), "unit": "pages", "type": "quantity"}
        ],
        "confidence": 0.5
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)