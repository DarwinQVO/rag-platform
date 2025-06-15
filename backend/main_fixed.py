import os
import json
import time
import requests
from typing import List, Dict, Any
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
import pickle
import pathlib

load_dotenv()

app = FastAPI(title="RAG Platform API", version="2.0")
# CORS configuration for production
origins = [
    "http://localhost:5173",  # Local development
    "http://localhost:3000",  # Alternative local port
    "https://rag-platform-three.vercel.app",  # Specific Vercel deployment
    "https://*.vercel.app",   # Other Vercel deployments
    "https://*.netlify.app",  # Netlify deployments
]

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

# Storage configuration - use Supabase if available
STORAGE_FILE = "document_storage.pkl"
USE_SUPABASE = os.getenv("ENVIRONMENT") == "production" and SUPABASE_URL and SUPABASE_KEY

# Initialize Supabase if available
supabase_client = None
if USE_SUPABASE:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✅ Supabase connected for persistent storage")
    except Exception as e:
        print(f"⚠️ Supabase connection failed: {e}")
        USE_SUPABASE = False

def load_storage():
    """Load documents from Supabase or local file"""
    if USE_SUPABASE and supabase_client:
        try:
            result = supabase_client.table('documents').select('*').execute()
            if hasattr(result, 'data') and result.data:
                print(f"✅ Loaded {len(result.data)} documents from Supabase")
                # Convert to the format expected by the app
                storage = {}
                for doc in result.data:
                    doc_id = str(doc.get('id'))
                    
                    # Get text from chunks if it exists
                    text = doc.get('content', '')
                    if not text:
                        try:
                            chunks_result = supabase_client.table('chunks').select('content').eq('document_id', doc.get('id')).execute()
                            if chunks_result.data:
                                text = ' '.join([chunk.get('content', '') for chunk in chunks_result.data])
                        except:
                            pass
                    
                    storage[doc_id] = {
                        'doc_id': doc_id,
                        'filename': doc.get('filename', 'Unknown'),
                        'text': text,
                        'total_pages': doc.get('pages', 0),
                        'total_characters': len(text),
                        'uploaded_at': doc.get('created_at', time.time())
                    }
                return storage
        except Exception as e:
            print(f"⚠️ Failed to load from Supabase: {e}")
    
    # Fallback to local file
    if pathlib.Path(STORAGE_FILE).exists():
        try:
            with open(STORAGE_FILE, 'rb') as f:
                storage = pickle.load(f)
                print(f"📁 Loaded {len(storage)} documents from local file")
                return storage
        except:
            pass
    return {}

def save_storage():
    """Save to both Supabase and local file"""
    try:
        # Always save to local file as backup
        with open(STORAGE_FILE, 'wb') as f:
            pickle.dump(document_storage, f)
        
        # Also save to Supabase if available
        if USE_SUPABASE and supabase_client:
            for doc_id, doc_data in document_storage.items():
                save_document_to_supabase(doc_data)
                
    except Exception as e:
        print(f"Error saving storage: {e}")

def save_document_to_supabase(doc_data):
    """Save individual document to Supabase"""
    if not (USE_SUPABASE and supabase_client):
        return False
    
    try:
        # Check if document already exists
        existing = supabase_client.table('documents').select('id').eq('filename', doc_data['filename']).execute()
        if existing.data:
            print(f"📄 Document {doc_data['filename']} already exists in Supabase")
            return True
        
        # Insert new document
        document_record = {
            'filename': doc_data['filename'],
            'pages': doc_data.get('total_pages', 0),
        }
        
        result = supabase_client.table('documents').insert(document_record).execute()
        if result.data:
            document_id = result.data[0]['id']
            
            # Save text as chunks
            text = doc_data.get('text', '')
            if text:
                chunk_size = 3000
                chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
                
                for i, chunk_text in enumerate(chunks):
                    chunk_record = {
                        'document_id': document_id,
                        'content': chunk_text,
                        'chunk_index': i
                    }
                    supabase_client.table('chunks').insert(chunk_record).execute()
            
            print(f"✅ Saved {doc_data['filename']} to Supabase with {len(chunks) if text else 0} chunks")
            return True
    except Exception as e:
        print(f"❌ Failed to save {doc_data.get('filename')} to Supabase: {e}")
    
    return False

document_storage = load_storage()

def count_tokens_simple(text: str) -> int:
    """Simple token counting - roughly 4 chars per token"""
    return len(text) // 4

def call_openai_api(messages: List[Dict], model: str = "gpt-4o", temperature: float = 0.1) -> str:
    """Direct OpenAI API call"""
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
    
    response = requests.post(url, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

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
    except Exception as e:
        print(f"Unexpected error: {e}")
        return {
            "quotes": [],
            "entities": [],
            "metrics": [],
            "relations": [],
            "error": f"Unexpected error: {str(e)}",
            "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
        }

EXTRACTION_PROMPT = """EXTRACT ALL EVENTS FROM THIS TEXT. Focus on temporal information and everything that happened.

{full_text}

Extract these 3 categories - prioritize EVENTS:

EVENTS - Extract ALL temporal occurrences:
• Actions that happened at specific times
• Milestones, achievements, announcements
• Decisions made, meetings held, agreements signed  
• Product launches, releases, publications
• Changes in status, appointments, departures
• Financial events (funding, acquisitions, earnings)
• Market events, crises, opportunities
• Any occurrence with temporal significance

ENTITIES - Extract ALL people, organizations, products mentioned:
• People (names, titles, roles)
• Organizations (companies, institutions, agencies)
• Places (countries, cities, locations)
• Products (software, tools, systems, brands)
• Concepts (methodologies, frameworks, technologies)

METRICS - Extract ALL numbers and measurements:
• Dates and timeframes (most important for events)
• Numbers with meaning (percentages, amounts, counts)
• Statistics and measurements
• Financial figures, performance indicators

JSON FORMAT:
{{
  "events": [
    {{
      "id": "ev1",
      "title": "brief event title",
      "description": "detailed description of what happened",
      "temporal_marker": "2023-Q4" | "January 2023" | "last year" | "recently" | "before the merger",
      "date_parsed": "2023-01-15" | null,
      "certainty": "certain" | "estimate",
      "type": "milestone" | "announcement" | "decision" | "launch" | "meeting" | "change" | "financial" | "market",
      "entity_ids": ["e1", "e2"],
      "metric_ids": ["m1"],
      "supporting_text": "exact text from PDF that supports this event",
      "page_number": 5,
      "importance": "high" | "medium" | "low"
    }}
  ],
  "entities": [
    {{
      "id": "e1", 
      "name": "entity name",
      "type": "person" | "organization" | "place" | "product" | "concept",
      "description": "what/who they are",
      "event_ids": ["ev1", "ev2"],
      "metric_ids": ["m1"]
    }}
  ],
  "metrics": [
    {{
      "id": "m1",
      "value": "123",
      "unit": "%" | "$" | "years" | "Q4" | etc,
      "type": "date" | "percentage" | "currency" | "quantity",
      "context": "what it measures",
      "event_ids": ["ev1"],
      "entity_ids": ["e1"]
    }}
  ]
}}

CRITICAL RULES FOR EVENTS:
✓ Every event MUST have supporting_text (exact quote from PDF)
✓ Every event MUST have page_number where it was found
✓ temporal_marker: use the EXACT temporal reference from the text
✓ date_parsed: only if you can determine a specific date (YYYY-MM-DD format)
✓ certainty: "certain" if explicitly stated, "estimate" if implied/approximate
✓ Look for: "in 2023", "last quarter", "recently", "after the acquisition", "before launch"
✓ Extract ALL events - even small ones might be important
✓ Cross-reference: link events to entities and metrics

GOAL: Create a complete timeline of everything that happened, with full traceability back to the source text."""

@app.get("/")
async def root():
    return {
        "message": "RAG Platform API v2.0 - FIXED",
        "status": "running",
        "features": {
            "openai": bool(OPENAI_API_KEY),
            "gemini": bool(GEMINI_API_KEY),
            "documents": len(document_storage)
        }
    }

@app.get("/health")
async def health_check():
    # Get document count from Supabase if available
    supabase_docs = 0
    if USE_SUPABASE and supabase_client:
        try:
            result = supabase_client.table('documents').select('id').execute()
            supabase_docs = len(result.data) if result.data else 0
        except:
            pass
    
    return {
        "status": "healthy",
        "openai_configured": bool(OPENAI_API_KEY),
        "gemini_configured": bool(GEMINI_API_KEY),
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "supabase_connected": bool(USE_SUPABASE and supabase_client),
        "documents_in_storage": len(document_storage),
        "documents_in_supabase": supabase_docs,
        "storage_file_exists": pathlib.Path(STORAGE_FILE).exists(),
        "storage_mode": "supabase" if USE_SUPABASE else "local"
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not OPENAI_API_KEY:
        raise HTTPException(503, "OpenAI API key not configured")
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(400, "File must be a PDF")
    
    if file.size > 50 * 1024 * 1024:  # 50MB
        raise HTTPException(400, "File too large (max 50MB)")
    
    try:
        content = await file.read()
        pdf_reader = PyPDF2.PdfReader(BytesIO(content))
        
        if len(pdf_reader.pages) > 1000:
            raise HTTPException(400, "PDF too long (max 1000 pages)")
        
        # Extract text with page tracking
        text_parts = []
        for i, page in enumerate(pdf_reader.pages):
            try:
                page_text = page.extract_text()
                if page_text.strip():
                    text_parts.append(f"[PAGE {i+1}]\\n{page_text}")
            except Exception as e:
                print(f"Warning: Could not extract text from page {i+1}: {e}")
                continue
        
        if not text_parts:
            raise HTTPException(400, "Could not extract any text from PDF")
        
        full_text = "\\n\\n".join(text_parts)
        doc_id = file.filename.replace('.pdf', '').replace(' ', '_')
        
        # Store document
        document_storage[doc_id] = {
            "filename": file.filename,
            "text": full_text,
            "pages": len(pdf_reader.pages),
            "uploaded_at": time.time()
        }
        
        # Save to disk
        save_storage()
        print(f"📄 Saved document: {doc_id}")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file.filename,
            "total_pages": len(pdf_reader.pages),
            "total_characters": len(full_text),
            "message": f"Successfully processed {len(pdf_reader.pages)} pages"
        }
        
    except PyPDF2.errors.PdfReadError:
        raise HTTPException(400, "Invalid or corrupted PDF file")
    except Exception as e:
        print(f"Upload error: {e}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    documents = []
    for doc_id, doc_data in document_storage.items():
        documents.append({
            "doc_id": doc_id,
            "filename": doc_data["filename"],
            "total_pages": doc_data["pages"],
            "total_characters": len(doc_data["text"]),
            "uploaded_at": doc_data["uploaded_at"]
        })
    
    return {"documents": documents, "total": len(documents)}

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document from storage"""
    if doc_id not in document_storage:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    try:
        # Get document info before deletion
        doc_info = document_storage[doc_id]
        filename = doc_info["filename"]
        
        # Remove from storage
        del document_storage[doc_id]
        
        # Save updated storage
        save_storage()
        
        print(f"🗑️  Deleted document: {doc_id}")
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": filename,
            "message": f"Document '{filename}' deleted successfully"
        }
        
    except Exception as e:
        print(f"Delete error: {e}")
        raise HTTPException(500, f"Failed to delete document: {str(e)}")

@app.get("/debug/{doc_id}")
async def debug_document(doc_id: str):
    """Debug endpoint to see document content"""
    if doc_id not in document_storage:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    doc = document_storage[doc_id]
    return {
        "doc_id": doc_id,
        "filename": doc["filename"],
        "pages": doc["pages"],
        "text_length": len(doc["text"]),
        "text_preview": doc["text"][:500] + "..." if len(doc["text"]) > 500 else doc["text"],
        "tokens_estimate": count_tokens_simple(doc["text"])
    }

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
        
        # For large documents, use key sections
        tokens = count_tokens_simple(text)
        if tokens > 2000:
            # Take first 3000 characters and last 3000 characters
            text = text[:3000] + "\\n\\n[...CONTENT OMITTED...]\\n\\n" + text[-3000:]
        
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on document content. Always cite page numbers when available."
            },
            {
                "role": "user", 
                "content": f"""Document content:\\n{text}\\n\\nQuestion: {q}\\n\\nProvide a detailed answer in English, citing page numbers when possible."""
            }
        ]
        
        response_text = call_openai_api(messages)
        
        return {
            "answer": response_text,
            "doc_id": doc_id,
            "confidence": 0.85,
            "tokens_used": tokens
        }
        
    except Exception as e:
        print(f"Question answering error: {e}")
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
        tokens = count_tokens_simple(text)
        
        print(f"🔍 Starting extraction for {doc_id}: {tokens} tokens")
        
        async def generate_extraction():
            start_time = time.time()
            
            try:
                # Handle large documents by selecting key sections
                if tokens > 3000:
                    # Take beginning, middle, end
                    text_len = len(text)
                    sections = [
                        text[:2000],
                        text[text_len//2-1000:text_len//2+1000],
                        text[-2000:]
                    ]
                    processed_text = "\\n\\n[...SECTION BREAK...]\\n\\n".join(sections)
                else:
                    processed_text = text
                
                messages = [
                    {
                        "role": "system",
                        "content": "You are an expert knowledge extraction AI specialized in comprehensive information extraction. Your goal is to extract EVERY significant piece of information from the text. Be thorough and exhaustive - don't miss anything important. Extract obvious information AND subtle details. Return only valid JSON with no markdown formatting."
                    },
                    {
                        "role": "user",
                        "content": EXTRACTION_PROMPT.format(full_text=processed_text)
                    }
                ]
                
                response_text = call_openai_api(messages)
                
                # Clean and parse response
                extraction_data = clean_and_parse_json(response_text)
                extraction_data["model"] = "gpt-4o"
                extraction_data["confidence"] = 0.85 if tokens < 3000 else 0.75
                
                # Add metadata
                extraction_data["document_info"] = {
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "total_pages": doc["pages"],
                    "total_tokens": tokens,
                    "processing_time": f"{time.time() - start_time:.1f}s"
                }
                
                # Generate statistics
                extraction_data["extraction_stats"] = {
                    "totals": {
                        "events": len(extraction_data.get("events", [])),
                        "entities": len(extraction_data.get("entities", [])),
                        "metrics": len(extraction_data.get("metrics", []))
                    }
                }
                
                quality_score = 0
                if extraction_data["extraction_stats"]["totals"]["events"] > 0: quality_score += 0.4
                if extraction_data["extraction_stats"]["totals"]["entities"] > 0: quality_score += 0.3
                if extraction_data["extraction_stats"]["totals"]["metrics"] > 0: quality_score += 0.3
                
                extraction_data["extraction_quality"] = round(quality_score, 2)
                
                print(f"✅ Extraction completed for {doc_id}")
                
            except Exception as e:
                print(f"❌ Extraction failed: {e}")
                extraction_data = {
                    "events": [],
                    "entities": [],
                    "metrics": [],
                    "error": str(e),
                    "document_info": {"doc_id": doc_id, "total_tokens": tokens}
                }
            
            yield json.dumps(extraction_data, ensure_ascii=False, indent=2)
        
        return StreamingResponse(generate_extraction(), media_type="application/json")
        
    except Exception as e:
        print(f"Extract endpoint error: {e}")
        raise HTTPException(500, f"Extraction failed: {str(e)}")

@app.get("/events/{doc_id}")
async def get_events_by_entity(doc_id: str, entity_name: str = None):
    """Get all events for a document, optionally filtered by entity"""
    if doc_id not in document_storage:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    # First check if we have extracted data for this document
    # For now, we'll re-extract. In production, you'd cache this.
    try:
        doc = document_storage[doc_id]
        text = doc["text"]
        tokens = count_tokens_simple(text)
        
        # Handle large documents by selecting key sections
        if tokens > 3000:
            text_len = len(text)
            sections = [
                text[:2000],
                text[text_len//2-1000:text_len//2+1000],
                text[-2000:]
            ]
            processed_text = "\\n\\n[...SECTION BREAK...]\\n\\n".join(sections)
        else:
            processed_text = text
        
        messages = [
            {
                "role": "system",
                "content": "You are an expert event extraction AI. Focus on extracting ALL temporal events and their relationships to entities. Return only valid JSON."
            },
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(full_text=processed_text)
            }
        ]
        
        response_text = call_openai_api(messages)
        extraction_data = clean_and_parse_json(response_text)
        
        # Extract events and related data
        events = extraction_data.get("events", [])
        entities = extraction_data.get("entities", [])
        metrics = extraction_data.get("metrics", [])
        
        # Filter events by entity if requested
        if entity_name:
            # Find entity ID
            target_entity_ids = []
            for entity in entities:
                if entity_name.lower() in entity.get("name", "").lower():
                    target_entity_ids.append(entity["id"])
            
            # Filter events that mention this entity
            filtered_events = []
            for event in events:
                event_entity_ids = event.get("entity_ids", [])
                if any(eid in target_entity_ids for eid in event_entity_ids):
                    filtered_events.append(event)
            
            events = filtered_events
        
        # Sort events by temporal information (try to parse dates)
        def sort_key(event):
            # Simple temporal sorting - in production you'd use proper date parsing
            temporal = event.get("temporal_marker", "")
            if "2024" in temporal: return "2024"
            if "2023" in temporal: return "2023"
            if "2022" in temporal: return "2022"
            return temporal
        
        events.sort(key=sort_key, reverse=True)
        
        return {
            "doc_id": doc_id,
            "entity_filter": entity_name,
            "total_events": len(events),
            "events": events,
            "related_entities": [e for e in entities if not entity_name or entity_name.lower() in e.get("name", "").lower()],
            "related_metrics": metrics
        }
        
    except Exception as e:
        print(f"Events extraction error: {e}")
        raise HTTPException(500, f"Failed to extract events: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Platform API v2.0 - FIXED VERSION")
    uvicorn.run(app, host="0.0.0.0", port=8000)