import os
import json
import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
import PyPDF2
from io import BytesIO
from openai import OpenAI
import tiktoken

load_dotenv()

app = FastAPI(title="RAG Platform API", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment check
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    print("✅ OpenAI configured")
else:
    openai_client = None
    print("❌ Missing OPENAI_API_KEY")

# Simple storage for demo (in production use database)
import pickle
import pathlib

STORAGE_FILE = "document_storage.pkl"

def load_storage():
    if pathlib.Path(STORAGE_FILE).exists():
        try:
            with open(STORAGE_FILE, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    return {}

def save_storage():
    try:
        with open(STORAGE_FILE, 'wb') as f:
            pickle.dump(document_storage, f)
    except Exception as e:
        print(f"Error saving storage: {e}")

document_storage = load_storage()

EXTRACTION_PROMPT = """You are an expert knowledge extraction AI. Your task is to transform this document into structured, interconnected knowledge components.

DOCUMENT TO ANALYZE:
{full_text}

Extract information following this EXACT schema:

=== 1. QUOTES ===
Extract meaningful quotes, statements, declarations, key findings, conclusions.
Schema for each quote:
{{
  "id": "q1" | "q2" | "q3"...,
  "text": "exact verbatim quote text",
  "author": "name" | "unknown" | "document",
  "context": "brief description of surrounding context",
  "page": number | null,
  "importance": "high" | "medium" | "low",
  "type": "statement" | "finding" | "conclusion" | "claim" | "definition",
  "entity_ids": ["e1", "e2"],
  "metric_ids": ["m1", "m2"],
  "keywords": ["word1", "word2"]
}}

=== 2. ENTITIES ===
Extract all significant entities: people, organizations, places, products, concepts, technologies.
Schema for each entity:
{{
  "id": "e1" | "e2" | "e3"...,
  "name": "full entity name",
  "type": "person" | "organization" | "place" | "product" | "concept" | "technology",
  "description": "concise but informative description",
  "importance": "high" | "medium" | "low",
  "first_mentioned_page": number | null,
  "aliases": ["alternative name 1", "acronym"],
  "quote_ids": ["q1", "q3"],
  "metric_ids": ["m1", "m4"],
  "related_entity_ids": ["e2", "e5"]
}}

=== 3. METRICS ===
Extract all quantitative data: numbers, percentages, dates, measurements, statistics, KPIs.
Schema for each metric:
{{
  "id": "m1" | "m2" | "m3"...,
  "value": "123.45" | "2023-01-15" | "Q4 2023",
  "unit": "%" | "$" | "years" | "units" | "people" | null,
  "type": "percentage" | "currency" | "date" | "quantity" | "ratio" | "timeframe",
  "context": "what this metric represents and why it matters",
  "category": "financial" | "performance" | "demographic" | "temporal" | "operational",
  "trend": "increasing" | "decreasing" | "stable" | null,
  "significance": "high" | "medium" | "low",
  "entity_ids": ["e1"],
  "quote_ids": ["q2"]
}}

=== 4. RELATIONS ===
Extract explicit relationships between entities.
Schema for each relation:
{{
  "source_entity_id": "e1",
  "target_entity_id": "e2",
  "type": "owns" | "works_for" | "competes_with" | "partners_with" | "located_in" | "created_by" | "influences",
  "description": "detailed description of the relationship",
  "strength": "strong" | "moderate" | "weak",
  "evidence_quote_ids": ["q1"]
}}

CRITICAL RULES:
1. Extract ALL significant quotes, entities, and metrics
2. Only extract what's explicitly stated
3. Maintain cross-references between quotes, entities, and metrics
4. Use sequential IDs (q1, q2... e1, e2... m1, m2...)

Return a valid JSON object with exactly these four top-level keys: "quotes", "entities", "metrics", "relations"
"""

@app.get("/")
async def root():
    return {
        "message": "RAG Platform API v2.0",
        "status": "running",
        "features": {
            "openai": bool(openai_client),
            "gemini": bool(GEMINI_API_KEY)
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "openai_configured": bool(openai_client),
        "gemini_configured": bool(GEMINI_API_KEY),
        "documents_in_storage": len(document_storage),
        "storage_file_exists": pathlib.Path(STORAGE_FILE).exists()
    }

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not openai_client:
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

@app.get("/ask")
async def ask_question(doc_id: str, q: str):
    if not openai_client:
        raise HTTPException(503, "OpenAI API key not configured")
    
    if doc_id not in document_storage:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    if not q.strip():
        raise HTTPException(400, "Question cannot be empty")
    
    try:
        doc = document_storage[doc_id]
        text = doc["text"]
        
        # For large documents, use key sections
        tokens = len(tiktoken.encoding_for_model("gpt-4").encode(text))
        if tokens > 8000:
            # Take first 4000 characters and last 4000 characters
            text = text[:4000] + "\\n\\n[...CONTENT OMITTED...]\\n\\n" + text[-4000:]
        
        response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on document content. Always cite page numbers when available."
                },
                {
                    "role": "user", 
                    "content": f"""Document content:\\n{text}\\n\\nQuestion: {q}\\n\\nProvide a detailed answer in English, citing page numbers when possible."""
                }
            ],
            temperature=0.1
        )
        
        return {
            "answer": response.choices[0].message.content,
            "doc_id": doc_id,
            "confidence": 0.85,
            "tokens_used": tokens
        }
        
    except Exception as e:
        print(f"Question answering error: {e}")
        raise HTTPException(500, f"Failed to process question: {str(e)}")

@app.get("/extract")
async def extract_knowledge(doc_id: str):
    if not openai_client:
        raise HTTPException(503, "OpenAI API key not configured")
    
    if doc_id not in document_storage:
        raise HTTPException(404, f"Document '{doc_id}' not found")
    
    try:
        doc = document_storage[doc_id]
        text = doc["text"]
        tokens = len(tiktoken.encoding_for_model("gpt-4").encode(text))
        
        print(f"🔍 Starting extraction for {doc_id}: {tokens} tokens")
        
        async def generate_extraction():
            start_time = time.time()
            
            try:
                # Choose strategy based on document size
                if tokens > 25000 and GEMINI_API_KEY:
                    print("🤖 Using Gemini for large document")
                    extraction_data = await extract_with_gemini(text)
                else:
                    print("🧠 Using GPT-4")
                    extraction_data = await extract_with_gpt4(text, tokens)
                
                # Add metadata
                extraction_data["document_info"] = {
                    "doc_id": doc_id,
                    "filename": doc["filename"],
                    "total_pages": doc["pages"],
                    "total_tokens": tokens,
                    "processing_time": f"{time.time() - start_time:.1f}s"
                }
                
                # Validate structure
                extraction_data = validate_extraction(extraction_data)
                
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

async def extract_with_gpt4(text: str, tokens: int) -> Dict:
    """Extract using GPT-4"""
    # Handle large documents by selecting key sections
    if tokens > 15000:
        # Take beginning, middle, end
        text_len = len(text)
        sections = [
            text[:5000],
            text[text_len//2-2500:text_len//2+2500],
            text[-5000:]
        ]
        text = "\\n\\n[...SECTION BREAK...]\\n\\n".join(sections)
    
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": "You are an expert knowledge extraction AI. Return only valid JSON, no markdown formatting."
            },
            {
                "role": "user",
                "content": EXTRACTION_PROMPT.format(full_text=text)
            }
        ],
        temperature=0
    )
    
    response_text = response.choices[0].message.content.strip()
    
    # Clean response
    if response_text.startswith("```json"):
        response_text = response_text[7:]
    if response_text.endswith("```"):
        response_text = response_text[:-3]
    
    try:
        data = json.loads(response_text)
        data["model"] = "gpt-4o"
        data["confidence"] = 0.85 if tokens < 15000 else 0.75
        return data
    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        return {
            "quotes": [],
            "entities": [],
            "metrics": [],
            "relations": [],
            "error": f"JSON parsing failed: {str(e)}",
            "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
        }

async def extract_with_gemini(text: str) -> Dict:
    """Extract using Gemini for large documents"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        model = genai.GenerativeModel('gemini-1.5-pro')
        prompt = EXTRACTION_PROMPT.format(full_text=text)
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Clean Gemini response
        if response_text.startswith("```json"):
            response_text = response_text[7:]
        if response_text.endswith("```"):
            response_text = response_text[:-3]
        
        data = json.loads(response_text)
        data["model"] = "gemini-1.5-pro"
        data["confidence"] = 0.95
        
        return data
        
    except Exception as e:
        print(f"Gemini extraction error: {e}")
        # Fallback to GPT-4
        return await extract_with_gpt4(text[:25000], 25000)

def validate_extraction(data: Dict) -> Dict:
    """Validate and enhance extracted data"""
    # Ensure required fields
    data.setdefault("quotes", [])
    data.setdefault("entities", [])
    data.setdefault("metrics", [])
    data.setdefault("relations", [])
    
    # Generate statistics
    data["extraction_stats"] = {
        "totals": {
            "quotes": len(data["quotes"]),
            "entities": len(data["entities"]),
            "metrics": len(data["metrics"]),
            "relations": len(data["relations"])
        },
        "cross_references": {
            "quotes_with_entities": len([q for q in data["quotes"] if q.get("entity_ids")]),
            "quotes_with_metrics": len([q for q in data["quotes"] if q.get("metric_ids")]),
            "entities_with_quotes": len([e for e in data["entities"] if e.get("quote_ids")])
        }
    }
    
    # Calculate quality score
    quality_score = 0
    if data["extraction_stats"]["totals"]["quotes"] > 0: quality_score += 0.25
    if data["extraction_stats"]["totals"]["entities"] > 0: quality_score += 0.25
    if data["extraction_stats"]["totals"]["metrics"] > 0: quality_score += 0.25
    if data["extraction_stats"]["cross_references"]["quotes_with_entities"] > 0: quality_score += 0.25
    
    data["extraction_quality"] = round(quality_score, 2)
    
    return data

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RAG Platform API v2.0")
    uvicorn.run(app, host="0.0.0.0", port=8000)