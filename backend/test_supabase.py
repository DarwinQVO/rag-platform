import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Test fetching documents
    print("Testing Supabase connection...")
    
    try:
        # Try different table names
        for table_name in ['documents', 'document', 'docs']:
            print(f"\nTrying table: {table_name}")
            try:
                result = client.table(table_name).select('*').limit(5).execute()
                if result.data:
                    print(f"✅ Found {len(result.data)} documents in table '{table_name}'")
                    for doc in result.data[:2]:  # Show first 2
                        print(f"  - {doc.get('filename', 'Unknown')} (ID: {doc.get('doc_id', doc.get('id', 'N/A'))})")
                else:
                    print(f"❌ Table '{table_name}' exists but is empty")
            except Exception as e:
                print(f"❌ Table '{table_name}' error: {str(e)}")
                
    except Exception as e:
        print(f"Error: {e}")
else:
    print("Supabase credentials not found in environment")