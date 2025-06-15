-- ============================================
-- ENABLE COMPLETE RLS CONFIGURATION
-- ============================================

-- 1. Enable RLS on tables if not already enabled
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_embeddings ENABLE ROW LEVEL SECURITY;

-- 2. Grant necessary permissions to roles
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;

-- 3. Create comprehensive policies for DOCUMENTS table
CREATE POLICY "documents_select_policy" ON documents
    FOR SELECT USING (true);

CREATE POLICY "documents_insert_policy" ON documents
    FOR INSERT WITH CHECK (true);

CREATE POLICY "documents_update_policy" ON documents
    FOR UPDATE USING (true) WITH CHECK (true);

CREATE POLICY "documents_delete_policy" ON documents
    FOR DELETE USING (true);

-- 4. Create comprehensive policies for DOCUMENT_EMBEDDINGS table
CREATE POLICY "embeddings_select_policy" ON document_embeddings
    FOR SELECT USING (true);

CREATE POLICY "embeddings_insert_policy" ON document_embeddings
    FOR INSERT WITH CHECK (true);

CREATE POLICY "embeddings_update_policy" ON document_embeddings
    FOR UPDATE USING (true) WITH CHECK (true);

CREATE POLICY "embeddings_delete_policy" ON document_embeddings
    FOR DELETE USING (true);

-- 5. Additional security settings
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO anon, authenticated;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO anon, authenticated;

-- 6. Enable realtime if needed (optional)
-- alter publication supabase_realtime add table documents;
-- alter publication supabase_realtime add table document_embeddings;

-- 7. Test queries to verify everything works
SELECT 'Testing documents table access...' as test;
SELECT COUNT(*) as document_count FROM documents;

SELECT 'Testing embeddings table access...' as test;
SELECT COUNT(*) as embeddings_count FROM document_embeddings;

-- 8. Show current policies (for verification)
SELECT 
    schemaname,
    tablename,
    policyname,
    permissive,
    roles,
    cmd,
    qual,
    with_check
FROM pg_policies 
WHERE tablename IN ('documents', 'document_embeddings')
ORDER BY tablename, policyname;