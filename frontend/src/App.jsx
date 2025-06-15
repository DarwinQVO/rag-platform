import { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from './components/ui/card'
import { Button } from './components/ui/button'
import { Upload, FileText, Send, Bot, Loader2, Settings, Eye, EyeOff, MessageSquare, Database, BarChart3, Brain, Target, Quote, Users, TrendingUp, Network, CheckCircle2, ChevronRight, Calendar, Hash, Globe, Building2, Home, Menu, X, Bell, Search, Plus, Filter, MoreHorizontal, ArrowRight, Activity, Layers, Trash2, AlertTriangle, Clock, MapPin, Zap as EventIcon, Filter as FilterIcon } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function App() {
  const [docs, setDocs] = useState([])
  const [selectedDoc, setSelectedDoc] = useState(null)
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [loading, setLoading] = useState(false)
  const [extractedData, setExtractedData] = useState(null)
  const [uploadProgress, setUploadProgress] = useState(0)
  const [showRawJson, setShowRawJson] = useState(false)
  const [activeTab, setActiveTab] = useState('documents')
  const [showPromptEditor, setShowPromptEditor] = useState(false)
  const [prompts, setPrompts] = useState({
    extraction: '',
    chat: ''
  })
  const [systemHealth, setSystemHealth] = useState(null)
  const [showDeleteModal, setShowDeleteModal] = useState(false)
  const [docToDelete, setDocToDelete] = useState(null)
  const [deletingDoc, setDeletingDoc] = useState(false)
  const [eventsData, setEventsData] = useState(null)
  const [entityFilter, setEntityFilter] = useState('')
  const [loadingEvents, setLoadingEvents] = useState(false)
  const fileInputRef = useRef(null)

  useEffect(() => {
    fetchDocs()
    fetchSystemHealth()
    loadPrompts()
  }, [])

  const fetchSystemHealth = async () => {
    try {
      const res = await fetch(`${API_URL}/health`)
      const data = await res.json()
      setSystemHealth(data)
    } catch (error) {
      console.error('Error fetching health:', error)
    }
  }

  const loadPrompts = () => {
    const savedPrompts = localStorage.getItem('rag-prompts')
    if (savedPrompts) {
      setPrompts(JSON.parse(savedPrompts))
    } else {
      // Default prompts
      setPrompts({
        extraction: `EXTRACT ALL KEY INFORMATION FROM THIS TEXT. Miss nothing important.

{full_text}

Return JSON with these 4 categories. Be EXHAUSTIVE:

QUOTES - Extract ALL:
• Important statements, claims, conclusions
• Key insights, findings, recommendations  
• Direct quotes from people
• Significant definitions or explanations
• Critical facts or assertions

ENTITIES - Extract ALL:
• People (names, titles, roles)
• Organizations (companies, institutions, agencies)
• Places (countries, cities, locations)
• Products (software, tools, systems, brands)
• Concepts (methodologies, frameworks, technologies)

METRICS - Extract ALL:
• Numbers with meaning (percentages, amounts, counts)
• Dates and timeframes
• Statistics and measurements
• Financial figures
• Performance indicators
• Growth rates, ratios, comparisons

RELATIONS - Extract relationships between entities

JSON FORMAT:
{
  "quotes": [{"id":"q1", "text":"exact text", "author":"who said it", "context":"brief context", "page":1, "importance":"high", "entity_ids":["e1"], "metric_ids":["m1"]}],
  "entities": [{"id":"e1", "name":"entity name", "type":"person", "description":"what/who they are", "importance":"high", "quote_ids":["q1"], "metric_ids":["m1"]}],
  "metrics": [{"id":"m1", "value":"123", "unit":"%", "type":"percentage", "context":"what it measures", "significance":"high", "entity_ids":["e1"], "quote_ids":["q1"]}],
  "relations": [{"source_entity_id":"e1", "target_entity_id":"e2", "type":"works_for", "description":"relationship", "strength":"strong"}]
}

EXTRACTION RULES:
✓ Extract EVERYTHING of value - don't skip anything
✓ Include obvious AND subtle information  
✓ Cross-reference: link quotes↔entities↔metrics
✓ Use IDs: q1,q2... e1,e2... m1,m2...
✓ Return only valid JSON - no text before/after
✓ Be comprehensive - this chunk might contain critical info

GOAL: Extract ALL significant information so nothing important is lost.`,
        chat: `You are a helpful assistant that answers questions based on document content. Always cite page numbers when available.

Document content:
{document_content}

Question: {question}

Provide a detailed answer in English, citing page numbers when possible.`
      })
    }
  }

  const savePrompts = () => {
    localStorage.setItem('rag-prompts', JSON.stringify(prompts))
    alert('Prompts saved successfully!')
  }

  const fetchDocs = async () => {
    try {
      const res = await fetch(`${API_URL}/documents`)
      const data = await res.json()
      setDocs(data.documents || data)
    } catch (error) {
      console.error('Error fetching docs:', error)
    }
  }

  const handleFileUpload = async (e) => {
    const file = e.target.files[0]
    if (!file || !file.name.endsWith('.pdf')) return

    const formData = new FormData()
    formData.append('file', file)

    setLoading(true)
    setUploadProgress(30)

    try {
      const res = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData,
      })
      
      setUploadProgress(70)
      
      // Check if response is ok before parsing JSON
      if (!res.ok) {
        const errorText = await res.text()
        console.error('Upload error:', errorText)
        throw new Error(`Error ${res.status}: ${errorText || 'Failed to upload'}`)
      }
      
      // Try to parse JSON response
      let data
      try {
        data = await res.json()
      } catch (jsonError) {
        console.error('JSON parse error:', jsonError)
        throw new Error('Invalid response from server')
      }
      
      setUploadProgress(100)
      setTimeout(() => {
        alert(`¡Éxito! Documento procesado: ${data.pages} páginas, ${data.chunks} fragmentos`)
        fetchDocs()
        setUploadProgress(0)
      }, 500)
    } catch (error) {
      alert(`Error: ${error.message}`)
      setUploadProgress(0)
    } finally {
      setLoading(false)
      e.target.value = ''
    }
  }

  const handleAsk = async () => {
    if (!selectedDoc || !question.trim()) return

    const userMsg = { role: 'user', content: question }
    setMessages(prev => [...prev, userMsg])
    setQuestion('')
    setLoading(true)

    try {
      const res = await fetch(`${API_URL}/ask?doc_id=${selectedDoc.doc_id}&q=${encodeURIComponent(question)}`)
      const data = await res.json()
      
      const botMsg = {
        role: 'assistant',
        content: data.answer,
        sources: data.sources,
        confidence: data.confidence
      }
      setMessages(prev => [...prev, botMsg])
    } catch (error) {
      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: 'Error al procesar la pregunta' 
      }])
    } finally {
      setLoading(false)
    }
  }

  const handleExtract = async () => {
    if (!selectedDoc) return
    setLoading(true)
    setExtractedData(null)

    try {
      const res = await fetch(`${API_URL}/extract?doc_id=${selectedDoc.doc_id}`)
      const data = await res.json()
      setExtractedData(data)
    } catch (error) {
      alert('Error al extraer información')
    } finally {
      setLoading(false)
    }
  }

  const handleDeleteClick = (doc) => {
    setDocToDelete(doc)
    setShowDeleteModal(true)
  }

  const handleDeleteConfirm = async () => {
    if (!docToDelete) return
    
    setDeletingDoc(true)
    try {
      const res = await fetch(`${API_URL}/documents/${docToDelete.doc_id}`, {
        method: 'DELETE'
      })
      
      if (res.ok) {
        const data = await res.json()
        alert(`Document "${data.filename}" deleted successfully`)
        
        // Clear selected doc if it was the one deleted
        if (selectedDoc?.doc_id === docToDelete.doc_id) {
          setSelectedDoc(null)
          setMessages([])
          setExtractedData(null)
        }
        
        // Refresh document list
        fetchDocs()
      } else {
        const error = await res.json()
        throw new Error(error.detail || 'Failed to delete document')
      }
    } catch (error) {
      alert(`Error deleting document: ${error.message}`)
    } finally {
      setDeletingDoc(false)
      setShowDeleteModal(false)
      setDocToDelete(null)
    }
  }

  const handleDeleteCancel = () => {
    setShowDeleteModal(false)
    setDocToDelete(null)
  }

  const handleGetEvents = async (entityName = '') => {
    if (!selectedDoc) return
    
    setLoadingEvents(true)
    setEventsData(null)
    
    try {
      const url = entityName 
        ? `${API_URL}/events/${selectedDoc.doc_id}?entity_name=${encodeURIComponent(entityName)}`
        : `${API_URL}/events/${selectedDoc.doc_id}`
      
      const res = await fetch(url)
      const data = await res.json()
      setEventsData(data)
    } catch (error) {
      alert('Error loading events timeline')
    } finally {
      setLoadingEvents(false)
    }
  }

  const handleEntityFilterSubmit = () => {
    handleGetEvents(entityFilter)
  }

  const SidebarButton = ({ id, label, icon: Icon, active, onClick }) => (
    <button
      onClick={() => onClick(id)}
      className={`w-full flex items-center px-4 py-3 rounded-xl text-sm font-medium transition-all duration-300 hover-lift group ${
        active 
          ? 'bg-gradient-to-r from-blue-500 to-purple-500 text-white shadow-elegant transform scale-105' 
          : 'text-slate-600 hover:bg-gradient-to-r hover:from-blue-50 hover:to-purple-50 hover:text-slate-800 hover:shadow-elegant'
      }`}
    >
      <Icon className={`w-4 h-4 mr-3 transition-all duration-300 ${active ? 'text-white' : 'text-slate-500 group-hover:text-blue-500'}`} />
      {label}
    </button>
  )

  const StatusBadge = ({ status, label }) => (
    <div className={`flex items-center justify-between py-2 px-3 rounded-lg text-xs font-medium transition-all duration-300 ${
      status ? 'text-emerald-700 bg-emerald-50 border border-emerald-200' : 'text-red-700 bg-red-50 border border-red-200'
    }`}>
      <span className="flex items-center">
        <div className={`status-indicator w-3 h-3 rounded-full mr-2 shadow-sm ${
          status ? 'bg-emerald-500 online' : 'bg-red-500 offline'
        }`}></div>
        {label}
      </span>
      <div className={`w-2 h-2 rounded-full shadow-sm ${
        status ? 'bg-emerald-400' : 'bg-red-400'
      }`}></div>
    </div>
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 flex animate-fade-in">
      {/* Sidebar */}
      <div className="w-64 bg-white/90 backdrop-blur-xl border-r border-white/20 flex flex-col shadow-elegant-lg">
        {/* Logo */}
        <div className="p-6 border-b border-white/10">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center shadow-elegant animate-scale-in">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-gradient">RAG Platform</h1>
              <p className="text-xs text-gradient-brand">Knowledge Intelligence</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 p-4 space-y-1">
          <SidebarButton 
            id="documents" 
            label="Documents" 
            icon={Database} 
            active={activeTab === 'documents'} 
            onClick={setActiveTab}
          />
          <SidebarButton 
            id="chat" 
            label="AI Assistant" 
            icon={MessageSquare} 
            active={activeTab === 'chat'} 
            onClick={setActiveTab}
          />
          <SidebarButton 
            id="extraction" 
            label="Knowledge Mining" 
            icon={Target} 
            active={activeTab === 'extraction'} 
            onClick={setActiveTab}
          />
          <SidebarButton 
            id="events" 
            label="Events Timeline" 
            icon={Clock} 
            active={activeTab === 'events'} 
            onClick={setActiveTab}
          />
        </nav>

        {/* System Status */}
        <div className="p-4 border-t border-white/10">
          <div className="space-y-3">
            <div className="text-xs font-bold text-gradient uppercase tracking-wide">System Status</div>
            {systemHealth && (
              <div className="space-y-2">
                <StatusBadge status={systemHealth.openai_configured} label="OpenAI API" />
                <StatusBadge status={systemHealth.gemini_configured} label="Gemini API" />
                <StatusBadge status={systemHealth.documents_in_storage > 0} label={`${systemHealth.documents_in_storage} Documents`} />
              </div>
            )}
            <Button 
              variant="outline" 
              size="sm"
              className="w-full justify-start"
              onClick={() => setShowPromptEditor(!showPromptEditor)}
            >
              <Settings className="w-4 h-4 mr-2" />
              Configure AI
            </Button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <div className="bg-white/90 backdrop-blur-xl border-b border-white/20 px-6 py-4 shadow-elegant">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-xl font-bold text-gradient">
                {activeTab === 'documents' && 'Document Library'}
                {activeTab === 'chat' && 'AI Assistant'}
                {activeTab === 'extraction' && 'Knowledge Mining'}
                {activeTab === 'events' && 'Events Timeline'}
              </h2>
              <p className="text-sm text-slate-600 mt-1">
                {activeTab === 'documents' && 'Upload and manage your PDF documents'}
                {activeTab === 'chat' && 'Intelligent conversations with your documents'}
                {activeTab === 'extraction' && 'Extract structured knowledge with AI'}
                {activeTab === 'events' && 'Track temporal events and their relationships'}
              </p>
            </div>
            <div className="flex items-center space-x-3">
              <Button variant="outline" size="sm">
                <Bell className="w-4 h-4" />
              </Button>
              <Button variant="outline" size="sm">
                <Search className="w-4 h-4" />
              </Button>
            </div>
          </div>
        </div>

        {/* Content Area */}
        <div className="flex-1 p-6 bg-transparent overflow-auto">
          {/* Prompt Editor Modal */}
          {showPromptEditor && (
            <div className="fixed inset-0 bg-black/30 backdrop-blur-lg flex items-center justify-center z-50 p-4 animate-fade-in">
              <div className="card-modern rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-elegant-xl border-0 animate-scale-in">
                <div className="p-6 border-b border-gray-200">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                        <Settings className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <h2 className="text-xl font-bold text-gradient">AI Configuration</h2>
                        <p className="text-gray-500 text-sm mt-1">Customize prompts for optimal AI performance</p>
                      </div>
                    </div>
                    <Button 
                      variant="outline" 
                      onClick={() => setShowPromptEditor(false)}
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
                
                <div className="p-6 overflow-y-auto max-h-[calc(90vh-140px)]">
                  <div className="space-y-6">
                    {/* Extraction Prompt */}
                    <div className="border border-gray-200 rounded-lg p-6">
                      <label className="flex items-center space-x-2 text-sm font-medium text-gray-700 mb-3">
                        <Brain className="w-4 h-4" />
                        <span>Knowledge Extraction Prompt</span>
                      </label>
                      <textarea
                        value={prompts.extraction}
                        onChange={(e) => setPrompts({...prompts, extraction: e.target.value})}
                        className="w-full h-40 p-3 border border-gray-300 rounded-lg text-gray-900 placeholder-gray-500 font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 resize-none"
                        placeholder="Configure AI extraction behavior..."
                      />
                    </div>
                    
                    {/* Chat Prompt */}
                    <div className="border border-gray-200 rounded-lg p-6">
                      <label className="flex items-center space-x-2 text-sm font-medium text-gray-700 mb-3">
                        <MessageSquare className="w-4 h-4" />
                        <span>Chat Assistant Prompt</span>
                      </label>
                      <textarea
                        value={prompts.chat}
                        onChange={(e) => setPrompts({...prompts, chat: e.target.value})}
                        className="w-full h-32 p-3 border border-gray-300 rounded-lg text-gray-900 placeholder-gray-500 font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 resize-none"
                        placeholder="Configure chat assistant behavior..."
                      />
                    </div>
                    
                    <div className="flex justify-end space-x-3 pt-4">
                      <Button 
                        variant="outline" 
                        onClick={() => setShowPromptEditor(false)}
                      >
                        Cancel
                      </Button>
                      <Button 
                        onClick={savePrompts}
                        className="bg-blue-600 hover:bg-blue-700"
                      >
                        <CheckCircle2 className="w-4 h-4 mr-2" />
                        Save Changes
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Delete Confirmation Modal */}
          {showDeleteModal && docToDelete && (
            <div className="fixed inset-0 bg-black/30 backdrop-blur-lg flex items-center justify-center z-50 p-4 animate-fade-in">
              <div className="card-modern rounded-2xl max-w-md w-full shadow-elegant-xl border-0 animate-scale-in">
                <div className="p-6">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-red-100 rounded-lg flex items-center justify-center">
                      <AlertTriangle className="w-5 h-5 text-red-600" />
                    </div>
                    <div>
                      <h3 className="text-lg font-semibold text-gray-900">Delete Document</h3>
                      <p className="text-sm text-gray-500">This action cannot be undone</p>
                    </div>
                  </div>
                  
                  <div className="mb-6">
                    <p className="text-gray-700">
                      Are you sure you want to delete <strong>"{docToDelete.filename}"</strong>?
                    </p>
                    <p className="text-sm text-gray-500 mt-2">
                      This will permanently remove the document and all its associated data.
                    </p>
                  </div>
                  
                  <div className="flex justify-end space-x-3">
                    <Button 
                      variant="outline" 
                      onClick={handleDeleteCancel}
                      disabled={deletingDoc}
                    >
                      Cancel
                    </Button>
                    <Button 
                      onClick={handleDeleteConfirm}
                      disabled={deletingDoc}
                      className="bg-red-600 hover:bg-red-700 text-white"
                    >
                      {deletingDoc ? (
                        <div className="flex items-center">
                          <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                          Deleting...
                        </div>
                      ) : (
                        <div className="flex items-center">
                          <Trash2 className="h-4 w-4 mr-2" />
                          Delete Document
                        </div>
                      )}
                    </Button>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'documents' && (
            <div className="space-y-6">
              {/* Upload Section */}
              <Card>
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center">
                        <Upload className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <CardTitle className="text-lg text-gray-900">Upload Document</CardTitle>
                        <p className="text-sm text-slate-600 mt-1">Add PDFs to your knowledge base</p>
                      </div>
                    </div>
                    <Button size="sm" className="bg-blue-600 hover:bg-blue-700">
                      <Plus className="w-4 h-4 mr-2" />
                      New Upload
                    </Button>
                  </div>
                </CardHeader>
                <CardContent>
                  <div 
                    onClick={() => fileInputRef.current?.click()}
                    className="border-2 border-dashed border-blue-200 rounded-xl p-8 text-center cursor-pointer hover:border-blue-400 hover:bg-gradient-to-br hover:from-blue-50 hover:to-purple-50 transition-all duration-300 group hover-lift bg-white/50 backdrop-blur-sm"
                  >
                    <div className="w-12 h-12 bg-gradient-to-br from-blue-100 to-purple-100 group-hover:from-blue-200 group-hover:to-purple-200 rounded-xl flex items-center justify-center mx-auto mb-4 transition-all duration-300 shadow-elegant group-hover:scale-110">
                      <Upload className="h-6 w-6 text-blue-500 group-hover:text-blue-600 transition-all duration-300" />
                    </div>
                    <p className="text-base font-bold text-gradient mb-2">
                      Drop files here or click to browse
                    </p>
                    <p className="text-sm text-slate-600">PDF files up to 50MB, maximum 1000 pages</p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".pdf"
                      onChange={handleFileUpload}
                      className="hidden"
                    />
                  </div>
                  {uploadProgress > 0 && (
                    <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-xl border border-blue-200 shadow-elegant animate-slide-up">
                      <div className="flex justify-between text-sm text-blue-800 mb-2">
                        <span className="flex items-center font-medium">
                          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                          Processing document...
                        </span>
                        <span className="font-semibold">{uploadProgress}%</span>
                      </div>
                      <div className="w-full bg-blue-200 rounded-full h-3 overflow-hidden shadow-inner">
                        <div 
                          className="bg-gradient-to-r from-blue-500 to-purple-500 h-3 rounded-full transition-all duration-500 shadow-sm"
                          style={{ width: `${uploadProgress}%` }}
                        ></div>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Documents List */}
              <Card>
                <CardHeader className="pb-4">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="w-10 h-10 bg-gray-100 rounded-lg flex items-center justify-center">
                        <Database className="w-5 h-5 text-gray-600" />
                      </div>
                      <div>
                        <CardTitle className="text-lg text-gray-900">Documents</CardTitle>
                        <p className="text-sm text-slate-600 mt-1">{docs.length} document{docs.length !== 1 ? 's' : ''} in your library</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Button variant="outline" size="sm" onClick={fetchDocs}>
                        <Activity className="w-4 h-4 mr-2" />
                        Refresh
                      </Button>
                      <Button variant="outline" size="sm">
                        <Filter className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    {docs.length === 0 ? (
                      <div className="text-center py-12">
                        <div className="w-20 h-20 bg-gradient-to-br from-gray-100 to-gray-200 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-elegant">
                          <FileText className="w-10 h-10 text-slate-400" />
                        </div>
                        <h3 className="text-xl font-bold text-gradient mb-3">No documents yet</h3>
                        <p className="text-slate-600 text-base max-w-sm mx-auto">
                          Upload your first PDF document to get started with AI-powered knowledge extraction
                        </p>
                      </div>
                    ) : (
                      docs.map((doc) => (
                        <div
                          key={doc.doc_id}
                          onClick={() => setSelectedDoc(doc)}
                          className={`group relative flex items-center justify-between p-4 rounded-xl cursor-pointer transition-all duration-300 border hover-lift animate-slide-up ${
                            selectedDoc?.doc_id === doc.doc_id
                              ? 'bg-gradient-to-r from-blue-50 to-purple-50 border-blue-300 shadow-elegant transform scale-105'
                              : 'bg-white/80 backdrop-blur-sm border-gray-200 hover:border-blue-300 hover:shadow-elegant'
                          }`}
                        >
                          <div className="flex items-center flex-1">
                            <div className={`w-10 h-10 rounded-xl flex items-center justify-center mr-3 transition-all duration-300 shadow-elegant ${
                              selectedDoc?.doc_id === doc.doc_id 
                                ? 'bg-gradient-to-br from-blue-600 to-purple-600 transform scale-110' 
                                : 'bg-gradient-to-br from-gray-100 to-gray-200 group-hover:from-blue-100 group-hover:to-purple-100'
                            }`}>
                              <FileText className={`h-5 w-5 ${
                                selectedDoc?.doc_id === doc.doc_id ? 'text-white' : 'text-gray-600'
                              }`} />
                            </div>
                            <div className="flex-1 min-w-0">
                              <h3 className="font-bold text-gradient truncate">{doc.filename}</h3>
                              <div className="flex items-center space-x-4 text-sm text-slate-600 mt-1">
                                <span>{doc.total_pages} pages</span>
                                <span>{Math.round(doc.total_characters / 1000)}k chars</span>
                                <span>{new Date(doc.uploaded_at * 1000).toLocaleDateString()}</span>
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-3">
                            {selectedDoc?.doc_id === doc.doc_id && (
                              <div className="flex items-center space-x-2">
                                <div className="w-2 h-2 bg-blue-600 rounded-full"></div>
                                <span className="text-xs bg-gradient-to-r from-blue-500 to-purple-500 text-white px-3 py-1 rounded-full font-bold shadow-elegant animate-pulse">
                                  Selected
                                </span>
                              </div>
                            )}
                            <Button 
                              variant="ghost" 
                              size="sm"
                              onClick={(e) => {
                                e.stopPropagation()
                                handleDeleteClick(doc)
                              }}
                              className="hover:bg-red-50 hover:text-red-600"
                            >
                              <Trash2 className="w-4 h-4" />
                            </Button>
                            <ArrowRight className="w-4 h-4 text-gray-400 group-hover:text-gray-600" />
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {activeTab === 'chat' && (
            <div className="space-y-6">
              {selectedDoc ? (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center text-gray-900">
                      <div className="w-10 h-10 bg-blue-600 rounded-lg flex items-center justify-center mr-3">
                        <MessageSquare className="w-5 h-5 text-white" />
                      </div>
                      <div>
                        <div className="text-lg">AI Assistant</div>
                        <div className="text-sm text-gray-500 font-normal">Discussing: {selectedDoc.filename}</div>
                      </div>
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-6">
                      <div className="h-96 overflow-y-auto border border-gray-200 rounded-lg p-6 space-y-4 bg-gray-50">
                        {messages.length === 0 && (
                          <div className="text-center py-16 animate-fade-in">
                            <div className="w-16 h-16 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                              <MessageSquare className="w-8 h-8 text-blue-600" />
                            </div>
                            <p className="text-gray-900 font-medium text-lg mb-2">Ready to chat!</p>
                            <p className="text-gray-500 text-sm">Ask me anything about your document</p>
                          </div>
                        )}
                        {messages.map((msg, idx) => (
                          <div
                            key={idx}
                            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                          >
                            <div
                              className={`max-w-2xl p-4 rounded-lg shadow-sm ${
                                msg.role === 'user'
                                  ? 'bg-blue-600 text-white'
                                  : 'bg-white text-gray-800 border border-gray-200'
                              }`}
                            >
                              <div className="flex items-start space-x-3">
                                <div className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 ${
                                  msg.role === 'user' 
                                    ? 'bg-blue-500' 
                                    : 'bg-gray-100'
                                }`}>
                                  {msg.role === 'user' ? (
                                    <span className="text-xs font-bold text-white">You</span>
                                  ) : (
                                    <Bot className="w-4 h-4 text-gray-600" />
                                  )}
                                </div>
                                <div className="flex-1">
                                  <p className="text-sm leading-relaxed">{msg.content}</p>
                                  {msg.confidence && (
                                    <div className="flex items-center space-x-2 mt-3 text-xs opacity-75">
                                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                      <span>Confidence: {Math.round(msg.confidence * 100)}%</span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                        {loading && (
                          <div className="flex justify-start">
                            <div className="bg-white p-4 rounded-lg border border-gray-200 shadow-sm">
                              <div className="flex items-center space-x-3">
                                <div className="w-8 h-8 bg-gray-100 rounded-lg flex items-center justify-center">
                                  <Bot className="w-4 h-4 text-gray-600" />
                                </div>
                                <div className="flex items-center space-x-2">
                                  <Loader2 className="h-4 w-4 animate-spin text-blue-600" />
                                  <span className="text-gray-600 text-sm">AI is thinking...</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        )}
                      </div>
                      
                      <div className="flex gap-3">
                        <div className="flex-1">
                          <input
                            type="text"
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            onKeyPress={(e) => e.key === 'Enter' && handleAsk()}
                            placeholder="Ask anything about your document..."
                            className="w-full px-4 py-3 border border-gray-300 rounded-lg text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
                          />
                        </div>
                        <Button 
                          onClick={handleAsk} 
                          disabled={loading || !question.trim()} 
                          className="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          <Send className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ) : (
                <Card>
                  <CardContent className="text-center py-16">
                    <div className="w-24 h-24 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
                      <MessageSquare className="w-12 h-12 text-gray-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-3">No Document Selected</h3>
                    <p className="text-gray-500 mb-6 text-base max-w-md mx-auto">
                      Choose a document from your library to start an intelligent conversation
                    </p>
                    <Button 
                      className="bg-blue-600 hover:bg-blue-700 px-6 py-2 rounded-lg" 
                      onClick={() => setActiveTab('documents')}
                    >
                      <Database className="w-4 h-4 mr-2" />
                      Go to Documents
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {activeTab === 'events' && (
            <div className="space-y-6">
              {selectedDoc ? (
                <>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center text-gray-900">
                        <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center mr-3">
                          <Clock className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <div className="text-lg">Events Timeline</div>
                          <div className="text-sm text-gray-500 font-normal">Extract temporal events from: {selectedDoc.filename}</div>
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-4">
                        {/* Entity Filter */}
                        <div className="flex gap-3">
                          <div className="flex-1">
                            <input
                              type="text"
                              value={entityFilter}
                              onChange={(e) => setEntityFilter(e.target.value)}
                              placeholder="Filter by entity (e.g., 'Apple', 'CEO', 'Q4 2023')..."
                              className="w-full px-4 py-3 border border-gray-300 rounded-lg text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-purple-500 transition-all duration-200"
                            />
                          </div>
                          <Button 
                            onClick={handleEntityFilterSubmit}
                            disabled={loadingEvents}
                            className="bg-purple-600 hover:bg-purple-700 px-6 py-3 rounded-lg transition-all duration-200 disabled:opacity-50"
                          >
                            <FilterIcon className="h-4 w-4 mr-2" />
                            Filter Events
                          </Button>
                        </div>

                        {/* Extract All Events Button */}
                        <Button 
                          onClick={() => handleGetEvents()}
                          disabled={loadingEvents}
                          className="w-full bg-purple-600 hover:bg-purple-700 py-4 rounded-lg text-base font-medium transition-all duration-200 disabled:opacity-50"
                        >
                          {loadingEvents ? (
                            <div className="flex items-center">
                              <Loader2 className="h-5 w-5 mr-3 animate-spin" />
                              <span>Extracting events timeline...</span>
                            </div>
                          ) : (
                            <div className="flex items-center">
                              <EventIcon className="h-5 w-5 mr-3" />
                              <span>Extract Events Timeline</span>
                            </div>
                          )}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Events Timeline Display */}
                  {eventsData && (
                    <div className="space-y-6">
                      {/* Events Summary */}
                      <Card>
                        <CardHeader>
                          <CardTitle className="flex items-center text-gray-900 text-lg">
                            <BarChart3 className="w-6 h-6 text-purple-600 mr-3" />
                            Timeline Results
                          </CardTitle>
                        </CardHeader>
                        <CardContent>
                          <div className="grid grid-cols-3 gap-4">
                            <div className="text-center p-4 bg-purple-50 border border-purple-200 rounded-lg">
                              <EventIcon className="w-6 h-6 text-purple-600 mx-auto mb-2" />
                              <div className="text-2xl font-bold text-purple-600 mb-1">
                                {eventsData.total_events || 0}
                              </div>
                              <div className="text-sm font-medium text-purple-800">Events</div>
                            </div>
                            <div className="text-center p-4 bg-blue-50 border border-blue-200 rounded-lg">
                              <Users className="w-6 h-6 text-blue-600 mx-auto mb-2" />
                              <div className="text-2xl font-bold text-blue-600 mb-1">
                                {eventsData.related_entities?.length || 0}
                              </div>
                              <div className="text-sm font-medium text-blue-800">Entities</div>
                            </div>
                            <div className="text-center p-4 bg-green-50 border border-green-200 rounded-lg">
                              <Hash className="w-6 h-6 text-green-600 mx-auto mb-2" />
                              <div className="text-2xl font-bold text-green-600 mb-1">
                                {eventsData.related_metrics?.length || 0}
                              </div>
                              <div className="text-sm font-medium text-green-800">Metrics</div>
                            </div>
                          </div>
                          {eventsData.entity_filter && (
                            <div className="mt-4 text-center p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                              <div className="flex items-center justify-center space-x-2">
                                <FilterIcon className="w-4 h-4 text-yellow-600" />
                                <span className="text-yellow-800 font-medium">Filtered by: "{eventsData.entity_filter}"</span>
                              </div>
                            </div>
                          )}
                        </CardContent>
                      </Card>

                      {/* Events List */}
                      {eventsData.events && eventsData.events.length > 0 && (
                        <Card>
                          <CardHeader>
                            <CardTitle className="flex items-center text-gray-900">
                              <Clock className="w-5 h-5 text-purple-600 mr-2" />
                              Events Timeline ({eventsData.events.length})
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-4 max-h-80 overflow-y-auto">
                              {eventsData.events.map((event, idx) => (
                                <div key={idx} className="border-l-4 border-purple-500 pl-4 py-4 bg-purple-50 rounded-r-lg hover:bg-purple-100 transition-colors">
                                  <div className="flex items-start justify-between mb-2">
                                    <h3 className="font-semibold text-purple-900 text-lg">{event.title}</h3>
                                    <div className="flex items-center space-x-2">
                                      {event.certainty && (
                                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                                          event.certainty === 'certain' 
                                            ? 'bg-green-100 text-green-700' 
                                            : 'bg-yellow-100 text-yellow-700'
                                        }`}>
                                          {event.certainty}
                                        </span>
                                      )}
                                      {event.importance && (
                                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                                          event.importance === 'high' ? 'bg-red-100 text-red-700' : 
                                          event.importance === 'medium' ? 'bg-yellow-100 text-yellow-700' : 
                                          'bg-gray-100 text-gray-700'
                                        }`}>
                                          {event.importance}
                                        </span>
                                      )}
                                    </div>
                                  </div>
                                  
                                  <p className="text-purple-800 mb-3 leading-relaxed">{event.description}</p>
                                  
                                  {/* Temporal Information */}
                                  <div className="flex items-center space-x-4 text-sm text-purple-700 mb-3">
                                    <div className="flex items-center space-x-1">
                                      <Calendar className="w-4 h-4" />
                                      <span className="font-medium">{event.temporal_marker}</span>
                                    </div>
                                    {event.page_number && (
                                      <div className="flex items-center space-x-1">
                                        <FileText className="w-4 h-4" />
                                        <span>Page {event.page_number}</span>
                                      </div>
                                    )}
                                    {event.type && (
                                      <span className="bg-purple-200 px-2 py-1 rounded text-xs font-medium">
                                        {event.type}
                                      </span>
                                    )}
                                  </div>
                                  
                                  {/* Supporting Text */}
                                  {event.supporting_text && (
                                    <div className="bg-white border border-purple-200 rounded-lg p-3 mb-3">
                                      <div className="text-xs text-purple-600 font-medium mb-1">Supporting Evidence:</div>
                                      <p className="text-sm text-gray-700 italic leading-relaxed">"{event.supporting_text}"</p>
                                    </div>
                                  )}
                                  
                                  {/* Related Entities */}
                                  {event.entity_ids && event.entity_ids.length > 0 && (
                                    <div className="flex items-center space-x-2 text-xs">
                                      <span className="text-purple-600 font-medium">Related:</span>
                                      {event.entity_ids.slice(0, 3).map((entityId, i) => (
                                        <span key={i} className="bg-blue-100 text-blue-700 px-2 py-1 rounded">
                                          {entityId}
                                        </span>
                                      ))}
                                      {event.entity_ids.length > 3 && (
                                        <span className="text-purple-600">+{event.entity_ids.length - 3} more</span>
                                      )}
                                    </div>
                                  )}
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}

                      {/* JSON Export */}
                      <Card>
                        <CardHeader>
                          <CardTitle>
                            <Button
                              variant="ghost"
                              onClick={() => setShowRawJson(!showRawJson)}
                              className="p-0 h-auto font-semibold text-gray-700 hover:text-gray-900"
                            >
                              <div className="flex items-center space-x-2">
                                <FileText className="w-4 h-4" />
                                <span>JSON Export {showRawJson ? '▼' : '▶'}</span>
                              </div>
                            </Button>
                          </CardTitle>
                        </CardHeader>
                        {showRawJson && (
                          <CardContent>
                            <div className="bg-gray-900 border border-gray-300 rounded-lg p-4">
                              <pre className="text-green-400 overflow-x-auto text-xs max-h-96 overflow-y-auto font-mono leading-relaxed">
                                {JSON.stringify(eventsData, null, 2)}
                              </pre>
                            </div>
                          </CardContent>
                        )}
                      </Card>
                    </div>
                  )}
                </>
              ) : (
                <Card>
                  <CardContent className="text-center py-16">
                    <div className="w-24 h-24 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
                      <Clock className="w-12 h-12 text-gray-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-3">Ready for Events Timeline</h3>
                    <p className="text-gray-500 mb-6 text-base max-w-md mx-auto">
                      Select a document to extract temporal events and create an interactive timeline
                    </p>
                    <Button 
                      className="bg-purple-600 hover:bg-purple-700 px-6 py-2 rounded-lg" 
                      onClick={() => setActiveTab('documents')}
                    >
                      <Database className="w-4 h-4 mr-2" />
                      Choose Document
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          )}

          {activeTab === 'extraction' && (
            <div className="space-y-6">
              {selectedDoc ? (
                <>
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center text-gray-900">
                        <div className="w-10 h-10 bg-purple-600 rounded-lg flex items-center justify-center mr-3">
                          <Target className="w-5 h-5 text-white" />
                        </div>
                        <div>
                          <div className="text-lg">Knowledge Mining</div>
                          <div className="text-sm text-gray-500 font-normal">Extract insights from: {selectedDoc.filename}</div>
                        </div>
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Button 
                        onClick={handleExtract} 
                        disabled={loading}
                        className="w-full bg-purple-600 hover:bg-purple-700 py-4 rounded-lg text-base font-medium transition-all duration-200 disabled:opacity-50"
                      >
                        {loading ? (
                          <div className="flex items-center">
                            <Loader2 className="h-5 w-5 mr-3 animate-spin" />
                            <span>AI is extracting knowledge...</span>
                          </div>
                        ) : (
                          <div className="flex items-center">
                            <Brain className="h-5 w-5 mr-3" />
                            <span>Start Knowledge Extraction</span>
                          </div>
                        )}
                      </Button>
                    </CardContent>
                  </Card>

                  {/* Extracted Data Display */}
                  {extractedData && (
                    <div className="space-y-6">
                      {/* Summary Stats */}
                      {extractedData.extraction_stats && (
                        <Card>
                          <CardHeader>
                            <CardTitle className="flex items-center text-gray-900 text-lg">
                              <BarChart3 className="w-6 h-6 text-purple-600 mr-3" />
                              Extraction Results
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                              <div className="text-center p-4 bg-blue-50 border border-blue-200 rounded-lg">
                                <Quote className="w-6 h-6 text-blue-600 mx-auto mb-2" />
                                <div className="text-2xl font-bold text-blue-600 mb-1">
                                  {extractedData.extraction_stats.totals.quotes}
                                </div>
                                <div className="text-sm font-medium text-blue-800">Quotes</div>
                              </div>
                              <div className="text-center p-4 bg-green-50 border border-green-200 rounded-lg">
                                <Users className="w-6 h-6 text-green-600 mx-auto mb-2" />
                                <div className="text-2xl font-bold text-green-600 mb-1">
                                  {extractedData.extraction_stats.totals.entities}
                                </div>
                                <div className="text-sm font-medium text-green-800">Entities</div>
                              </div>
                              <div className="text-center p-4 bg-purple-50 border border-purple-200 rounded-lg">
                                <TrendingUp className="w-6 h-6 text-purple-600 mx-auto mb-2" />
                                <div className="text-2xl font-bold text-purple-600 mb-1">
                                  {extractedData.extraction_stats.totals.metrics}
                                </div>
                                <div className="text-sm font-medium text-purple-800">Metrics</div>
                              </div>
                              <div className="text-center p-4 bg-orange-50 border border-orange-200 rounded-lg">
                                <Network className="w-6 h-6 text-orange-600 mx-auto mb-2" />
                                <div className="text-2xl font-bold text-orange-600 mb-1">
                                  {extractedData.extraction_stats.totals.relations}
                                </div>
                                <div className="text-sm font-medium text-orange-800">Relations</div>
                              </div>
                            </div>
                            {extractedData.confidence && (
                              <div className="mt-6 text-center p-4 bg-green-50 border border-green-200 rounded-lg">
                                <div className="flex items-center justify-center space-x-2">
                                  <CheckCircle2 className="w-5 h-5 text-green-600" />
                                  <span className="text-green-800 font-medium">Extraction Confidence:</span>
                                  <span className="font-bold text-xl text-green-600">
                                    {Math.round(extractedData.confidence * 100)}%
                                  </span>
                                </div>
                              </div>
                            )}
                          </CardContent>
                        </Card>
                      )}

                      {/* Quotes */}
                      {extractedData.quotes && extractedData.quotes.length > 0 && (
                        <Card>
                          <CardHeader>
                            <CardTitle className="flex items-center text-gray-900">
                              <Quote className="w-5 h-5 text-blue-600 mr-2" />
                              Extracted Quotes ({extractedData.quotes.length})
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="space-y-4 max-h-80 overflow-y-auto">
                              {extractedData.quotes.map((quote, idx) => (
                                <div key={idx} className="border-l-4 border-blue-500 pl-4 py-3 bg-blue-50 rounded-r-lg">
                                  <p className="text-sm italic text-gray-800 leading-relaxed">"{quote.text}"</p>
                                  <div className="text-xs text-gray-600 mt-2 flex items-center space-x-2">
                                    <span className="font-medium">— {quote.author || 'Unknown'}</span>
                                    {quote.page && <span className="bg-gray-200 px-2 py-1 rounded">Page {quote.page}</span>}
                                    {quote.importance && (
                                      <span className={`px-2 py-1 rounded text-white text-xs ${
                                        quote.importance === 'high' ? 'bg-red-500' : 
                                        quote.importance === 'medium' ? 'bg-yellow-500' : 'bg-gray-500'
                                      }`}>
                                        {quote.importance}
                                      </span>
                                    )}
                                  </div>
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}

                      {/* Entities */}
                      {extractedData.entities && extractedData.entities.length > 0 && (
                        <Card>
                          <CardHeader>
                            <CardTitle className="flex items-center text-gray-900">
                              <Users className="w-5 h-5 text-green-600 mr-2" />
                              Key Entities ({extractedData.entities.length})
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-80 overflow-y-auto">
                              {extractedData.entities.map((entity, idx) => (
                                <div key={idx} className="border border-green-200 rounded-lg p-4 bg-green-50 hover:bg-green-100 transition-colors">
                                  <div className="font-semibold text-green-800 mb-1">{entity.name}</div>
                                  <div className="text-xs text-green-600 mb-2 flex items-center space-x-2">
                                    <span className="bg-green-200 px-2 py-1 rounded">{entity.type}</span>
                                    {entity.importance && (
                                      <span className={`px-2 py-1 rounded text-white text-xs ${
                                        entity.importance === 'high' ? 'bg-red-500' : 
                                        entity.importance === 'medium' ? 'bg-yellow-500' : 'bg-gray-500'
                                      }`}>
                                        {entity.importance}
                                      </span>
                                    )}
                                  </div>
                                  {entity.description && (
                                    <p className="text-xs text-green-700 leading-relaxed">
                                      {entity.description}
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}

                      {/* Metrics */}
                      {extractedData.metrics && extractedData.metrics.length > 0 && (
                        <Card>
                          <CardHeader>
                            <CardTitle className="flex items-center text-gray-900">
                              <TrendingUp className="w-5 h-5 text-purple-600 mr-2" />
                              Quantitative Data ({extractedData.metrics.length})
                            </CardTitle>
                          </CardHeader>
                          <CardContent>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 max-h-80 overflow-y-auto">
                              {extractedData.metrics.map((metric, idx) => (
                                <div key={idx} className="border border-purple-200 rounded-lg p-4 bg-purple-50 hover:bg-purple-100 transition-colors">
                                  <div className="font-bold text-purple-800 text-lg mb-1">
                                    {metric.value} {metric.unit || ''}
                                  </div>
                                  <div className="text-xs text-purple-600 mb-2">
                                    <span className="bg-purple-200 px-2 py-1 rounded mr-2">{metric.type}</span>
                                    {metric.significance && (
                                      <span className={`px-2 py-1 rounded text-white text-xs ${
                                        metric.significance === 'high' ? 'bg-red-500' : 
                                        metric.significance === 'medium' ? 'bg-yellow-500' : 'bg-gray-500'
                                      }`}>
                                        {metric.significance}
                                      </span>
                                    )}
                                  </div>
                                  {metric.context && (
                                    <p className="text-xs text-purple-700 leading-relaxed">
                                      {metric.context}
                                    </p>
                                  )}
                                </div>
                              ))}
                            </div>
                          </CardContent>
                        </Card>
                      )}

                      {/* Raw JSON for developers */}
                      <Card>
                        <CardHeader>
                          <CardTitle>
                            <Button
                              variant="ghost"
                              onClick={() => setShowRawJson(!showRawJson)}
                              className="p-0 h-auto font-semibold text-gray-700 hover:text-gray-900"
                            >
                              <div className="flex items-center space-x-2">
                                <FileText className="w-4 h-4" />
                                <span>Developer Data {showRawJson ? '▼' : '▶'}</span>
                              </div>
                            </Button>
                          </CardTitle>
                        </CardHeader>
                        {showRawJson && (
                          <CardContent>
                            <div className="bg-gray-900 border border-gray-300 rounded-lg p-4">
                              <pre className="text-green-400 overflow-x-auto text-xs max-h-96 overflow-y-auto font-mono leading-relaxed">
                                {JSON.stringify(extractedData, null, 2)}
                              </pre>
                            </div>
                          </CardContent>
                        )}
                      </Card>
                    </div>
                  )}
                </>
              ) : (
                <Card>
                  <CardContent className="text-center py-16">
                    <div className="w-24 h-24 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-6">
                      <Target className="w-12 h-12 text-gray-400" />
                    </div>
                    <h3 className="text-xl font-semibold text-gray-900 mb-3">Ready for Knowledge Mining</h3>
                    <p className="text-gray-500 mb-6 text-base max-w-md mx-auto">
                      Select a document to extract valuable insights with AI
                    </p>
                    <Button 
                      className="bg-purple-600 hover:bg-purple-700 px-6 py-2 rounded-lg" 
                      onClick={() => setActiveTab('documents')}
                    >
                      <Database className="w-4 h-4 mr-2" />
                      Choose Document
                    </Button>
                  </CardContent>
                </Card>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App