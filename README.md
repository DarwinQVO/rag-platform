# 🧠 RAG Platform - Knowledge Intelligence

A **modern, beautifully designed** RAG (Retrieval-Augmented Generation) platform for intelligent document processing and knowledge extraction with a stunning glassmorphism UI.

![RAG Platform](https://img.shields.io/badge/Status-Production_Ready-brightgreen)
![Version](https://img.shields.io/badge/Version-2.0-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

## ✨ Features

- **🎨 Modern Glassmorphism UI**: Stunning visual design with gradients, blur effects, and smooth animations
- **📄 Smart Document Upload**: Drag & drop PDF processing with real-time progress
- **🤖 AI-Powered Chat**: Interactive conversations with your documents using GPT-4
- **🔍 Knowledge Mining**: Extract quotes, entities, metrics, and relationships automatically
- **⏰ Events Timeline**: Extract and visualize temporal events from documents
- **🚀 Real-time Processing**: Live updates with elegant loading states
- **🎯 Semantic Search**: Advanced document search capabilities
- **📊 Visual Analytics**: Beautiful data visualization with charts and metrics

## 🛠 Tech Stack

### Backend
- **FastAPI**: High-performance async Python web framework
- **OpenAI GPT-4o**: State-of-the-art language model
- **Google Gemini**: Alternative AI model for large documents
- **LangChain**: Advanced LLM application framework
- **Supabase + pgvector**: Vector database for semantic search
- **PostgreSQL**: Structured data storage

### Frontend
- **React 19**: Latest React with concurrent features
- **Vite**: Lightning-fast build tool
- **Tailwind CSS 4**: Modern utility-first CSS framework
- **Lucide React**: Beautiful, consistent icons
- **shadcn/ui**: Professional component library
- **Custom Animations**: Smooth CSS animations and transitions

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 18+
- OpenAI API key
- Git

### Local Development

```bash
# Clone and setup
git clone https://github.com/your-username/rag-platform.git
cd rag-platform

# Install all dependencies
make install

# Set up environment variables
cp .env.example .env
# Add your OpenAI API key to .env

# Start both frontend and backend
make dev
```

**Access the application:**
- 🎨 **Frontend**: http://localhost:5173
- 🔧 **Backend API**: http://localhost:8000
- 📚 **API Docs**: http://localhost:8000/docs

## 🌐 Production Deployment

### Option 1: Vercel + Railway (Recommended)

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed step-by-step instructions.

**Quick Deploy:**
1. Push to GitHub
2. Deploy backend on Railway
3. Deploy frontend on Vercel
4. Configure environment variables

### Option 2: Docker Deployment

```bash
# Coming soon - Docker Compose setup
docker-compose up --build
```

## 📖 Usage

### 1. Document Management
- **Upload PDFs**: Drag & drop or click to upload
- **View Library**: Browse all uploaded documents
- **Document Stats**: See page count, character count, upload date

### 2. AI Assistant
- **Smart Chat**: Ask questions about document content
- **Context Aware**: AI remembers conversation context
- **Source Citations**: Get page numbers and confidence scores

### 3. Knowledge Mining
- **Auto Extraction**: AI identifies key information automatically
- **Structured Data**: Quotes, entities, metrics, relationships
- **Visual Results**: Beautiful cards with importance indicators

### 4. Events Timeline
- **Temporal Analysis**: Extract time-based events
- **Entity Filtering**: Focus on specific people/organizations
- **Evidence Support**: Direct quotes backing each event

## 🎨 Design Highlights

- **Glassmorphism Effects**: Translucent surfaces with backdrop blur
- **Gradient Overlays**: Subtle blue-to-purple gradients throughout
- **Smooth Animations**: Fade-in, slide-up, and scale animations
- **Hover Effects**: Interactive elements with 3D transformations
- **Status Indicators**: Animated status badges with pulse effects
- **Modern Typography**: Gradient text and improved font hierarchy
- **Custom Scrollbars**: Styled scrollbars matching the theme
- **Loading States**: Elegant shimmer and progress animations

## 🔧 Configuration

### Environment Variables

**Backend (.env):**
```env
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key (optional)
SUPABASE_URL=your_supabase_url (optional)
SUPABASE_KEY=your_supabase_key (optional)
```

**Frontend (.env.local):**
```env
VITE_API_URL=http://localhost:8000
```

### Customization

- **Colors**: Edit `src/globals.css` for color scheme changes
- **Animations**: Modify animation classes in globals.css
- **Components**: Customize UI components in `src/components/ui/`
- **AI Prompts**: Configure extraction prompts in the settings panel

## 📊 Performance

- **Frontend**: Sub-100ms page loads with Vite
- **Backend**: Async FastAPI with <200ms response times
- **AI Processing**: Optimized chunking for large documents
- **UI Rendering**: 60fps animations with hardware acceleration

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **OpenAI** for GPT-4 API
- **Vercel** for hosting platform
- **Railway** for backend deployment
- **shadcn** for UI components
- **Tailwind CSS** for styling system

---

**Built with ❤️ for the future of document intelligence**