# 🚀 RAG Platform Deployment Guide

## Prerequisites

1. GitHub account
2. Vercel account (free): https://vercel.com
3. Railway account (free): https://railway.app
4. OpenAI API key
5. Git repository

## Step 1: Push to GitHub

```bash
# Initialize git (if not done)
git init
git add .
git commit -m "Initial commit with modern UI"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/rag-platform.git
git push -u origin main
```

## Step 2: Deploy Backend on Railway

1. Go to https://railway.app
2. Click "Start a New Project"
3. Select "Deploy from GitHub repo"
4. Choose your `rag-platform` repository
5. Railway will auto-detect Python and deploy

**Environment Variables to set in Railway:**
- `OPENAI_API_KEY` = your_openai_api_key
- `GEMINI_API_KEY` = your_gemini_api_key (optional)
- `ENVIRONMENT` = production

6. After deployment, copy the Railway URL (e.g., `https://your-app.railway.app`)

## Step 3: Deploy Frontend on Vercel

1. Go to https://vercel.com
2. Click "New Project"
3. Import your GitHub repository
4. Set these settings:
   - **Framework Preset**: Vite
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`

**Environment Variables to set in Vercel:**
- `VITE_API_URL` = your_railway_backend_url

## Step 4: Update CORS

After getting your Vercel URL, update the CORS in `backend/main_fixed.py`:

```python
origins = [
    "http://localhost:5173",
    "https://your-vercel-app.vercel.app",  # Add your actual Vercel URL
]
```

Then redeploy Railway.

## Step 5: Test

Visit your Vercel URL and test all functionality!

## Troubleshooting

- **CORS errors**: Check the origins array in backend
- **API not loading**: Verify VITE_API_URL in Vercel
- **Build fails**: Check Node.js version (use 18.x)
- **Backend fails**: Check Railway logs for Python errors

## URLs Structure

- Frontend: https://your-app.vercel.app
- Backend API: https://your-app.railway.app
- API Health: https://your-app.railway.app/health