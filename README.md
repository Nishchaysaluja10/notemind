# NoteMind: Local LLM RAG Pipeline

This is a completely local Retrieval-Augmented Generation (RAG) system built from scratch to demonstrate core LLM concepts.

## How to Test Locally

1. **Setup the Backend Environment:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Run the FastAPI Server:**
   ```bash
   uvicorn main:app --reload
   ```
   *Note: On first run, this will download the embedding model (~90MB) and the Qwen 1.5B model (~3-4GB).*

3. **Open the Frontend:**
   Just double-click `frontend/index.html` in your web browser. 
   (It connects automatically to `http://localhost:8000`).

## Deployment

**Backend (Hugging Face Spaces):**
1. Create a new "Docker" Space on Hugging Face.
2. Upload the contents of the `backend/` folder (including the Dockerfile).
3. The Space will build and serve your API on port 7860.

**Frontend (Netlify / Vercel):**
1. Open `frontend/index.html` and update `const API_URL = "http://localhost:8000";` to your Hugging Face Space URL.
2. Deploy the `frontend/` folder to Netlify (drag and drop) or Vercel.
