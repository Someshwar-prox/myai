====================================
    RAG CHATBOT - QUICK START GUIDE
====================================

Yo, welcome! Here's how to use this thing 👋

WHAT IS THIS?
A local AI chatbot that learns from YOUR files. No internet needed, no API limits.

HOW TO USE IT:

1. ADD YOUR FILES
   - Drop your PDFs, Word docs, or images in THIS FOLDER (assets/)
   - Supports: .txt, .pdf, .docx, .doc, .jpg, .jpeg, .png
   - Mix and match whatever you want

2. TRAIN THE BOT
   Run this command from the project folder:
   
   python rag_system.py learn
   
   This will:
   - Read all your files
   - Split them into chunks
   - Build a search index
   - Setup the local LLM (Gemma3:1b)
   - Takes like 2-5 minutes first time

3. ASK QUESTIONS
   
   Three ways to use it:
   
   a) Interactive Mode (Recommended)
      python rag_system.py
      Then type your questions and get answers
      Type 'exit' to quit, 'learn' to retrain
   
   b) Single Question
      python rag_system.py "What is machine learning?"
      Gets answer and closes
   
   c) Retrain Anytime
      python rag_system.py learn
      Use this when you add new files

REQUIREMENTS:
- Python 3.8+
- Ollama installed and running (ollama serve)
- Gemma3:1b model (downloads automatically first time)

HOW IT WORKS:
- Loads your docs → Splits into chunks → Creates embeddings
- Builds a FAISS vector index for fast searching
- When you ask something, it finds relevant parts of your docs
- Sends context to local Ollama LLM for intelligent answers
- No data leaves your computer

TIPS:
- More documents = smarter answers
- Be specific with questions for better results
- Retrain after adding new files
- Works fully offline (after first setup)
- JSON fallback if LLM isn't available

TROUBLESHOOTING:
- "Ollama is not running" → Start with: ollama serve
- "No documents found" → Make sure files are in assets/
- "Model not found" → It'll auto-download Gemma3:1b on first learn
- Slow responses? → Your CPU just needs time, be patient

That's it bro! Ask away 🚀
