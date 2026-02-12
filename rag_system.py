import os
import json
import hashlib
import sys
from pathlib import Path
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredImageLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from uuid import uuid4
import numpy as np

class SimpleEmbeddings:
    def embed_documents(self, texts):
        result = []
        i = 0
        for text in texts:
            embedding = [hash(text + str(i)) % 1000 / 1000.0 for i in range(384)]
            result.append(embedding)
        return result
    
    def embed_query(self, text):
        return [hash(text + str(i)) % 1000 / 1000.0 for i in range(384)]

class RAGSystem:
    def __init__(self):
        self.assets_dir = Path("./assets")
        self.json_file = Path("./documents.json")
        self.faiss_index = Path("./faiss_index")
        
        self.assets_dir.mkdir(exist_ok=True)
        
        print("[LOADING] --> Initializing search system...")
        self.embeddings = SimpleEmbeddings()
        
        self.vectorstore = None
        self.qa_chain = None
        self.llm = None
        
    def check_ollama(self):
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                gemma_available = any(model.get('name') == 'gemma3:1b' for model in models)
                
                if not gemma_available:
                    print("\n[WARNING] --> Gemma3:1b model not found. Pulling it now...")
                    print("   This will take 2-3 minutes on first run...")
                    os.system("ollama pull gemma3:1b")
                return True
        except:
            print("\n[ERROR] --> Ollama is not running!")
            print("   Please start Ollama first:")
            print("   ollama serve")
            return False
        return True
    
    def init_llm(self):
        if not self.check_ollama():
            return False
        
        try:
            self.llm = OllamaLLM(
                model="gemma3:1b",
                base_url="http://localhost:11434",
                temperature=0.1,
                num_predict=256,
                top_k=10,
                top_p=0.5,
                timeout=30
            )
            return True
        except Exception as e:
            print(f"[WARNING] --> LLM initialization warning: {e}")
            return False
    
    def get_file_hash(self, file_path: Path) -> str:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def load_documents(self):
        if not self.assets_dir.exists():
            print("[ERROR] --> Assets folder not found!")
            return []
        
        files = []
        for ext in ['.txt', '.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png']:
            files.extend(self.assets_dir.glob(f"*{ext}"))
            files.extend(self.assets_dir.glob(f"**/*{ext}"))
        
        if not files:
            print("[FOLDER] --> No documents found in assets folder")
            print("   Please add files to ./assets/ directory")
            return []
        
        print(f"[INFO] --> Found {len(files)} file(s) to process")
        
        documents = []
        for file_path in files:
            try:
                print(f"   Reading: {file_path.name}...", end=" ")
                
                if file_path.suffix.lower() == '.txt':
                    loader = TextLoader(str(file_path), encoding='utf-8')
                elif file_path.suffix.lower() == '.pdf':
                    loader = PyPDFLoader(str(file_path))
                elif file_path.suffix.lower() in ['.docx', '.doc']:
                    loader = UnstructuredWordDocumentLoader(str(file_path))
                elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    loader = UnstructuredImageLoader(str(file_path))
                else:
                    print("[WARNING] --> Unsupported format")
                    continue
                
                docs = loader.load()
                
                for doc in docs:
                    doc.metadata['source'] = file_path.name
                    doc.metadata['file_hash'] = self.get_file_hash(file_path)
                
                documents.extend(docs)
                print(f"[OK] --> ({len(docs)} sections)")
                
            except Exception as e:
                print(f"[ERROR] --> Failed: {str(e)[:50]}")
        
        return documents
    
    def learn(self):
        print("\n" + "="*60)
        print("              [SYSTEM] --> RAG LEARNING MODE")
        print("="*60)
        
        documents = self.load_documents()
        
        if not documents:
            print("\n[ERROR] --> No documents to process!")
            return
        
        print("\n[SPLIT] --> Splitting documents into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"   Created {len(chunks)} text chunks")
        
        print("\n[SAVE] --> Saving to JSON format...")
        json_data = {
            "metadata": {
                "total_files": len(set(doc.metadata['source'] for doc in documents)),
                "total_chunks": len(chunks),
                "processed_date": str(Path(documents[0].metadata.get('processed_date', '')) 
                                     if documents else ''),
                "files": list(set(doc.metadata['source'] for doc in documents))
            },
            "chunks": [
                {
                    "id": i,
                    "content": chunk.page_content[:500],
                    "source": chunk.metadata.get('source', 'unknown'),
                    "preview": chunk.page_content[:100] + "..."
                }
                for i, chunk in enumerate(chunks)
            ]
        }
        
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"   [DONE] --> JSON saved: {self.json_file}")
        
        print("\n[BUILD] --> Building search index...")
        try:
            self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
            self.vectorstore.save_local(str(self.faiss_index))
            print(f"   [DONE] --> FAISS index saved: {self.faiss_index}")
        except Exception as e:
            print(f"   [ERROR] --> Failed to build index: {e}")
            return
        
        print("\n[LLM] --> Initializing local LLM (Gemma3:1b)...")
        if self.init_llm():
            print("   [DONE] --> LLM ready")
        else:
            print("   [WARNING] --> LLM not available - will use JSON only mode")
        
        print("\n" + "="*60)
        print("[SUCCESS] --> Learning process completed!")
        print(f"[STATS] --> Summary:")
        print(f"   - Files processed: {len(json_data['metadata']['files'])}")
        print(f"   - Total chunks: {len(chunks)}")
        print(f"   - JSON size: {os.path.getsize(self.json_file) / 1024:.1f} KB")
        print("="*60)
        print("\n[INFO] --> You can now ask questions!")
        print("   Type 'python rag_system.py' to start asking")
    
    def query(self, question):
        
        if not self.vectorstore and self.faiss_index.exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(self.faiss_index), 
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
            except:
                pass
        
        if not self.vectorstore:
            return self.query_json_fallback(question)
        
        if self.llm or self.init_llm():
            try:
                retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.get_relevant_documents(question)
                
                context = "\n".join([doc.page_content for doc in docs])
                
                prompt = f"""Based ONLY on the following context, answer the question.
If you cannot find the answer, say "I don't have information about that in my notes."

Context:
{context}

Question: {question}

Answer:"""
                
                response = self.llm.invoke(prompt)
                print(f"\n[ANSWER] --> {response}")
                
                print(f"\n[SOURCES] --> Found in:")
                for doc in docs[:2]:
                    print(f"   - {doc.metadata.get('source', 'Unknown')}")
                    
            except Exception as e:
                self.query_json_fallback(question)
        else:
            self.query_json_fallback(question)
    
    def query_json_fallback(self, question):
        if not self.json_file.exists():
            print("\n[ERROR] --> No knowledge base found!")
            print("   Please run: python rag_system.py learn")
            return
        
        with open(self.json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        question_lower = question.lower()
        matches = []
        
        for chunk in data['chunks']:
            content = chunk['content'].lower()
            score = sum(1 for word in question_lower.split() if word in content)
            if score > 0:
                matches.append((score, chunk))
        matches.sort(key=lambda x: x[0], reverse=True)
        
        if matches:
            print(f"\n[SEARCH] --> Found in notes:")
            print(f"   {matches[0][1]['content'][:300]}...")
            print(f"\n[SOURCE] --> {matches[0][1]['source']}")
        else:
            print("\n[ERROR] --> No relevant information found in your notes.")

def main():
    print("\n" + "="*60)
    print("     [SYSTEM] --> RAG ASSISTANT")
    print("="*60)
    print("\nCommands:")
    print("   python rag_system.py learn    --> Learn from documents in assets/")
    print("   python rag_system.py          --> Ask questions interactively")
    print("   python rag_system.py \"query\"  --> Ask a single question")
    print("="*60)
    
    rag = RAGSystem()
    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "learn":
            rag.learn()
            
        else: 
            question = " ".join(sys.argv[1:])
            if rag.faiss_index.exists() or rag.json_file.exists():
                rag.query(question)
            else:
                print("\n[ERROR] --> No knowledge base found")
                print("   Please run: python rag_system.py learn")
    else:
        if not rag.faiss_index.exists() and not rag.json_file.exists():
            print("\n[FOLDER] --> No documents learned yet!")
            print("   Please run: python rag_system.py learn")
            return
        
        print("\n[MODE] --> INTERACTIVE - Ask your questions")
        print("   Type 'exit' to quit, 'learn' to reprocess documents")
        print("-"*60)
        
        while True:
            try:
                question = input("\n[INPUT] --> You: ").strip()
                
                if question.lower() == 'exit':
                    print("\n[EXIT] --> Closing session...")
                    break
                elif question.lower() == 'learn':
                    rag.learn()
                    continue
                elif not question:
                    continue
                
                rag.query(question)
                
            except KeyboardInterrupt:
                print("\n\n[EXIT] --> Session terminated")
                break
            except Exception as e:
                print(f"\n[WARNING] --> Error: {e}")

if __name__ == "__main__":
    main()