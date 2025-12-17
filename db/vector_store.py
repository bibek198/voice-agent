import chromadb
import uuid
import datetime
from chromadb.config import Settings

class VectorStore:
    def __init__(self, persistence_path: str = "db/chroma_db"):
        self.client = chromadb.PersistentClient(path=persistence_path)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(name="interactions")

    def add_interaction(self, question: str, answer: str, audio_path: str, language: str = "hi-en"):
        """
        Stores an interaction in the vector DB.
        """
        interaction_id = str(uuid.uuid4())
        timestamp = datetime.datetime.now().isoformat()
        
        # Meta data
        metadata = {
            "question_text": question,
            "llm_answer_text": answer,
            "audio_file_path": audio_path,
            "timestamp": timestamp,
            "language": language
        }
        
        # We store the question related text in documents for semantic search
        document_text = f"Q: {question} A: {answer}"
        
        self.collection.add(
            documents=[document_text],
            metadatas=[metadata],
            ids=[interaction_id]
        )
        print(f"[DB] Stored interaction {interaction_id}")

    def get_recent_history(self, limit: int = 5):
        """
        Retrieves recent interactions (naive implementation using peek/query).
        ChromaDB is optimized for search, not naive history, but we can query all or just peek.
        """
        # Peek retrieves first N logic, not time sorted necessarily, strictly speaking. 
        # But for persistent simple usage it might suffice or we query by empty embedding with limit? 
        # Actually simplest is to rely on client usage or just peek.
        results = self.collection.peek(limit=limit)
        
        history = []
        if results['metadatas']:
            for i in range(len(results['metadatas'])):
                meta = results['metadatas'][i]
                history.append({
                    "role": "user", "content": meta["question_text"]
                })
                history.append({
                    "role": "assistant", "content": meta["llm_answer_text"]
                })
        
        return history

if __name__ == "__main__":
    db = VectorStore()
    print("DB initialized.")
