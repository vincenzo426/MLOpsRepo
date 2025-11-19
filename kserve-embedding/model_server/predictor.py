import kserve
from typing import Dict
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingPredictor(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.ready = False

    def load(self):
        """Carica il modello sentence-transformers"""
        print(f"Loading model: sentence-transformers/all-MiniLM-L6-v2")
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.ready = True
        print("Model loaded successfully")

    def predict(self, request: Dict, headers: Dict = None) -> Dict:  # <-- AGGIUNTO headers
        """
        Input format: {"instances": ["text1", "text2", ...]}
        Output format: {"predictions": [[embedding1], [embedding2], ...]}
        """
        try:
            texts = request["instances"]
            
            if isinstance(texts, str):
                texts = [texts]
            
            # Genera embeddings
            embeddings = self.model.encode(texts)
            
            # Converti in lista per JSON serialization
            embeddings_list = embeddings.tolist()
            
            return {"predictions": embeddings_list}
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

if __name__ == "__main__":
    model = EmbeddingPredictor("embedding-model")
    model.load()
    kserve.ModelServer().start([model])