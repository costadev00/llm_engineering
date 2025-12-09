from typing import List, Tuple

import modal
from sentence_transformers import SentenceTransformer
from agents.agent import Agent


class SpecialistAgent(Agent):
    """Runs the fine-tuned LLM hosted on Modal with optional RAG context."""

    name = "Specialist Agent"
    color = Agent.RED

    def __init__(self, collection=None):
        """Set up Modal client and optional retrieval components."""
        self.log("Specialist Agent is initializing - connecting to modal")
        Pricer = modal.Cls.from_name("pricer-service", "Pricer")
        self.pricer = Pricer()
        self.collection = collection
        self.encoder = None
        if self.collection is not None:
            self.log("Specialist Agent is configuring vector encoder for RAG context")
            self.encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.log("Specialist Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """Create a short context section describing similar products."""
        message = (
            "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        )
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def enrich_description(self, description: str, similars: List[str], prices: List[float]) -> str:
        """Augment the item description with retrieved context."""
        context = self.make_context(similars, prices)
        context += "Item to estimate:\n\n"
        context += description
        return context

    def find_similars(self, description: str) -> Tuple[List[str], List[float]]:
        """Look up similar products in the Chroma datastore."""
        if not self.collection or not self.encoder:
            return [], []
        self.log("Specialist Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        vector = self.encoder.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=5)
        documents = results['documents'][0][:]
        prices = [metadata['price'] for metadata in results['metadatas'][0][:]]
        self.log("Specialist Agent has found similar products")
        return documents, prices
        
    def price(self, description: str) -> float:
        """Make a remote call to return the estimate of the price of this item."""
        prompt = description
        if self.collection and self.encoder:
            similars, prices = self.find_similars(description)
            if similars:
                prompt = self.enrich_description(description, similars, prices)
        else:
            self.log("Specialist Agent has no collection configured; skipping RAG context")
        self.log("Specialist Agent is calling remote fine-tuned model")
        result = self.pricer.price.remote(prompt)
        self.log(f"Specialist Agent completed - predicting ${result:.2f}")
        return result
