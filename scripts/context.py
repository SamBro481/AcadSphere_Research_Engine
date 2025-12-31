import numpy as np

class ContextManager:
    def __init__(self, max_history=5, alpha=0.7):
        """
        max_history: how many past queries to remember
        alpha: how strongly recent queries matter
        """
        self.max_history = max_history
        self.alpha = alpha
        self.history = []

    def add_query(self, query_embedding):
        """
        query_embedding: numpy array (1, dim)
        """
        self.history.append(query_embedding)

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def get_context_vector(self):
        """
        Returns a single context embedding
        """
        if not self.history:
            return None

        weights = []
        for i in range(len(self.history)):
            weights.append(self.alpha ** (len(self.history) - i - 1))

        weights = np.array(weights)
        weights = weights / weights.sum()

        context = np.zeros_like(self.history[0])
        for w, emb in zip(weights, self.history):
            context += w * emb

        return context
