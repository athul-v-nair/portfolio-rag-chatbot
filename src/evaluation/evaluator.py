from typing import List, Dict
import numpy as np

from src.retrieval.vector_search import search_vector_db
from src.generation.generation import Generator
from src.utils.constants import MEMORY_STORE

class RAGEvaluator:
    def __init__(self):
        self.generator = Generator()

    # -------------------------
    # RETRIEVAL METRICS
    # -------------------------

    def recall_at_k(self, retrieved_docs, keywords):
        for doc in retrieved_docs:
            content = doc["content"].lower()
            if any(k.lower() in content for k in keywords):
                return 1
        return 0

    def reciprocal_rank(self, retrieved_docs, keywords):
        for idx, doc in enumerate(retrieved_docs, start=1):
            content = doc["content"].lower()
            if any(k.lower() in content for k in keywords):
                return 1 / idx
        return 0

    def context_precision(self, retrieved_docs, keywords):
        relevant = 0
        for doc in retrieved_docs:
            content = doc["content"].lower()
            if any(k.lower() in content for k in keywords):
                relevant += 1
        return relevant / len(retrieved_docs) if retrieved_docs else 0

    # -------------------------
    # GENERATION METRIC
    # -------------------------

    def answer_similarity(self, answer, ground_truth):
        a_tokens = set(answer.lower().split())
        gt_tokens = set(ground_truth.lower().split())

        if not gt_tokens:
            return 0

        return len(a_tokens & gt_tokens) / len(gt_tokens)

    # -------------------------
    # FULL EVALUATION
    # -------------------------

    def evaluate(self, dataset: List[Dict], top_k=3):

        results = []

        for sample in dataset:
            query = sample["query"]
            gt_answer = sample["ground_truth_answer"]
            keywords = sample["relevant_doc_keywords"]

            # Retrieval
            retrieved_docs = search_vector_db(query, top_k=top_k)

            recall = self.recall_at_k(retrieved_docs, keywords)
            mrr = self.reciprocal_rank(retrieved_docs, keywords)
            precision = self.context_precision(retrieved_docs, keywords)

            # Generation
            response = self.generator.generate_answer(query, session_id='test_001')
            answer = ""
            for chunk in response:
                answer += chunk

            # Splitting sources
            answer = answer.split("[SOURCES]")[0].strip()
            
            similarity = self.answer_similarity(answer, gt_answer)

            results.append({
                "query": query,
                "recall@k": recall,
                "mrr": mrr,
                "context_precision": precision,
                "answer_similarity": similarity
            })

        return results
    
    @staticmethod
    def summarize(results):
        return {
            "avg_recall": np.mean([r["recall@k"] for r in results]),
            "avg_mrr": np.mean([r["mrr"] for r in results]),
            "avg_precision": np.mean([r["context_precision"] for r in results]),
            "avg_similarity": np.mean([r["answer_similarity"] for r in results])
        }