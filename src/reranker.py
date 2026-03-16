from sentence_transformers import CrossEncoder


class Reranker:

    def __init__(self):
        self.model = CrossEncoder("BAAI/bge-reranker-base")

    def rerank(self, query, docs, top_k=3):

        if not docs:
            return []

        pairs = []

        for doc in docs:
            pairs.append([query, doc.page_content])

        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, docs),
            key=lambda x: x[0],
            reverse=True
        )

        reranked_docs = [doc for _, doc in ranked[:top_k]]

        return reranked_docs