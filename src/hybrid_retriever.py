from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, vector_store, documents):
        """
        vector_store : FAISS vector database
        documents    : list of document chunks
        """

        self.vector_store = vector_store
        self.documents = documents

        # Prepare corpus for BM25
        corpus = [doc.page_content.split() for doc in documents]

        self.bm25 = BM25Okapi(corpus)

        # Store debug info
        self.last_vector_results = []
        self.last_bm25_results = []
        self.last_combined_results = []


    def retrieve(self, query, k=4):
        """
        Hybrid retrieval using
        1. Vector search (FAISS)
        2. Keyword search (BM25)

        Returns final merged results but also stores
        intermediate retrieval results for debugging.
        """

        # ---------- Vector Search ----------
        vector_results = self.vector_store.similarity_search(query, k=2)

        # Save for debug panel
        self.last_vector_results = vector_results


        # ---------- BM25 Keyword Search ----------
        tokenized_query = query.split()

        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda i: bm25_scores[i],
            reverse=True
        )[:2]

        keyword_results = [self.documents[i] for i in top_indices]

        # Save for debug panel
        self.last_bm25_results = keyword_results


        # ---------- Merge Results ----------
        results = vector_results + keyword_results


        # ---------- Remove Duplicates ----------
        unique_results = []
        seen = set()

        for doc in results:
            content = doc.page_content

            if content not in seen:
                unique_results.append(doc)
                seen.add(content)


        final_results = unique_results[:k]

        # Save final merged results
        self.last_combined_results = final_results


        return final_results