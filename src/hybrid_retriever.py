from rank_bm25 import BM25Okapi


class HybridRetriever:

    def __init__(self, vector_store, documents):

        self.vector_store = vector_store
        self.documents = documents if documents else []

        # debug storage
        self.last_vector_results = []
        self.last_bm25_results = []
        self.last_combined_results = []

        # Build BM25 safely
        if self.documents and len(self.documents) > 0:

            corpus = []

            for doc in self.documents:
                text = doc.page_content.strip()

                if text:
                    corpus.append(text.split())

            if len(corpus) > 0:
                self.bm25 = BM25Okapi(corpus)
            else:
                self.bm25 = None

        else:
            self.bm25 = None


    def retrieve(self, query, k=4):

        if not self.vector_store:
            return []

        # vector search
        vector_results = self.vector_store.similarity_search(query, k=2)

        self.last_vector_results = vector_results

        keyword_results = []

        # BM25 search only if initialized
        if self.bm25:

            tokenized_query = query.split()

            bm25_scores = self.bm25.get_scores(tokenized_query)

            top_indices = sorted(
                range(len(bm25_scores)),
                key=lambda i: bm25_scores[i],
                reverse=True
            )[:2]

            keyword_results = [self.documents[i] for i in top_indices]

        self.last_bm25_results = keyword_results

        # combine
        results = vector_results + keyword_results

        unique_results = []
        seen = set()

        for doc in results:

            content = doc.page_content

            if content not in seen:
                unique_results.append(doc)
                seen.add(content)

        final_results = unique_results[:k]

        self.last_combined_results = final_results

        return final_results