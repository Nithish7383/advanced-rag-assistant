def search_documents(vector_store, query, k=3):
    """
    Search the FAISS vector database for the most relevant document chunks.

    Parameters
    ----------
    vector_store : FAISS
        Vector database containing embeddings
    query : str
        User question
    k : int
        Number of results to retrieve

    Returns
    -------
    list
        List of retrieved document chunks
    """

    results = vector_store.similarity_search(query, k=k)

    return results