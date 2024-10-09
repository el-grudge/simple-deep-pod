import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Import Hugging Face tokenizer and OpenAI tokenizer (if applicable)
from transformers import T5Tokenizer, T5Model
import torch

class Index:
    """
    A simple search index using TF-IDF, Hugging Face Tokenizer, or other tokenizers like OpenAI for text fields and
    exact matching for keyword fields.

    Attributes:
        text_fields (list): List of text field names to index.
        keyword_fields (list): List of keyword field names to index.
        vectorizers (dict): Dictionary of TfidfVectorizer or tokenizers instances for each text field.
        keyword_df (pd.DataFrame): DataFrame containing keyword field data.
        text_matrices (dict): Dictionary of TF-IDF matrices or embedding matrices for each text field.
        docs (list): List of documents indexed.
        use_tfidf (bool): Whether to use TF-IDF or another tokenizer.
    """
    def __init__(self, index_name, text_fields, keyword_fields, vectorizer_params={}, tokenizer_name='tfidf'):
        """
        Initializes the Index with specified text and keyword fields and tokenizer type.

        Args:
            text_fields (list): List of text field names to index.
            keyword_fields (list): List of keyword field names to index.
            vectorizer_params (dict): Optional parameters to pass to TfidfVectorizer or tokenizers.
            tokenizer_name (str): Type of tokenizer to use: 'tfidf', 't5', or 'openai'.
        """
        self.index_name = index_name
        self.text_fields = text_fields
        self.keyword_fields = keyword_fields
        self.tokenizer_name = tokenizer_name

        if tokenizer_name == 'tfidf':
            self.vectorizers = {field: TfidfVectorizer(**vectorizer_params) for field in text_fields}
        elif tokenizer_name == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained("t5-small")
            self.model = T5Model.from_pretrained("t5-small")
            self.vectorizers = {field: self._huggingface_embed for field in text_fields}
        elif tokenizer_name == 'openai':
            # Placeholder for OpenAI tokenizer initialization
            # You can replace this with the actual OpenAI tokenizer setup
            self.vectorizers = {field: self._openai_embed for field in text_fields}

        self.keyword_df = None
        self.text_matrices = {}
        self.docs = []

    def _huggingface_embed(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()

    def _openai_embed(self, text):
        # Replace with the actual OpenAI embedding function.
        # This is a placeholder. The OpenAI API can provide embeddings for the text.
        # Assuming you have some `openai` function to get embeddings.
        return openai.Embedding.create(input=text)["data"]

    def fit(self, docs):
        """
        Fits the index with the provided documents.

        Args:
            docs (list of dict): List of documents to index. Each document is a dictionary.
        """
        self.docs = docs
        keyword_data = {field: [] for field in self.keyword_fields}

        for field in self.text_fields:
            texts = [doc.get(field, '') for doc in docs]

            if self.tokenizer_name == 'tfidf':
                self.text_matrices[field] = self.vectorizers[field].fit_transform(texts)
            else:
                embeddings = np.vstack([self.vectorizers[field](text) for text in texts])
                self.text_matrices[field] = embeddings

        for doc in docs:
            for field in self.keyword_fields:
                keyword_data[field].append(doc.get(field, ''))

        self.keyword_df = pd.DataFrame(keyword_data)

        return self

    def search(self, query, filter_dict={}, boost_dict={}, num_results=10):
        """
        Searches the index with the given query, filters, and boost parameters.

        Args:
            query (str): The search query string.
            filter_dict (dict): Dictionary of keyword fields to filter by. Keys are field names and values are the values to filter by.
            boost_dict (dict): Dictionary of boost scores for text fields. Keys are field names and values are the boost scores.
            num_results (int): The number of top results to return. Defaults to 10.

        Returns:
            list of dict: List of documents matching the search criteria, ranked by relevance.
        """
        if self.tokenizer_name == 'tfidf':
            query_vecs = {field: self.vectorizers[field].transform([query]) for field in self.text_fields}
        else:
            query_vecs = {field: self.vectorizers[field](query) for field in self.text_fields}

        scores = np.zeros(len(self.docs))

        # Compute cosine similarity for each text field and apply boost
        for field, query_vec in query_vecs.items():
            sim = cosine_similarity(query_vec, self.text_matrices[field]).flatten()
            boost = boost_dict.get(field, 1)
            scores += sim * boost

        # Apply keyword filters
        for field, value in filter_dict.items():
            if field in self.keyword_fields:
                mask = self.keyword_df[field] == value
                scores = scores * mask.to_numpy()

        # Use argpartition to get top num_results indices
        top_indices = np.argpartition(scores, -num_results)[-num_results:]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        # Filter out zero-score results
        top_docs = [self.docs[i] for i in top_indices if scores[i] > 0]

        return top_docs
