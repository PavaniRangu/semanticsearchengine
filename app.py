# Install Streamlit
#!pip install streamlit

# Import necessary libraries
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer

# Load BERT model for sentence embeddings
model = SentenceTransformer('bert-base-nli-mean-tokens')

# Define text preprocessing function (you can replace it with your own)
def text_preprocessing(text):
    return text.lower()  # Example: Convert text to lowercase

# Define function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

# Define main function to search documents
def search_documents(user_query, document_embeddings, data, top_k=10):
    # Preprocess user query
    preprocessed_query = text_preprocessing(user_query)

    # Encode query into embedding
    query_embedding = model.encode([preprocessed_query])[0]

    # Calculate cosine similarity between query and document embeddings
    similarity_scores = np.dot(query_embedding, document_embeddings.T) / (np.linalg.norm(query_embedding) * np.linalg.norm(document_embeddings, axis=1))

    # Rank documents based on similarity scores
    ranked_documents = sorted(enumerate(similarity_scores), key=lambda x: x[1], reverse=True)[:top_k]

    # Return top-ranked documents
    top_documents = [(data.iloc[idx]['name'], score) for idx, score in ranked_documents]
    return top_documents

# Load your data and precompute document embeddings here

# Streamlit app
def main():
    st.title('Search Engine')

    # Get user query
    user_query = st.text_input('Enter your search query')

    # Search documents when user submits query
    if st.button('Search'):
        # Retrieve top documents
        top_documents = search_documents(user_query, document_embeddings, data)

        # Display top documents
        st.subheader('Top 10 Results:')
        for i, (document_name, score) in enumerate(top_documents):
            st.write(f'{i+1}. {document_name} (Similarity Score: {score})')

if __name__ == '__main__':
    main()
