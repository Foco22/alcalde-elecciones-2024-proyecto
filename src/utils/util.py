from dotenv import load_dotenv
from llama_parse import LlamaParse
from openai import OpenAI
import pandas as pd 
import os
from groq import Groq
import uuid
from pymongo import MongoClient
import numpy as np
import streamlit as st
import concurrent.futures
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import Settings
import pickle

#openai.api_key = os.getenv('OPENAI_API_KEY')

load_dotenv()

def cosine_similarity(vec1, vec2):
    """Calculate the cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def find_closest_embeddings_in_faiss(user_embedding, name_file, file_path='src/files/embeddings/nodes_data.pkl', top_n=3):

    with open(file_path, 'rb') as file:
        documents = pickle.load(file)
    
    similar_documents = []        
    for document in documents:
        if document.metadata['file_name'] == name_file:
            stored_embedding = document.embedding
            similarity = cosine_similarity(user_embedding, stored_embedding)
            similar_documents.append((similarity, document))
    
    similar_documents.sort(key=lambda x: x[0], reverse=True)
    top_similar_documents = similar_documents[:top_n]
    if top_similar_documents:
        return {
            'status': True,
            'top_similar_documents': [
                {
                    'document': doc.text,
                    'metadata': {key: value for key, value in doc.metadata.items() if key not in ('summary_embedding', 'summary')},
                    'similarity_score': similarity
                } for similarity, doc in top_similar_documents
            ]
        }
    else:
        return {
            'status': False,
            'top_similar_documents': [],
            'message': 'No documents found'
        }


def find_closest_embeddings_in_mongodb(user_embedding, name_file, top_n=3):

    username = os.environ.get("USER_NAME_COSMO")
    password = os.environ.get("PASSWORD_COSMO")
    host = os.environ.get("HOST_COSMO")
    database = os.environ.get("DATABASE_COSMO")
    collection_name = os.environ.get("COLLECTION_COSMO")

    uri = f"mongodb+srv://{username}:{password}@{host}/{database}?retryWrites=true&w=majority"
    client = MongoClient(uri)
    db = client[database]
    collection = db[collection_name]
    similar_documents = []
    documents = collection.find({}, {"embedding": 1, "text": 1, "metadata": 1})  # Retrieve embeddings, text, and metadata

    for document in documents:
        if document['metadata']['file_name'] == name_file:
            stored_embedding = document['embedding']
            similarity = cosine_similarity(user_embedding, stored_embedding)
            similar_documents.append((similarity, document))
    
    similar_documents.sort(key=lambda x: x[0], reverse=True)
    top_similar_documents = similar_documents[:top_n]

    if top_similar_documents:
        return {
            'status': True,
            'top_similar_documents': [
                {
                    'document': doc['text'],
                    'metadata': {key: value for key, value in doc['metadata'].items() if key not in ('summary_embedding', 'summary')},
                    'similarity_score': similarity
                } for similarity, doc in top_similar_documents
            ]
        }
    else:
        return {
            'status': False,
            'top_similar_documents': [],
            'message': 'No documents found'
        }

