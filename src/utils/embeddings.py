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

load_dotenv()

def get_openai_embedding(text, api_key, model="text-embedding-3-small"):

    os.environ["OPENAI_API_KEY"] = api_key
    embed_model = OpenAIEmbedding(model=model)
    embeddings = embed_model.get_text_embedding(
        text
    )
    return embeddings
