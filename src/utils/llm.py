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


def get_llm_groq_response(context, chat_history, system_prompt, message, model="mixtral-8x7b-32768"):

    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY"),
    )
    
    system_message = {
        "role": "system",
        "content": system_prompt
    }

    user_message = {
        "role": "user",
        "content": message
    }

    chat_completion = client.chat.completions.create(
        messages=[system_message, user_message],
        model=model,  # Using the appropriate model
    )        
    return {
        'status': True,
        'message': chat_completion.choices[0].message.content
    }


def get_llm_openai_response(prompt_system, api_key, model='gpt-4o-mini'):
    """
    Generates a response from OpenAI's language model based on the input message,
    prompt system instructions, and chat history.

    Args:
        message (str): The user's input message.
        prompt_system (str): The system prompt or initial instruction for the model.
        chat_history (list): The list of previous chat messages with roles.
        model_openai (str): The OpenAI model to use (e.g., "gpt-3.5-turbo").

    Returns:
        str: The response generated by the OpenAI model.
    """
    try:
        # Construct the message list for the OpenAI API call
        message_llm = [{"role": "system", "content": prompt_system}]

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        # Create a chat completion request
        response = client.chat.completions.create(
            model=model,
            messages=message_llm
        )
        return {
            'status': True, 
            'message': response.choices[0].message.content,
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens
        }
    except:
        return {
            'status': False, 'message': 'Disculpa, lamentablemente tuvimos un problema con el modelo de lenguage. ¿Podria revisar si la API key esta correcta o volver a preguntar?',
            'prompt_tokens': 0,
            'completion_tokens': 0
        }        

