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
import json
from src.utils.llm import get_llm_openai_response
from src.utils.embeddings import get_openai_embedding
from src.utils.util import find_closest_embeddings_in_faiss
import ast
import time
import asyncio

load_dotenv()

class AgentRetrieval:
    
    def __init__(self, message, chat_history, option, api_key):
        self.message = message
        self.chat_history = chat_history
        self.path_file_excel='src/files/excel/base.ods'
        self.df = pd.read_excel(self.path_file_excel)
        self.option = option
        self.api_key = api_key

    def agent_get_relevant_pdfs(self, rag, agent_person_user_question=None):

        # Return all unique PDFs if RAG is False or agent_person_user_question is None or empty
        if not rag or not agent_person_user_question:
            filtered_df = self.df[self.df['Region'] == self.option]
            return list(filtered_df['PDF'].unique())

        # If agent_person_user_question is a string that looks like a list, convert it safely
        if isinstance(agent_person_user_question, str):
            try:
                # Convert string to list if it's a valid list string
                agent_person_user_question = ast.literal_eval(agent_person_user_question)
            except (ValueError, SyntaxError):
                raise ValueError("Invalid input: 'agent_person_user_question' is not a valid list or list-like string.")

        # If it's a list, filter the DataFrame based on the list of names
        if isinstance(agent_person_user_question, list):
            filtered_df = self.df[self.df['Region'] == self.option]
            filtered_df = filtered_df[filtered_df['Nombre'].isin(agent_person_user_question)]
            return list(filtered_df['PDF'].unique())

        # Raise an error if the input is neither a list nor a valid list-like string
        raise ValueError("Invalid input for 'agent_person_user_question'. Expected a list or a valid list-like string.")


    def get_prompts_systems(self, path_file='src/files/prompts/prompts_systems.json'):

        with open(path_file, 'r') as file:
            prompt_system_json = json.load(file)

        filtered_df = self.df[self.df['Region'] == self.option]
        list_name_files = list(filtered_df['Nombre'].unique())

        prompt_system_rag_definition = prompt_system_json['prompt_system_rag_definition']   
        prompt_system_rag_definition = prompt_system_rag_definition.replace('chat_history', str(self.chat_history))
        #prompt_system_rag_definition = prompt_system_rag_definition.replace('user_message', str(self.message))

        prompt_system_to_defined_persons = prompt_system_json['prompt_system_to_defined_persons']
        prompt_system_to_defined_persons = prompt_system_to_defined_persons.replace('chat_history', str(self.chat_history))
        #prompt_system_to_defined_persons = prompt_system_to_defined_persons.replace('user_message', str(self.message))
        prompt_system_to_defined_persons = prompt_system_to_defined_persons.replace('list_name_files', str(list_name_files))
        
        prompt_system_markdown_question = prompt_system_json['prompt_system_markdown_question']
        #prompt_system_markdown_question = prompt_system_markdown_question.replace('user_question', self.message)

        prompt_system_best_retrieval_information = prompt_system_json['prompt_system_best_retrieval_information']

        prompt_system_final_answer = prompt_system_json['prompt_system_final_answer']  
        prompt_system_final_answer = prompt_system_final_answer.replace('chat_history', str(self.chat_history))
        #prompt_system_final_answer = prompt_system_final_answer.replace('user_message', str(self.message))

        pricing_prompt_token = float(prompt_system_json['pricing_prompt_token'])
        pricing_completion_token = float(prompt_system_json['pricing_completion_token'])

        prompt_system_redefine_message = prompt_system_json['prompt_system_redefine_message'] 
        prompt_system_redefine_message = prompt_system_redefine_message.replace('chat_history', str(self.chat_history))
        prompt_system_redefine_message = prompt_system_redefine_message.replace('user_message', str(self.message))

        return {
            'prompt_system_rag_definition': prompt_system_rag_definition,
            'prompt_system_to_defined_persons': prompt_system_to_defined_persons,
            'prompt_system_markdown_question': prompt_system_markdown_question,
            'prompt_system_best_retrieval_information': prompt_system_best_retrieval_information,
            'prompt_system_final_answer': prompt_system_final_answer,
            'pricing_prompt_token': pricing_prompt_token,
            'pricing_completion_token': pricing_completion_token,
            'prompt_system_redefine_message': prompt_system_redefine_message

        }

    def agent_router_rag(self, prompt_system_rag_definition, prompt_system_to_defined_persons):
        
        total_prompt_token = 0
        total_completion_token = 0

        filtered_df = self.df[self.df['Region'] == self.option]
        list_name_collection = list(filtered_df['Nombre'].unique())
        
        response_message_rag = get_llm_openai_response(prompt_system_rag_definition, self.api_key)
        response_message_rag_prompt_token = response_message_rag['prompt_tokens']
        response_message_rag_completion_token = response_message_rag['completion_tokens']
        total_prompt_token = total_prompt_token + response_message_rag_prompt_token
        total_completion_token = total_completion_token + response_message_rag_completion_token

        if response_message_rag['message'] in ('True', True):            
            response_message_name_list = get_llm_openai_response(prompt_system_to_defined_persons, self.api_key)
            response_message_name_list_prompt_token = response_message_name_list['prompt_tokens']
            response_message_name_list_completion_token = response_message_name_list['completion_tokens']
            total_prompt_token = total_prompt_token + response_message_name_list_prompt_token
            total_completion_token = total_completion_token + response_message_name_list_completion_token            

            if  response_message_name_list['message'] not in ('False', False):
                return {
                    'status': True,
                    'rag': True,
                    'person_user_question': response_message_name_list['message'],
                    'prompt_tokens': total_prompt_token, 
                    'completion_tokens':total_completion_token
                }
            else:
                return {
                    'status': True,
                    'rag': True,
                    'person_user_question': list_name_collection,
                    'prompt_tokens': total_prompt_token, 
                    'completion_tokens':total_completion_token
                }

        return {
            'status': True,
            'rag': False,
            'person_user_question': list_name_collection,
            'prompt_tokens': total_prompt_token, 
            'completion_tokens':total_completion_token
        }

    async def agent_retrieval_sql_information(self, person_user_question, prompt_system):

        if not isinstance(person_user_question, list):
            # Use `ast.literal_eval()` for safer evaluation of strings that represent lists
            try:
                lista_name = ast.literal_eval(person_user_question)
            except (ValueError, SyntaxError):
                raise ValueError("Invalid input for person_user_question. Expected a list or a valid string representation of a list.")
        else:
            lista_name = person_user_question

        filtered_df = self.df[self.df['Region'] == self.option]
        filtered_df = filtered_df[filtered_df['Nombre'].isin(lista_name)]
        
        if not filtered_df.empty:
            markdown_table = filtered_df.to_markdown(index=False)
        else:
            markdown_table = "No matching records found."

        # Insert the markdown_table into the prompt system
        filled_prompt = prompt_system.replace("tabla_markdown", markdown_table)

        # Call the LLM to get the response
        response_llm_general_information = get_llm_openai_response(filled_prompt, self.api_key)
        prompt_tokens = response_llm_general_information['prompt_tokens']
        completion_tokens = response_llm_general_information['completion_tokens']

        return {
            'retrieval_information': response_llm_general_information.get('message', 'No message received.'),
            'prompt_tokens': prompt_tokens, 
            'completion_tokens':completion_tokens
        }

    async def agent_retrieval_docs_information(self, agent_rag, agent_person_user_question, user_embedding, total_closest_embeddings=[]):

        with concurrent.futures.ThreadPoolExecutor() as executor:

            names_context = self.agent_get_relevant_pdfs(agent_rag, agent_person_user_question)
            futures = [executor.submit(find_closest_embeddings_in_faiss, user_embedding, name_file) for name_file in names_context]
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    total_closest_embeddings = total_closest_embeddings + result['top_similar_documents']
                except Exception as e:
                    total_closest_embeddings = total_closest_embeddings + []


        texto_closet_embeddings = '# Embeddings más cercanos\n\n'
        for row_embedding in total_closest_embeddings:

            texto_closet_embeddings += '### Metadata\n'
            for key, value in row_embedding["metadata"].items():
                texto_closet_embeddings += f'- **{key}**: {value}\n'

            texto_closet_embeddings += '## Embedding Document\n'
            texto_closet_embeddings += f'{row_embedding["document"]}\n\n'
            texto_closet_embeddings += '\n---\n\n'

        return {
            'retrieval_information': texto_closet_embeddings
        }

    def agent_choose_retrieval_information(self, doc_retrieval, sql_retrieval, message, prompt_system):

        prompt_system = prompt_system.replace('sql_retrieval_information', sql_retrieval)
        prompt_system = prompt_system.replace('doc_retrieval_information', doc_retrieval)
        prompt_system = prompt_system.replace('user_message', message)
        response = get_llm_openai_response(prompt_system, self.api_key)
        response_message = response['message']
        prompt_tokens = response['prompt_tokens']
        completion_tokens = response['completion_tokens']
    
        return {
            'message': response_message,
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens

        }

    async def agent_response_retrieval_message(self):
        
        ### Parameters message ### 
        total_prompt_token = 0
        total_completion_token = 0

        ### Step 1: Retrieve system prompts ###
        start_time = time.time()
        prompt_systems = self.get_prompts_systems()
        prompt_system_rag_definition = prompt_systems['prompt_system_rag_definition']
        prompt_system_to_defined_persons = prompt_systems['prompt_system_to_defined_persons']
        prompt_system_markdown_question = prompt_systems['prompt_system_markdown_question']
        prompt_system_best_retrieval_information = prompt_systems['prompt_system_best_retrieval_information']
        prompt_system_final = prompt_systems['prompt_system_final_answer']
        pricing_prompt_token = float(prompt_systems['pricing_prompt_token'])
        pricing_completion_token = float(prompt_systems['pricing_completion_token'])
        prompt_system_redefine_message = prompt_systems['prompt_system_redefine_message']
        print(f"Step 2 - Retrieve system prompts took: {time.time() - start_time:.2f} seconds")

        ### Step 2: Redefine message ###
        response_llm_redefined = get_llm_openai_response(prompt_system_redefine_message, self.api_key, model='gpt-3.5-turbo')
        response_llm_prompt_redefined = response_llm_redefined['prompt_tokens']
        response_llm_completion_redefined = response_llm_redefined['completion_tokens']
        message_redefined = response_llm_redefined['message']
        total_prompt_token = total_prompt_token + response_llm_prompt_redefined
        total_completion_token = total_completion_token + response_llm_completion_redefined  

        ### Step 3: Detect if the user is asking about region ###
        list_region = (self.df['Region'].unique())
        filtered_df = self.df[self.df['Region'] == self.option]
        list_name_files = list(filtered_df['Nombre'].unique())
        prompt_system_start = """Tu eres un router en una conversacion con un usuario. 
        Tu mision es determinar si el mensaje del usuario esta relacionado a la region establecida.
        Basado en el mensaje del usuario: {}
        Junto con el historial de conversacion: {}
        Las regiones de la base de datos son: {}
        La region establecida es: {}
        Los candidatos a gobernador en la region establecida son: {}
        Si el usuario esta preguntando sobre otra region, debes contestar False. Por el contrario, debe responder True.
        Si con la informacion no puede determinar exacta la region a que se refiere el usuario, responde True en todo esos casos.""".format(message_redefined, self.chat_history, str(list_region), self.option, list_name_files)
        response_first = get_llm_openai_response(prompt_system_start, self.api_key)['message']       

        if response_first in ('False', False):
            
            #prompt_system_final = prompt_system_final.replace('user_message', message_redefined)
            prompt_system_final = """Tu eres un asistente de las elecciones a Gobernador de Chile en Octubre del 2024. Tu mision es de manera politica y amable pedile al usuario que pregunte sobre temas relacionado a los candidatos de la region establecida, no otra region en Chile.
            La region establecida es: {}
            La pregunta del usuario es: {}
            El Historial de conversacion es: {}""".format(self.option, message_redefined, self.chat_history)

            response_llm = get_llm_openai_response(prompt_system_final, self.api_key)
            response_llm_prompt = response_llm['prompt_tokens']
            response_llm_completion = response_llm['completion_tokens']
            response_llm_final = response_llm['message']
            total_prompt_token = total_prompt_token + response_llm_prompt
            total_completion_token = total_completion_token + response_llm_completion    
            total_pricing_openai = total_prompt_token*pricing_prompt_token + total_completion_token*pricing_completion_token
            # Store the token information separately and adjust the styling
            token_info = (
                f"<div style='color:black; font-size:12px; text-align:left; margin-top:10px;'>"
                f"Prompt Tokens: {total_prompt_token}<br>"
                f"Completion Tokens: {total_completion_token}<br>"
                f"Pricing OpenAI: {total_pricing_openai:.5f}<br>"
                f"</div>"
            )
            print(f"Final LLM response generation took: {time.time() - start_time:.2f} seconds")
            return {
                'status': True,
                'message': response_llm_final,
                'token_info': token_info,
                'prompt_tokens': total_prompt_token,
                'completion_tokens': total_completion_token
            }


        ### Step 4: User Embedding: Transform message to embedding ###
        start_time = time.time()
        user_embedding = get_openai_embedding(message_redefined, self.api_key)
        print(f"Step 1 - User Embedding took: {time.time() - start_time:.2f} seconds")

        ### Step 5: Route decision - Apply RAG or not ###
        prompt_system_rag_definition = prompt_system_rag_definition.replace('user_message', message_redefined)
        prompt_system_to_defined_persons = prompt_system_to_defined_persons.replace('user_message', message_redefined)

        start_time = time.time()
        agent_response_retrieval = self.agent_router_rag(prompt_system_rag_definition, prompt_system_to_defined_persons)
        agent_rag = agent_response_retrieval['rag']
        agent_person_user_question = agent_response_retrieval['person_user_question']
        agent_response_retrieval_prompt_token = agent_response_retrieval['prompt_tokens']
        agent_response_retrieval_completion_token = agent_response_retrieval['completion_tokens']
        total_prompt_token = total_prompt_token + agent_response_retrieval_prompt_token
        total_completion_token = total_completion_token + agent_response_retrieval_completion_token
        print(f"Step 3 - Route decision took: {time.time() - start_time:.2f} seconds")

        if agent_rag in ('True', True):

            ### Step 6: Extract Docs and SQL Information ###
            start_time = time.time()

            # Run both doc retrieval and SQL retrieval concurrently
            prompt_system_markdown_question = prompt_system_markdown_question.replace('user_message', message_redefined)
            docs_task = self.agent_retrieval_docs_information(agent_rag, agent_person_user_question, user_embedding)
            sql_task = self.agent_retrieval_sql_information(agent_person_user_question, prompt_system_markdown_question)
            
            # Use asyncio.gather to run them asynchronously
            docs_retrieval, sql_retrieval = await asyncio.gather(docs_task, sql_task)

            sql_retrieval_info = sql_retrieval['retrieval_information']
            docs_retrieval_info = docs_retrieval['retrieval_information']
            sql_retrieval_prompt_tokens = sql_retrieval['prompt_tokens']
            sql_retrieval_completion_tokens = sql_retrieval['completion_tokens']
            total_prompt_token += sql_retrieval_prompt_tokens
            total_completion_token += sql_retrieval_completion_tokens
            print(f"Step 4 - Extract Docs and SQL Information took: {time.time() - start_time:.2f} seconds")

            ### Step 7: Choose best retrieval information (Docs or SQL) ###
            start_time = time.time()
            best_retrieval = self.agent_choose_retrieval_information(
                docs_retrieval_info, sql_retrieval_info, self.message, prompt_system_best_retrieval_information
            )
            print(f"Step 5 - Choose best retrieval information took: {time.time() - start_time:.2f} seconds")

            best_retrieval_choice = best_retrieval['message']
            best_retrieval_prompt = best_retrieval['prompt_tokens']
            best_retrieval_completion = best_retrieval['completion_tokens']
            total_prompt_token = total_prompt_token + best_retrieval_prompt
            total_completion_token = total_completion_token + best_retrieval_completion    

            # Use SQL or Docs based on the best retrieval choice
            chosen_retrieval_info = sql_retrieval_info if best_retrieval_choice == 'SQL' else docs_retrieval_info

            ### Step 8: Generate final LLM response based on chosen context ###
            start_time = time.time()
            prompt_system_final = prompt_system_final.replace('retrieval_information', str(chosen_retrieval_info))
            print(f"Step 6 - Generate final LLM response took: {time.time() - start_time:.2f} seconds")
        else:
            ### Handle when RAG is not needed ###
            start_time = time.time()
            chosen_retrieval_info = []  # Add logic to calculate embeddings if needed
            prompt_system_final = prompt_system_final.replace('retrieval_information', str(chosen_retrieval_info))
            print(f"Step 6 (No RAG) - Generate final LLM response took: {time.time() - start_time:.2f} seconds")

        ### Get the final response from the LLM ###
        start_time = time.time()
        prompt_system_final = prompt_system_final.replace('user_message', message_redefined)
        response_llm = get_llm_openai_response(prompt_system_final, self.api_key)
        response_llm_prompt = response_llm['prompt_tokens']
        response_llm_completion = response_llm['completion_tokens']
        response_llm_final = response_llm['message']
        total_prompt_token = total_prompt_token + response_llm_prompt
        total_completion_token = total_completion_token + response_llm_completion    
        total_pricing_openai = total_prompt_token*pricing_prompt_token + total_completion_token*pricing_completion_token
        # Store the token information separately and adjust the styling
        token_info = (
            f"<div style='color:black; font-size:12px; text-align:left; margin-top:10px;'>"
            f"Prompt Tokens: {total_prompt_token}<br>"
            f"Completion Tokens: {total_completion_token}<br>"
            f"Pricing OpenAI: {total_pricing_openai:.5f}<br>"
            f"</div>"
        )
        print(f"Final LLM response generation took: {time.time() - start_time:.2f} seconds")
        return {
            'status': True,
            'message': response_llm_final,
            'token_info': token_info,
            'prompt_tokens': total_prompt_token,
            'completion_tokens': total_completion_token
        }


