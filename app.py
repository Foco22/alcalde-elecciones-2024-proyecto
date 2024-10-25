import os
from pathlib import Path
import streamlit as st
from src.services.router import AgentRetrieval
import asyncio

# Set paths for directories
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Set page config for the Streamlit app
st.set_page_config(page_title="RAG Chatbot")
st.title("Candidaturas a Gobernador de Elecciones 'Octubre 2024'")

def app():
    # Sidebar for OpenAI API key input
    st.sidebar.title("Configuración")
    
    # Input field for the API key (use password_input to hide the key)
    api_key = st.sidebar.text_input("Ingresa tu OpenAI API Key:", type="password")
    
    # Store the API key in session state
    if api_key:
        st.session_state['OPENAI_API_KEY'] = api_key
    
    # Check if the API key is present
    if 'OPENAI_API_KEY' not in st.session_state:
        st.warning("Por favor, ingresa tu OpenAI API Key para continuar.")
        return  # Stop execution until the API key is provided
    
    # Sidebar with options
    st.sidebar.title("Opciones")
    option = st.sidebar.radio("Elige una opción:", ('Metropolitana (Santiago)', 
                                                    'Valparaiso',
                                                    'Arica y Parinacota',
                                                    'Tarapaca',
                                                    'Antofagasta',
                                                    'Atacama',
                                                    'Coquimbo',
                                                    'O’higgins',
                                                    'Maule',
                                                    'Nuble',
                                                    'Los Rios',
                                                    'Aysen',
                                                    'Los Lagos',
                                                    'Magallanes',
                                                    'La Araucania',
                                                    'Bio Bio',
                                                    ))

    # Check if the selected option has changed, and reset session if necessary
    if "selected_option" in st.session_state:
        if st.session_state.selected_option != option:
            st.session_state.messages = []  # Reset messages if the option changes
    else:
        st.session_state.selected_option = option  # Initialize the selected option

    # Update the selected option in session state
    st.session_state.selected_option = option
    
    # Initialize session state for storing messages if not already initialized
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display previous messages in the chat interface
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    
    # Input from user
    if query := st.chat_input():
        st.chat_message("human").write(query)
        
        # Send the selected option to AgentRetrieval along with message and chat history
        chat_history = st.session_state.messages  # Assuming chat history is stored in session_state
        agent = AgentRetrieval(query, chat_history, option, api_key=st.session_state['OPENAI_API_KEY'])
        response = asyncio.run(agent.agent_response_retrieval_message())
        response_llm_final_message = response['message']
        response_token_info = response['token_info']

        # Display AI response
        st.chat_message("ai").write(response_llm_final_message)
        st.markdown(response_token_info, unsafe_allow_html=True)
        
        # Save the conversation in session state
        st.session_state.messages.append((query, response_llm_final_message))

# Run the app
if __name__ == '__main__':
    app()
