import streamlit as st
from utils import search_podcasts
from openai import OpenAI
import replicate
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import AuthenticationException, ConnectionError
from ingest import create_index
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer
import chromadb

def update_session(**kwargs):
    for k, v in kwargs.items():
        st.session_state[k] = v

# Function to apply settings and reset states
def apply_settings():
    update_session(settings_applied=True)

def reset_settings():
    st.session_state.clear()

def choose_podcast_option():
    help_message = "Recommended choice: Try a sample.\n\nUse the iTunes URL option if you want to interact with a specific episode.\n\nUse the name option if you want to interact with a podcast's most recent episode."

    st.header("Select a podcast source", help=help_message)
    episode_option = st.radio(
        "Choose your option:",
        options = (
            "1. Try a sample",
            "2. Provide the iTunes URL for a specific podcast episode",
            "3. Provide a name of a podcast to explore its most recent episode"
            ),
        index=None,
    )
    update_session(episode_option_selected=False)
    if episode_option == "1. Try a sample":
        update_session(episode_option_selected=True, episode_option=episode_option)
    elif episode_option == "2. Provide the iTunes URL for a specific podcast episode":
        episode_url = st.text_input("Enter the iTunes URL of the episode you want:")
        update_session(episode_option_selected=True, episode_option=episode_option, episode_url=episode_url)
    elif episode_option == "3. Provide a name of a podcast to explore its most recent episode":
        term = st.text_input("Enter a search term for podcasts:")
        try:
            if term != '':
                found_podcasts = search_podcasts(term)
                if found_podcasts['status'] == 'Fail':
                    raise Exception
                else:
                    podcast_names = [f"{podcast['collectionName']} by {podcast['artistName']}" for podcast in found_podcasts['podcasts']]
                    selected_podcast = st.selectbox("Select a podcast:", podcast_names)
                    selected_index=podcast_names.index(selected_podcast)
                    update_session(episode_option_selected=True, episode_option=episode_option, found_podcasts=found_podcasts['podcasts'], selected_index=selected_index)
        except Exception:
            st.warning("Please enter a valid search term.")

def choose_encoder():
    help_message = "Recommended choice: T5.\n\nThe OpenAI tokenizer is less performant.\n\n⚠️ Please note that an API key will be required in order to use the OpenAI tokenizer, which may incur an additional cost. For more details see: https://openai.com/index/openai-api/"
    st.subheader('Select a sentence encoder', help=help_message)
    sentence_encoder = st.radio(
        "Choose your option:",
        options = (
            "1. T5",
            "2. OpenAI"
        ),
        index=None
    )
    update_session(sentence_encoder_selected=False)
    if sentence_encoder == "1. T5":
        encoder=SentenceTransformer("sentence-transformers/sentence-t5-base")
        update_session(sentence_encoder_selected=True, sentence_encoder=sentence_encoder, encoder=encoder)
    elif sentence_encoder == "2. OpenAI":
        embedding_model = "text-embedding-3-large"
        openai_api_key = st.text_input("OpenAI API Key", key="file_oa_api_key", type="password")
        if openai_api_key != '':
            try:
                oa_embedding_client = OpenAI(api_key=openai_api_key)
                response = oa_embedding_client.models.list()
                update_session(sentence_encoder_selected=True, sentence_encoder=sentence_encoder, embedding_client=oa_embedding_client, embedding_model=embedding_model)
            except:
                st.warning("Invalid API key. Please provide a valid API token.")

def choose_transcription_method():
    help_message = "Recommended choice: Replicate.\n\nUse the local transcription if you have access to a GPU and can deploy this app locally.\n\n⚠️ Please note that an API key will be required in order to use the Replicate transcriber. For more details see: https://replicate.com/home"
    st.subheader('Select a transcription method', help=help_message)
    if st.session_state.get('episode_option', False):
        if st.session_state['episode_option'] != "1. Try a sample":
            transcription_method = st.radio(
                "Choose your option:",
                options = (
                    "1. Replicate",
                    "2. Local transcription",
                ),
                index=None,
            )
            update_session(transcription_method_selected=False)
            if transcription_method=="1. Replicate":
                replicate_api_key = st.text_input("Replicate API Key", key="file_replicate_api_key", type="password")            
                if replicate_api_key != '':
                    try:
                        replicate_client = replicate.Client(api_token=replicate_api_key)
                        response = replicate_client.models.list()
                        update_session(transcription_method_selected=True, transcription_method=transcription_method, transcription_client=replicate_client)
                    except:
                        st.warning("Invalid API key. Please provide a valid API token.")
            elif transcription_method=="2. Local transcription":
                update_session(transcription_method_selected=True, transcription_method=transcription_method)
        else:
            st.success("The sample podcast doesn't require a transcription method.")
            update_session(transcription_method_selected=True)

def choose_vector_db():
    help_message = "Recommended choice: ChromaDB.\n\nUse Minsearch if you want a quick preview of the app.\n\nElasticsearch is less performant.\n\n⚠️ Please note that an API key will be required in order to use the Elasticsearch vectorb database. For more details see: https://elasticsearch-py.readthedocs.io/en/v8.10.1/quickstart.html"
    st.subheader('Select a vector database', help=help_message)
    st.session_state['index_name'] = "podcast-transcriber"
    vector_db = st.radio(
        "Choose your option:",
        options = (
            "1. Minsearch",
            "2. Elasticsearch",
            "3. ChromaDB"
        ),
        index=None,
    )
    update_session(vector_db_selected=False)
    if vector_db=="1. Minsearch":
        update_session(vector_db=vector_db)
        update_session(index=create_index(**st.session_state))
        update_session(vector_db_selected=True, index_created=True)
        st.success(f"Index {st.session_state['index'].index_name} was created successfully.")
    elif vector_db=="2. Elasticsearch":
        elasticsearch_api_key = st.text_input("Elasticsearch API Key", key="file_es_api_key", type="password")
        elasticsearch_cloud_id = st.text_input("Elasticsearch Cloud ID", key="file_es_cloud_id", type="password")
        if elasticsearch_api_key != '' and elasticsearch_cloud_id != '':
            try:
                es_client = Elasticsearch(cloud_id=elasticsearch_cloud_id, api_key=elasticsearch_api_key)
                response = es_client.cluster.health()
                update_session(vector_db=vector_db, vector_db_client=es_client)
                update_session(index=create_index(**st.session_state))
                update_session(vector_db_selected=True, index_created=True)
                st.success(f"Index {[k for k,v in st.session_state['index'].items()][0]} was created successfully.")
            except AuthenticationException:
                st.warning("Invalid API key or Cloud ID. Please provide a valid tokens.")
            except ConnectionError:
                st.warning("Connection error. Could not connect to the cluster.")
            except Exception as e:
                st.warning(f"An error occurred: {e}")
    elif vector_db=="3. ChromaDB":
        chroma_client = chromadb.PersistentClient(path="./chroma_db")
        update_session(vector_db=vector_db, vector_db_client=chroma_client)
        update_session(index=create_index(**st.session_state))
        update_session(vector_db_selected=True, index_created=True)
        st.success(f"Index {st.session_state['vector_db_client'].list_collections()[0].name} was created successfully.")

def choose_llm():
    help_message = "Recommended choice: FLAN-5.\n\nUse GPT-4o if want to interact with a more conversant LLM.\n\n⚠️ Please note that an API key will be required in order to use GPT-4o, which may incur an additional cost. For more details see: https://openai.com/index/openai-api/"
    st.subheader('Select an LLM', help=help_message)
    llm_option = st.radio(
        "Choose your option:",
        options = (
            "1. GPT-4o",
            "2. FLAN-5"
        ),
        index=None,
    )
    update_session(llm_option_selected=False)
    if llm_option == "1. GPT-4o":
        if st.session_state['sentence_encoder'] != "2. OpenAI":
            openai_api_key = st.text_input("OpenAI API Key", key="file_oa_api_key", type="password")
            if openai_api_key != '':
                try:
                    oa_client = OpenAI(api_key=openai_api_key)
                    response = oa_client.models.list()
                    update_session(llm_option_selected=True, llm_option=llm_option, llm_client=oa_client)
                except:
                    st.warning("Invalid API key. Please provide a valid API token.")
        else:
            oa_client = st.session_state['embedding_client']
            update_session(llm_option_selected=True, llm_option=llm_option, llm_client=oa_client)

    elif llm_option == "2. FLAN-5":
        model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
        tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
        update_session(llm_option_selected=True, llm_option=llm_option, llm_client=model, llm_tokenizer=tokenizer)