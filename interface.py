import pysqlite3
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from streamlit_functions import *
from ingest import download_podcast, transcribe_podcast, encode_podcast, index_podcast
from rag import rag
from bs4 import BeautifulSoup
import pandas as pd
from helpers import list_topics


if __name__ == "__main__":
    # ask user to provide the necessary api keys
    with st.sidebar:

        # podcast option
        if not st.session_state.get('interaction_started', False):
            choose_podcast_option()

        # sentence encoder
        if st.session_state.get('episode_option_selected', False) and not st.session_state.get('interaction_started', False):
            choose_encoder()

        # transcription method
        if st.session_state.get('sentence_encoder_selected', False) and not st.session_state.get('interaction_started', False):
            choose_transcription_method()

        # vector database
        if st.session_state.get('transcription_method_selected', False) and not st.session_state.get('interaction_started', False):
            choose_vector_db()

        # llm
        if st.session_state.get('vector_db_selected', False) and not st.session_state.get('interaction_started', False):
            choose_llm()

        # Check if all settings are selected to enable/disable the button
        if st.session_state['episode_option_selected'] and st.session_state['sentence_encoder_selected'] and st.session_state['transcription_method_selected'] and st.session_state['vector_db_selected'] and st.session_state['llm_option_selected']:
            update_session(settings_ready=True)
        else:
            update_session(settings_ready=False)
        
        # Create two columns
        col1, col2 = st.columns(2)

        # Add a button to each column
        col1.button("Apply Settings", on_click=apply_settings, disabled=not st.session_state.get('settings_ready', False))

        col2.button("Reset Settings", on_click=reset_settings, disabled=not st.session_state.get('settings_applied', False))

    st.title("Podcast Search and Download Tool")

    if st.session_state.get('settings_applied', False):
        # download
        if not st.session_state.get('interaction_started', False):
            with st.spinner('Downloading...'):
                episode_details = download_podcast(**st.session_state)
                if episode_details['status'] == 'Success':
                    st.success(episode_details['status_message'])
                    update_session(episode_details=episode_details, podcast_downloaded=True)
                else:
                    st.warning(episode_details['status_message'])
                    update_session(podcast_downloaded=False)

        # transcribe
        if st.session_state['podcast_downloaded'] and not st.session_state.get('interaction_started', False):
            with st.spinner('Transcribing...'):
                st.session_state['episode_details'].update(transcribe_podcast(**st.session_state))
                update_session(podcast_transcribed=True)

        # encode
        if st.session_state['podcast_transcribed'] and not st.session_state.get('interaction_started', False):
            with st.spinner('Encoding...'):
                if st.session_state['vector_db'] != "1. Minsearch":
                    try:
                        st.session_state['episode_details'].update(encode_podcast(**st.session_state))
                        update_session(podcast_embedded=True)
                    except:
                        st.warning("Encoding failed.")          
                        update_session(podcast_embedded=False)
                else:
                    update_session(podcast_embedded=True)

        # populate index
        if st.session_state['podcast_embedded'] and not st.session_state.get('interaction_started', False):
            with st.spinner('Indexing...'):
                index_podcast(**st.session_state)
                update_session(podcast_indexed=True)

        # interact
        if st.session_state['podcast_indexed']:
            update_session(interaction_started=True)
            with st.expander("Need some ideas?"):
                fig_col1, fig_col2 = st.columns(2)

                with fig_col1:
                    st.subheader("Podcast summary")
                    if st.button("Show summary"):
                        episode_details = st.session_state['episode_details']
                        soup = BeautifulSoup(episode_details['summary'], "html.parser")
                        for br in soup.find_all("br"):
                            br.replace_with("\n")
                        parsed_text = soup.get_text()
                        st.markdown(parsed_text)

                with fig_col2:
                    st.subheader("Show topics discussed in the episode")
                    episode_details = st.session_state['episode_details']
                    labels = ["Person", "Organization", "Location", "Date", "Money", "Percent", "Event", "Product", "Work of Art", "Book", "Song", "Movie", "Language", "Law", "Facility", "Country", "City", "State", "Nationality", "Religion", "Political group"]
                    selected_labels = st.multiselect("Select or add labels:", labels)
                    new_label = st.text_input("Add a custom label:")
                    if new_label and new_label not in selected_labels:
                        selected_labels.append(new_label)
                    st.write("Selected Labels:", selected_labels)

                    if st.button("Show topics"):
                        with st.spinner('Extracting topics...'):
                            entities = list_topics(episode_details, selected_labels)
                            max_length = max(len(words) for words in entities.values())
                            data = {key: words + [''] * (max_length - len(words)) for key, words in entities.items()}
                            df = pd.DataFrame(data)
                            st.dataframe(df, hide_index=True, height=400)

            chatbox_container = st.container()
            with chatbox_container:
                st.subheader("Chat with your podcast")
                episode_details = st.session_state['episode_details']
                index_name = st.session_state['index_name']

                if "messages" not in st.session_state:
                    st.session_state.messages = []

                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                if query := st.chat_input(""):
                    st.session_state.messages.append({"role": "user", "content": query})
                    with st.chat_message("user"):
                        st.markdown(query)

                    with st.chat_message("assistant"):
                        response = st.write_stream(rag(query, **st.session_state))

                    st.session_state.messages.append({"role": "assistant", "content": response})
