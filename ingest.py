from minsearch import Index as minsearch
from utils import get_episode_title, get_podcast_details, get_feed_details, search_for_episode, fetch_latest_episode, download_all
import json
from transcribe import transcribe_with_replicate, transcribe_with_whistler

def create_minsearch_index(index_name):
    return minsearch(
        index_name = index_name,
        text_fields = ['text'],
        keyword_fields = ['id']
    )

def create_es_index(client, index_name):
    # Create mapping
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "id": {"type": "keyword", "store": True},
                "text": {"type": "text"},
                "text_vector": {"type": "dense_vector", "dims": 768},
            }
        }
    }

    client.indices.delete(index=index_name, ignore_unavailable=True)
    client.indices.create(index=index_name, body=index_settings)
    
    return client.indices.get_alias(index=index_name)

def create_chroma_index(client, index_name):
    existing_collections = client.list_collections()

    # Check if the collection exists
    if index_name in [collection.name for collection in existing_collections]:
        # Delete the collection if it exists
        client.delete_collection(index_name)
    else:
        print(f"Index {index_name} does not exist in the current collection.")

    # Create or get a collection with cosine distance
    index = client.get_or_create_collection(
        name=index_name,
        metadata={"hnsw:space": "cosine"}
    )

    return index

def create_index(**kwargs):
    vector_db = kwargs['vector_db']

    if vector_db == "1. Minsearch":
        return create_minsearch_index(index_name=kwargs['index_name'])
    elif vector_db == "2. Elasticsearch":
        return create_es_index(client=kwargs['vector_db_client'], index_name=kwargs['index_name'])
    elif vector_db == "3. ChromaDB":
        return create_chroma_index(client=kwargs['vector_db_client'], index_name=kwargs['index_name'])

def download_episode_from_url(url, sentence_encoder, **kwargs):
    try:
        podcast_id, episode_title = get_episode_title(url)
        podcast_details = get_podcast_details(podcast_id)

        feed_details = get_feed_details(
            podcast_details['feedUrl'], 
            sentence_encoder=sentence_encoder, 
            encoder=kwargs['encoder'], 
            embedding_client=kwargs['embedding_client'], 
            embedding_model=kwargs['embedding_model']
            )    

        episode_details = search_for_episode(
            episode_title, 
            feed_details,
            sentence_encoder=sentence_encoder, 
            encoder=kwargs['encoder'], 
            embedding_client=kwargs['embedding_client'], 
            embedding_model=kwargs['embedding_model']        
            )

        if episode_details['cos_sim'] < 0.95:
            raise Exception
        episode_details['filenames'] = []
        episode_details['filenames'] += download_all(episode_details['audio_urls'], podcast_details['collectionName'])
        episode_details['status'] = 'Success'
        episode_details['status_message'] = f"Podcast {podcast_details['collectionName']} downloaded successfully."
    except Exception:
        episode_details = {}
        episode_details['status'] = 'Fail'
        episode_details['status_message'] = "Failed to download the podcast. Please try again."
        return episode_details

    return episode_details

def download_episode_from_name(id, name):
    try:
        podcast_details = get_podcast_details(id)
        episode_details = fetch_latest_episode(podcast_details['feedUrl'])
        episode_details['filenames'] = []
        episode_details['filenames'] += download_all(episode_details['audio_urls'], name)
        episode_details['status'] = 'Success'
        episode_details['status_message'] = f"Podcast {podcast_details['collectionName']} - {name} downloaded successfully."
        return episode_details
    except Exception:
        episode_details = {}
        episode_details['status'] = 'Fail'
        episode_details['status_message'] = "Failed to download the podcast. Please try again."
        return episode_details

def download_podcast(**kwargs):
    option = kwargs['episode_option']
    if option == "1. Try a sample":
        with open('sample/episode_details.json', 'r') as f:
            episode_details = json.load(f)
    elif option == "2. Provide the iTunes URL for a specific podcast episode":
        episode_url = kwargs['episode_url']
        episode_details = download_episode_from_url(
            episode_url, 
            kwargs['sentence_encoder'],
            encoder=kwargs['encoder'] if 'encoder' in kwargs.keys() else None,
            embedding_client=kwargs['embedding_client'] if 'embedding_client' in kwargs.keys() else None,
            embedding_model=kwargs['embedding_model'] if 'embedding_model' in kwargs.keys() else None
            )
    elif option == "3. Provide a name of a podcast to explore its most recent episode":
        found_podcasts = kwargs['found_podcasts']
        selected_index = kwargs['selected_index']
        episode_details = download_episode_from_name(found_podcasts[selected_index]['collectionId'], found_podcasts[selected_index]['collectionName'])
    return episode_details

def transcribe_podcast(**kwargs):
    podcast_option = kwargs['episode_option']
    episode_details = kwargs['episode_details']
    
    if podcast_option == "1. Try a sample":
        chunks, text = episode_details['chunks'], episode_details['text']
    else:
        transcription_method = kwargs['transcription_method']
        if transcription_method == "1. Replicate":
            chunks, text = transcribe_with_replicate(kwargs['transcription_client'], episode_details['filenames'], n_splits=1)
        elif transcription_method == "2. Local transcription":
            chunks, text = transcribe_with_whistler(episode_details['filenames'], n_splits=1) # 4 splits took 800 seconds / try it in streamlit
    
    return {'chunks': chunks, 'text': text}

def create_oa_embedding(client, model, chunks):
    documents = []

    for sentence in chunks:
        temp_dict = {
            'id': sentence['id'],
            'text': sentence['text'],
            'text_vector': client.embeddings.create(model=model, input=sentence['text']).data[0].embedding[:768] # openai embeddings 3 provides flexibility when cutting embedding size 
        }
        documents.append(temp_dict)

    return documents

def create_t5_embedding(encoder, chunks):
    documents = []

    for sentence in chunks:
        temp_dict = {
            'id': sentence['id'],
            'text': sentence['text'],
            'text_vector': encoder.encode(sentence["text"]).tolist()
        }
        documents.append(temp_dict)
    
    return documents

def encode_podcast(**kwargs):
    episode_details = kwargs['episode_details']
    sentence_encoder = kwargs['sentence_encoder']

    if sentence_encoder == "1. T5":
        documents = create_t5_embedding(kwargs['encoder'], episode_details['chunks'])
    elif sentence_encoder == "2. OpenAI":
        documents = create_oa_embedding(kwargs['embedding_client'], kwargs['embedding_model'], episode_details['chunks'])

    return {'documents': documents}

def populate_minsearch_index(docs, index):
    documents = [{'id': str(doc['id']), 'text': doc['text']} for doc in docs]
    index.fit(documents)

def populate_es_index(documents, index_name, client):
    # add documents 
    for doc in documents:
        try:
            client.index(index=index_name, body=doc)
        except Exception as e:
            print(e)

    return index_name

def populate_chroma_collection(documents, collection):
    ids = [str(i+1) for i in range(len(documents))]
    embeddings = [doc['text_vector'] for doc in documents]
    texts = [doc['text'] for doc in documents]

    # print(ids[:4])
    # print('string')
    
    collection.add(
        ids=ids,
        embeddings=embeddings,
        metadatas=[{"text": text} for text in texts]
    )

def index_podcast(**kwargs):
    episode_details = kwargs['episode_details']
    vector_db = kwargs['vector_db']

    if vector_db=="1. Minsearch":
        populate_minsearch_index(episode_details['chunks'], kwargs['index'])
    elif vector_db=="2. Elasticsearch":
        populate_es_index(episode_details['documents'], kwargs['index_name'], kwargs['vector_db_client'])
    elif vector_db=="3. ChromaDB":
        populate_chroma_collection(episode_details['documents'], kwargs['index'])        
