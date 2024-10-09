from utils import get_episode_title, get_podcast_details, get_feed_details, search_for_episode, fetch_latest_episode, download_all

def download_episode_from_url(sentence_encoder, url):
    try:
        podcast_id, episode_title = get_episode_title(url)
        podcast_details = get_podcast_details(podcast_id)
        feed_details = get_feed_details(sentence_encoder, podcast_details['feedUrl'])
        episode_details = search_for_episode(sentence_encoder, episode_title, feed_details)
        if episode_details['cos_sim'] < 0.95:
            raise Exception
        episode_details['filenames'] = []
        episode_details['filenames'] += download_all(episode_details['audio_urls'], podcast_details['collectionName'])
        episode_details['status'] = 'Success'
        episode_details['status_message'] = f"Podcast {podcast_details['collectionName']} downloaded successfully."
        return episode_details
    except Exception:
        episode_details = {}
        episode_details['status'] = 'Fail'
        episode_details['status_message'] = "Failed to download the podcast. Please try again."
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

# encode
def encode_documents(sentence_encoder, chunks):
    documents = []

    for sentence in chunks:
        temp_dict = {
            'text': sentence['text'],
            'text_vector': sentence_encoder.encode(sentence["text"]).tolist()
        }
        documents.append(temp_dict)
    
    return documents

# create index 
def create_index(encoder, es_client, docs):
    chunks = docs['chunks']
    documents = encode_documents(encoder, chunks)
    # Setup elasticsearch
    index_name = "podcast-transcriber"

    # Create mapping
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "text_vector": {"type": "dense_vector", "dims": 768},
            }
        }
    }

    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    es_client.indices.create(index=index_name, body=index_settings)

    # add documents 
    for doc in documents:
        try:
            # es_client.index(index=index_name, document=doc)
            es_client.index(index=index_name, body=doc)
        except Exception as e:
            print(e)

    return index_name
