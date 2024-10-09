from gliner import GLiNER
from collections import defaultdict

# Function to rank and return top words based on average score and count
def get_top_words_per_topic(topics, top_n=15):
    grouped = defaultdict(lambda: defaultdict(list))

    for item in topics:
        grouped[item['label']][item['text']].append(item['score'])

    entities = [
        {label: [
            {
                'text': text,
                'total_score': sum(scores),
                'count': len(scores)
            } for text, scores in texts.items()
        ]} for label, texts in grouped.items()
    ]

    sorted_data = defaultdict(list)

    # Sort each category by count and average_score (both descending)
    for item in entities:
        for label, values in item.items():
            sorted_values = sorted(values, key=lambda x: (-x['count'], -x['total_score']))
            sorted_data[label].extend(sorted_values)

    # Prepare the final dictionary with the top 5 words based on sorted order
    top_words_per_topic = {}

    for label, items in sorted_data.items():
        top_words_per_topic[label] = [item['text'] for item in items[:top_n]]  # Get top 5 words

    return top_words_per_topic

def list_topics(documents, labels):
    # list of topics in bar 
    model = GLiNER.from_pretrained("urchade/gliner_base")
    
    chunks = documents['chunks']
    entities = []

    for chunk in chunks:
        entities += model.predict_entities(chunk['text'], labels)

    return get_top_words_per_topic(entities)
    # return entities
