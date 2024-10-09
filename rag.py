import time

def search(query, **kwargs):
    vector_db = kwargs['vector_db']

    if vector_db == "1. Minsearch":
        boost = {'text':3.0}
        results = kwargs['index'].search(
            query=query,
            boost_dict=boost, 
            num_results=kwargs['num_results']
        )
    elif vector_db=="2. Elasticsearch":
        # Encode the query
        if kwargs['sentence_encoder'] == "1. T5":
            query_vector = kwargs['encoder'].encode(query).tolist()
        elif kwargs['sentence_encoder'] == "2. OpenAI":
            query_vector = kwargs['embedding_client'].embeddings.create(model=kwargs['embedding_model'], input=query).data[0].embedding[:768]

        # Construct the search query
        search_query = {
            "size": kwargs['num_results'],  # Limit the number of results
            "query": {
                "script_score": {
                    "query": {
                        "match_all": {}
                    },
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            }
        }
        # Execute the search query
        results = kwargs['vector_db_client'].search(index=kwargs['index_name'], body=search_query)
        results = results['hits']['hits']
        
    elif vector_db=="3. ChromaDB":
        # Encode the query
        if kwargs['sentence_encoder'] == "1. T5":
            query_vector = kwargs['encoder'].encode(query).tolist()
        elif kwargs['sentence_encoder'] == "2. OpenAI":
            query_vector = kwargs['embedding_client'].embeddings.create(model=kwargs['embedding_model'], input=query).data[0].embedding[:768]
        
        # Perform cosine similarity search in ChromaDB
        results = kwargs['vector_db_client'].get_or_create_collection(kwargs['index_name']).query(
            query_embeddings=[query_vector],
            n_results=kwargs['num_results'],
            include=["metadatas", "documents", "distances"]
        )
        
        # return results["metadatas"]
        results = results["metadatas"][0]


    return results

# prompt
def build_prompt(query, search_results):
    prompt_template = """
    You're a podcast chat bot. Answer the QUESTION based on the CONTEXT from the RESULTS database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}
    
    CONTEXT: 
    {context}
    """.strip()
    
    context = ""
    
    for search_result in search_results:
        doc = search_result['_source']['text'] if '_source' in search_result.keys() else search_result['text']
        context = context + f"{doc}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()

    return prompt

# generate
def llm(prompt, **kwargs):
    if kwargs['llm_option'] == "1. GPT-4o":
        outputs = kwargs['llm_client'].chat.completions.create(
            model='gpt-4o',
            messages=[{'role': 'user', 'content': prompt}]
        )
        response = outputs.choices[0].message.content
    elif kwargs['llm_option'] == "2. FLAN-5":
        inputs = kwargs['llm_tokenizer'](prompt, return_tensors="pt")
        outputs = kwargs['llm_client'].generate(
            inputs["input_ids"], 
            max_length=100,
            num_beams=5,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            top_p=0.95,    
            )
        response = kwargs['llm_tokenizer'].decode(outputs[0], skip_special_tokens=True)
        
    return response

# rag 
def rag(query, **kwargs):

    search_results = search(
        query, 
        vector_db=kwargs['vector_db'], 
        sentence_encoder=kwargs['sentence_encoder'], 
        encoder=kwargs['encoder'] if 'encoder' in kwargs.keys() else None, 
        index_name=kwargs['index_name'], 
        index=kwargs['index'], 
        vector_db_client=kwargs['vector_db_client'] if 'vector_db_client' in kwargs.keys() else None, 
        embedding_model=kwargs['embedding_model'] if 'embedding_model' in kwargs.keys() else None, 
        embedding_client=kwargs['embedding_client'] if 'embedding_client' in kwargs.keys() else None,
        num_results=5
        )

    prompt = build_prompt(query, search_results)

    answer = llm(
        prompt, 
        llm_option=kwargs['llm_option'], 
        llm_client=kwargs['llm_client'], 
        llm_tokenizer=kwargs['llm_tokenizer'] if 'llm_tokenizer' in kwargs.keys() else None
        )
    
    for word in answer.split():
        yield word + " "
        time.sleep(0.05)
