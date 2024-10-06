import torch

from sklearn.metrics.pairwise import cosine_similarity

def preprocess_query(text, nlp):
    doc = nlp(text)
    # Remove stop words and lemmatize the remaining words
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]

    # Remove duplicate words while preserving order
    filtered_tokens = list(dict.fromkeys(filtered_tokens))
    
    return ' '.join(filtered_tokens)

def get_word_embeddings(text, tokenizer, model):
    # Tokenize the input text and get the input IDs and attention mask
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    
    # Get the embeddings from the BERT model
    with torch.no_grad():
        outputs = model(**inputs)
    
    # outputs[0] contains the hidden states of all tokens in the input
    # Shape of outputs[0]: [batch_size, sequence_length, hidden_size]
    token_embeddings = outputs.last_hidden_state.squeeze(0)
    
    # Get the embeddings for each token (excluding special tokens like [CLS], [SEP])
    token_embeddings = token_embeddings[1:-1]
    
    # Get the corresponding tokens
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))[1:-1]
    
    return tokens, token_embeddings


def process_and_sort_results(results, top_n=5):
    # Initialize a dictionary to hold cumulative similarities and count of query tokens for each document
    doc_similarity_aggregate = {}

    # Iterate over each result to calculate cumulative similarity for each document
    for result in results:
        doc_idx = result['document_index']
        similarity = result['similarity']

        # If the document index is not already in the dictionary, initialize it
        if doc_idx not in doc_similarity_aggregate:
            doc_similarity_aggregate[doc_idx] = {"total_similarity": 0, "count": 0}

        # Add the similarity score to the total similarity for the document and increment the token count
        doc_similarity_aggregate[doc_idx]["total_similarity"] += similarity
        doc_similarity_aggregate[doc_idx]["count"] += 1

    # Calculate average similarity for each document
    avg_similarities = []
    for doc_idx, data in doc_similarity_aggregate.items():
        avg_similarity = data["total_similarity"] / data["count"]  # Average similarity
        avg_similarities.append({"document_index": doc_idx, "avg_similarity": avg_similarity})

    # Sort documents by their average similarity in descending order and get the top N
    sorted_docs = sorted(avg_similarities, key=lambda x: x['avg_similarity'], reverse=True)[:top_n]

    return sorted_docs


def word_level_search(query, document_tokens, document_embeddings, tokenizer, model):
    # Get word embeddings for the query
    query_tokens, query_embeddings = get_word_embeddings(query, tokenizer, model)
    
    # Store results (document index, query token, document token, similarity score)
    results = []

    # For each document, compute the cosine similarity between query words and document words
    for doc_idx, doc_embedding in enumerate(document_embeddings):
        if len(doc_embedding) == 0:  # Prevent errors if document has no embeddings
            continue
        
        # Compute similarity scores for all query tokens against the document embedding
        similarity_scores = cosine_similarity(query_embeddings, doc_embedding)
        
        # For each query token, find the most similar token in the document
        for i, query_token in enumerate(query_tokens):
            max_sim_idx = similarity_scores[i].argmax()
            max_sim_score = similarity_scores[i][max_sim_idx]
            
            # Append results with the corresponding query token and the document index
            results.append({
                "document_index": doc_idx,  # Index of the document in the DataFrame
                "document_token": document_tokens[doc_idx][max_sim_idx],
                "query_token": query_token,
                "similarity": max_sim_score
            })
    
    return process_and_sort_results(results)

