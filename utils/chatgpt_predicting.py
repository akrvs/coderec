import pickle
import re
import os
import torch
from neural.architectures.chatgpt_model import get_completion
from transformers import BartModel, BartTokenizer

def generate_candidates(initial_prompt):

    initial_prompt = initial_prompt.lower()

    # Check if candidates file exists
    candidates_filename = f"/Users/akrvs/PycharmProjects/Project/candidates.pkl"
    if os.path.exists(candidates_filename):
        with open(candidates_filename, "rb") as f:
            split_candidate_sentences, embeddings = pickle.load(f)
    else:
        response = get_completion(initial_prompt, num_candidates=1)
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        model = BartModel.from_pretrained('facebook/bart-base')
        candidate_responses = [choice['message']['content'] for choice in response['choices']]
        candidate_sentences = [re.split(r'#', response) for response in candidate_responses]
        split_candidate_sentences = [[sentence] for sublist in candidate_sentences for sentence in sublist]

        embeddings = []
        for sentence in split_candidate_sentences:
            tokens = tokenizer(sentence[0], return_tensors="pt",
                               max_length=1024, truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**tokens)
            sentence_embeddings = outputs.last_hidden_state.mean(dim=1)
            flattened_embedding = sentence_embeddings.flatten().tolist()
            embeddings.append(flattened_embedding)

        with open(candidates_filename, "wb") as f:
            pickle.dump((split_candidate_sentences, embeddings), f)

    return split_candidate_sentences, embeddings