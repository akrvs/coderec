import torch

import neural
import data
from utils.candidates_generation import *
from cluster_analysis import genetic_algorithm
from neural.mlp_training_2 import train_mlp_model
from utils.load_or_train import load_or_train_lstm_model
from neural.mlp_training import train_mlp_model as train_lstm_mlp_model

def run_experiment(prompt, model_architecture, num_epochs, batch_size, learning_rate, num_genes, loss_function=None,
                   window_length=None, database_path=None, pretrained_model_path=None):

    if model_architecture == 'LSTM':

        word_list = data.read_database(database_path)
        word_list, word_to_int, int_to_word, n_words = data.preprocess_data(word_list)
        best_lstm_model, word_to_int, int_to_word = load_or_train_lstm_model(neural.LstmModel, database_path, window_length, batch_size, num_epochs,
                                                   pretrained_model_path, n_words)

        candidates = generate_candidates_lstm(initial_prompt=prompt, window_length=window_length,
                                                  n_words=n_words, word_to_int=word_to_int,
                                               int_to_word=int_to_word, best_lstm_model=best_lstm_model)
        print()
        print(candidates)
        '''candidate_embeddings = train_lstm_mlp_model(neural.MlpModel, database_path, best_lstm_model,
                                              candidates, window_length, batch_size, num_epochs)'''


        candidate_embeddings = neural.get_candidate_embeddings(embedding_model=best_lstm_model.lstm,
                                                               candidates=candidates,
                                                               window_length=window_length, n_words=n_words,
                                                               word_to_int=word_to_int)
        mlp_model = neural.MlpModel(candidate_embeddings.shape[1], 512, n_words)
        train_mlp_model(mlp_model, candidate_embeddings, candidates, learning_rate,
                        num_epochs=num_epochs,
                        loss_function=loss_function)
        MLP_output = mlp_model(candidate_embeddings)
        MLP_output_detach = MLP_output.detach().numpy()
        selected_indices, selected_embeddings = genetic_algorithm(MLP_output_detach, num_genes, num_epochs)
        result_info = []
        for i, indices in enumerate(selected_indices):
            sentences = [candidates[idx] for idx in indices]
            cleaned_sentences = [sentence.strip().replace('\\n', '') for sublist in sentences for sentence in sublist]
            formatted_sentences = [f"[{sentence}]" for sentence in cleaned_sentences]
            result_info.extend(formatted_sentences)

        return '\n'.join(result_info), selected_indices, selected_embeddings, MLP_output, candidates, candidate_embeddings

    elif model_architecture == 'ChatGPT':
        initial_prompt = prompt
        split_candidate_sentences, embeddings = generate_candidates_chatgpt(initial_prompt)

        mlp_model = neural.MlpModel(len(embeddings[0]), 512, len(embeddings[0]))
        train_mlp_model(mlp_model, embeddings, split_candidate_sentences, learning_rate,
                                                 num_epochs=num_epochs,
                                                 loss_function=loss_function)
        embeddings_tensor = torch.tensor(embeddings)
        MLP_output = mlp_model(embeddings_tensor)
        MLP_output_detach = MLP_output.detach().numpy()
        selected_indices, selected_embeddings = genetic_algorithm(MLP_output_detach, num_genes, num_epochs)
        result_info = []
        for i, indices in enumerate(selected_indices):
            sentences = [split_candidate_sentences[idx] for idx in indices]
            cleaned_sentences = [sentence.strip().replace('\\n', '') for sublist in sentences for sentence in sublist]
            formatted_sentences = [f"[{sentence}]" for sentence in cleaned_sentences]
            result_info.extend(formatted_sentences)

        return '\n'.join(result_info)

    return 'No results'