import neural
from data.preproccessing import *
from utils.candidates_generation import *
from cluster_analysis import genetic_algorithm
from neural.mlp_training import train_mlp_model
from utils.load_or_train import load_or_train_lstm_model

def run_experiment(prompt, model_architecture, num_epochs, batch_size, learning_rate, num_genes, loss_function=None,
                   window_length=None, database=None, pretrained_model_path=None):
    """
        Runs an experiment with the specified parameters using either LSTM or ChatGPT model.

        Args:
            prompt (str): The initial prompt for the experiment.
            model_architecture (str): The model architecture to use ('LSTM' or 'ChatGPT').
            num_epochs (int): The number of epochs for training.
            batch_size (int): The batch size for training.
            learning_rate (float): The learning rate for training.
            num_genes (int): The number of genes in each candidate solution for genetic algorithm.
            loss_function (torch.nn.Module, optional): The loss function for training.
            window_length (int, optional): The window length for LSTM model.
            database (str, optional): The path to the database file for the LSTM model.
            pretrained_model_path (str, optional): The path to a pretrained model for the LSTM model.

        Returns:
            tuple: A tuple containing experiment results and information:
                - result_info (str): Formatted information about selected candidates.
                - selected_indices (numpy.ndarray): Indices of the selected candidates.
                - selected_embeddings: Embeddings of the selected candidates.
                - MLP_output: Output from the MLP model.
                - candidates: Generated candidates from the model.
                - candidate_embeddings: Embeddings of the generated candidates.
    """
    if model_architecture == 'LSTM':

        word_list, word_to_int, int_to_word, n_words = tokenizer(database)
        best_lstm_model, word_to_int, int_to_word = load_or_train_lstm_model(neural.LstmModel, database, word_list=word_list,
                                                                             word_to_int=word_to_int, int_to_word=int_to_word,
                                                                             window_length=window_length, batch_size=batch_size,
                                                                             n_epochs=num_epochs,
                                                                             pretrained_model_path=pretrained_model_path,
                                                                             n_words=n_words)
        candidates = generate_candidates_lstm(initial_prompt=prompt, window_length=window_length,
                                                  n_words=n_words, word_to_int=word_to_int,
                                               int_to_word=int_to_word, best_lstm_model=best_lstm_model)
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

        return '\n'.join(result_info), selected_indices, selected_embeddings, MLP_output

    return 'No results'