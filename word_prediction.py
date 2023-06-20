import numpy as np
import torch
from architectures.lstm_model import LstmModel
from utils.preproccessing import preprocess_data
from data.database import read_database

# Global Variables
window_length = 1

# Read the database.db
word_list = read_database()

# Preprocess data
word_list, word_to_int, int_to_word, n_words = preprocess_data(word_list)

# Load the pre-trained LSTM model
best_lstm_model = LstmModel(n_words)
best_lstm_model.load_state_dict(torch.load("lstm_model.pth"))
best_lstm_model.eval()

while True:
    try:
        # Prompt the user for a word
        initial_prompt = input("Enter a word from the database: ").lower()

        # List to store predicted words
        initial_predictions = []

        # Iterate for predictions
        for _ in range(5):
            # Prepare the input pattern
            words = initial_prompt.split()
            if len(words) >= window_length:
                pattern = np.zeros((window_length, n_words))
                for i, w in enumerate(words[-window_length:]):
                    pattern[i, word_to_int[w]] = 1
                x = torch.tensor([pattern], dtype=torch.float32)

                # Perform prediction
                prediction = best_lstm_model(x)
                index = int(prediction.argmax())
                result = int_to_word[index]

                # Update the prompt with the predicted word
                initial_prompt += " " + result

                # Store the predicted word
                initial_predictions.append(result)

        print(initial_predictions)

        # Generate predictions for each word in initial_predictions
        for word in initial_predictions:
            # Set the current prompt to the word from initial_predictions
            prompt = word

            # List to store predictions for the current word
            word_predictions = []

            # Iterate for predictions
            for i in range(5):
                words = prompt.split()
                if len(words) >= window_length:
                    pattern = np.zeros((window_length, n_words))
                    for i, w in enumerate(words[-window_length:]):
                        pattern[i, word_to_int[w]] = 1
                    x = torch.tensor([pattern], dtype=torch.float32)

                    # Perform prediction
                    prediction = best_lstm_model(x)
                    index = int(prediction.argmax())
                    result = int_to_word[index]

                    # Update the prompt with the predicted word
                    prompt += " " + result

                    # Store the predicted word
                    word_predictions.append(result)

            # Print the predictions for the current word
            print("Predictions for", word, ":", word_predictions)

    except Exception as e:
        print("Error:", str(e))








