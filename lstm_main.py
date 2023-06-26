from utils.preproccessing import preprocess_data
from data.database import read_database
from neural.lstm_predicting import generate_candidates
from utils.load_or_train import load_or_train_lstm_model

window_length = 1
n_epochs = 100
batch_size = 128

database_path = '/Users/akrvs/PycharmProjects/database.db'
word_list = read_database(database_path)

word_list, word_to_int, int_to_word, n_words = preprocess_data(word_list)

pretrained_model_path = "/Users/akrvs/PycharmProjects/Project/lstm_model.pth"
best_lstm_model = load_or_train_lstm_model(database_path, window_length, batch_size, n_epochs, pretrained_model_path,
                                           n_words)

print()
initial_prompt = input("Enter a word from the database: ")
candidates = generate_candidates(initial_prompt, window_length, n_words, word_to_int, int_to_word, best_lstm_model)
print(candidates)









