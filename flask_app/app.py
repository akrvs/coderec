from flask import Flask, render_template, request, jsonify
from flasgger import Swagger
import openai
from flask_app.run_experiments import run_experiment
from utils.similarities import *
import json
import os

app = Flask(__name__)
Swagger(app)

openai.api_key = os.environ.get('OPENAI_API_KEY', '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')

with open('config.json', 'r') as config_file:
    config = json.load(config_file)

batch_size = config.get('batch_size', 128)
learning_rate = config.get('learning_rate', 0.01)
num_genes = config.get('num_genes', 100)
pretrained_model_path = config.get('pretrained_model_path', '/lstm_model.pth')
database_path = config.get('database_path', '/S08_question_answer_pairs_new.txt')

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Endpoint to run experiments based on user input.
    ---
    parameters:
      - name: prompt
        in: formData
        type: string
        description: The prompt for the experiment.
        required: true
      - name: model_choice
        in: formData
        type: string
        enum: ['lstm', 'chatgpt']
        description: Choose the model for the experiment.
        required: true
    responses:
      200:
        description: Result of the experiment.
    """
    result = None  # Initialize result to None

    if request.method == 'POST':
        prompt = request.form.get('prompt')
        model_choice = request.form.get('model_choice')

        if prompt:
            if model_choice == 'LSTM':
                result = run_lstm_experiment(prompt)
            elif model_choice == 'ChatGPT':
                result = run_chatgpt_experiment(prompt)
            else:
                result = "Invalid model choice"

    return render_template('index.html', result=result)

def run_lstm_experiment(prompt):
    return run_experiment(prompt, model_architecture='LSTM', num_epochs=100, batch_size=batch_size,
                          learning_rate=learning_rate, num_genes=num_genes, loss_function=None, window_length=2,
                          database_path=database_path, pretrained_model_path=pretrained_model_path)

def run_chatgpt_experiment(prompt):
    return run_experiment(prompt, model_architecture='ChatGPT', num_epochs=100, batch_size=batch_size,
                          learning_rate=learning_rate, num_genes=num_genes, loss_function=lukasiewicz_implication_2)

@app.route('/run_lstm', methods=['POST'])
def run_lstm():
    """
    Endpoint to run LSTM experiments.
    ---
    parameters:
      - name: prompt
        in: formData
        type: string
        description: The prompt for the LSTM experiment.
        required: true
    responses:
      200:
        description: Result of the LSTM experiment.
    """
    prompt = request.form.get('prompt')
    result = run_lstm_experiment(prompt)
    return jsonify({'result': result})

@app.route('/run_chatgpt', methods=['POST'])
def run_chatgpt():
    """
    Endpoint to run ChatGPT experiments.
    ---
    parameters:
      - name: prompt
        in: formData
        type: string
        description: The prompt for the ChatGPT experiment.
        required: true
    responses:
      200:
        description: Result of the ChatGPT experiment.
    """
    prompt = request.form.get('prompt')
    result = run_chatgpt_experiment(prompt)
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
