import openai

def get_completion(prompt, model="gpt-3.5-turbo", num_candidates=10):
    """
        Retrieves completions for a given prompt using the OpenAI Chat API.

        Args:
            prompt (str): The user prompt for which completions are requested.
            model (str): The OpenAI language model to use (default: "gpt-3.5-turbo").
            num_candidates (int): The number of completion candidates to generate (default: 10).

        Returns:
            dict: The response object containing completion information.
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,
        n=num_candidates
    )
    return response