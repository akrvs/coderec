import openai

def get_completion(prompt, model="gpt-3.5-turbo", num_candidates=10):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0.7,
        n=num_candidates
    )
    return response