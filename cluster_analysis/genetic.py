import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def initialize_population(num_genes, num_candidates):
    return np.random.choice(num_candidates, size=(num_genes, 5), replace=True)

def calculate_fitness(embeddings, selected_indices):
    selected_embeddings = [embeddings[i] for i in selected_indices]
    similarity_matrix = cosine_similarity(selected_embeddings, selected_embeddings)
    average_similarity = np.mean(similarity_matrix)
    return -average_similarity

def selection(population, fitness_scores):
    sorted_indices = np.argsort(fitness_scores)
    selected_indices = population[sorted_indices[-1:]]
    return selected_indices

def crossover(parents):
    return np.mean(parents, axis=0).astype(int)

def mutation(child, num_candidates):
    mutation_point = np.random.choice(1)
    child[mutation_point] = np.random.choice(num_candidates)
    return child

def genetic_algorithm(embeddings, num_genes, num_epochs):
    num_candidates = len(embeddings)
    population = initialize_population(num_genes, num_candidates)

    best_fitness = float('-inf')
    best_selected_indices = None
    best_selected_embeddings = None

    for epoch in range(num_epochs):
        fitness_scores = [calculate_fitness(embeddings, genes) for genes in population]
        selected_indices = selection(population, fitness_scores)

        current_fitness = max(fitness_scores)
        if current_fitness > best_fitness:
            best_fitness = current_fitness
            best_selected_indices = selected_indices
            best_selected_embeddings = [embeddings[i] for i in best_selected_indices[0]]

        new_population = []

        for i in range(num_genes):
            parents = population[np.random.choice(len(population), size=2, replace=False)]
            child = crossover(parents)
            child = mutation(child, num_candidates)
            new_population.append(child)

        population = np.array(new_population)

    return best_selected_indices, best_selected_embeddings

