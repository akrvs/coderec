import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def initialize_population(num_genes, num_candidates):
    """
        Initializes a population matrix for genetic algorithms.

        Args:
            num_genes (int): The number of genes in each candidate solution.
            num_candidates (int): The number of candidates in the population.

        Returns:
            numpy.ndarray: A 2D array representing the initial population.
    """
    return np.random.choice(num_candidates, size=(num_genes, 5), replace=True)

def calculate_fitness(embeddings, selected_indices):
    """
        Calculates the fitness of a set of selected embeddings based on cosine similarity.

        Args:
            embeddings (numpy.ndarray): An array containing the embeddings of all candidates.
            selected_indices (list): A list of indices corresponding to selected embeddings.

        Returns:
            float: The negative average cosine similarity of the selected embeddings.
    """
    selected_embeddings = [embeddings[i] for i in selected_indices]
    similarity_matrix = cosine_similarity(selected_embeddings, selected_embeddings)
    average_similarity = np.mean(similarity_matrix)
    return -average_similarity

def selection(population, fitness_scores):
    """
        Selects candidates from the population based on their fitness scores.

        Args:
            population (numpy.ndarray): The current population matrix.
            fitness_scores (numpy.ndarray): An array containing the fitness scores of each candidate.

        Returns:
            numpy.ndarray: The selected candidates based on their fitness.
    """
    sorted_indices = np.argsort(fitness_scores)
    selected_indices = population[sorted_indices[-1:]]
    return selected_indices

def crossover(parents):
    """
        Performs crossover by taking the element-wise mean of parent candidates.

        Args:
            parents (numpy.ndarray): A matrix containing parent candidates.

        Returns:
            numpy.ndarray: The offspring candidate resulting from crossover.
    """
    return np.mean(parents, axis=0).astype(int)

def mutation(child, num_candidates):
    """
        Applies mutation to a child candidate by randomly changing one gene.

        Args:
            child (numpy.ndarray): The candidate to undergo mutation.
            num_candidates (int): The total number of candidate solutions.

        Returns:
            numpy.ndarray: The mutated child candidate.
    """
    mutation_point = np.random.choice(1)
    child[mutation_point] = np.random.choice(num_candidates)
    return child

def genetic_algorithm(embeddings, num_genes, num_epochs):
    """
        Applies a genetic algorithm to select candidates with high fitness.

        Args:
            embeddings (numpy.ndarray): An array containing the embeddings of all candidates.
            num_genes (int): The number of genes in each candidate solution.
            num_epochs (int): The number of generations (epochs) for the genetic algorithm.

        Returns:
            tuple: A tuple containing:
                - best_selected_indices (numpy.ndarray): Indices of the best selected candidates.
                - best_selected_embeddings (list): Embeddings of the best selected candidates.
    """
    num_candidates = len(embeddings)
    population = initialize_population(num_genes, num_candidates)

    best_fitness = float('-inf')
    best_selected_indices = None
    best_selected_embeddings = None

    for epoch in range(num_epochs):
        fitness_scores = [calculate_fitness(embeddings, genes) for genes in population]
        fitness_scores = np.array(fitness_scores)
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