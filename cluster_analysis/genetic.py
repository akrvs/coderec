import random

def genetic_algorithm(candidates, similarity_results, k, num_genes, num_permutations, num_epochs):
    n = len(candidates)

    def create_gene(k, n):
        gene = [0 for _ in range(n)]
        selected_positions = random.sample(range(n), k=k)
        for pos in selected_positions:
            gene[pos] = 1
        return gene

    def fitness(gene):
        positions = [i for i, exists in enumerate(gene) if exists == 1]
        ret = 0
        for i in positions:
            for j in positions:
                ret += similarity_results[i][j]
        return ret

    def permute(gene):
        while True:
            positions = [i for i, exists in enumerate(gene) if exists == 1]
            pos = positions[int(random.random() * len(positions))]
            new_gene = [value for value in gene]
            new_gene[pos] = 0
            new_gene[int(random.random() * len(new_gene))] = 1
            if sum(new_gene) >= 5:
                return new_gene

    pool = [create_gene(k, n) for _ in range(num_genes)]

    for epoch in range(num_epochs):
        new_pool = []
        for gene in pool:
            for _ in range(num_permutations):
                new_pool.append(permute(gene))
            new_pool.append(gene)
        evals = {i: -fitness(gene) for i, gene in enumerate(pool)}
        pool_ids = sorted(list(evals.keys()), key=lambda i: evals[i])[:len(pool)]
        pool = [new_pool[i] for i in pool_ids]
        '''print("Best Fitness: ", -evals[pool_ids[0]])
        print(evals)
        print(pool_ids)'''

    results = [candidates[i] for i, exists in enumerate(pool[0]) if exists == 1]
    return results
