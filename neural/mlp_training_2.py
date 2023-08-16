import torch
import torch.optim as optim
import utils
from utils.similarities import lukasiewicz_implication_2

def train_mlp_model(model, embeddings, split_candidate_sentences, learning_rate, num_epochs, loss_function):

    divided_embeddings_lists = [embeddings[i:i + 3] for i in range(0, len(embeddings), 3)]
    divided_sentences_lists = [split_candidate_sentences[i:i + 3] for i in
                               range(0, len(split_candidate_sentences), 3)]

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    all_similarity_results = []

    for epoch in range(num_epochs):
        model.train()
        loss = 0
        optimizer.zero_grad()
        n = 0
        impl = 0

        sentence_similarity_results = []

        for triplet_index, (triplet_embeddings, _) in enumerate(zip(divided_embeddings_lists, divided_sentences_lists),
                                                                start=1):
            num_sentences = len(triplet_embeddings)

            for i in range(num_sentences):
                for j in range(num_sentences):
                    for k in range(num_sentences):
                        embedding_i = triplet_embeddings[i]
                        embedding_j = triplet_embeddings[j]
                        embedding_k = triplet_embeddings[k]

                        tensor_i = torch.tensor(embedding_i).unsqueeze(0)
                        tensor_j = torch.tensor(embedding_j).unsqueeze(0)
                        tensor_k = torch.tensor(embedding_k).unsqueeze(0)

                        output_i = model(tensor_i)
                        output_j = model(tensor_j)
                        output_k = model(tensor_k)

                        x = utils.cosine_similarity(output_i, output_j)
                        y = utils.cosine_similarity(output_j, output_k)
                        z = utils.cosine_similarity(output_k, output_i)

                        if x > 0.9999 or y > 0.9999 or z > 0.9999:
                            continue

                        loss += loss_function(x, y, z) + x + y + z
                        n += 1
                        impl += lukasiewicz_implication_2(x, y, z)
                        sentence_similarity_results.append([x, y, z])

        all_similarity_results.append(sentence_similarity_results)

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {(loss.item()) / n:.4f} \t {(impl.item()) / n:.4f}")

    return all_similarity_results