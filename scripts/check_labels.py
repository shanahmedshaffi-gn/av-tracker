
import pickle

with open('data/all_embeddings.pkl', 'rb') as f:
    data = pickle.load(f)
    labels = data['labels']
    unique_labels = sorted(list(set(labels)))
    print("Selected actors:")
    for label in unique_labels:
        print(label)
