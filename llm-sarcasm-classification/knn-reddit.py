from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import pandas as pd

# Load dataset dari Hugging Face
dataset = load_dataset("enoubi/Reddit-Indonesian-Sarcastic-Fix-Text")

# Ambil data train & test dengan field "text" dan "label"
train_texts = dataset["train"]["text"]
train_labels = dataset["train"]["label"]

test_texts = dataset["test"]["text"]
test_labels = dataset["test"]["label"]

# Fungsi untuk mencari kNN
def knn_sampling(train_texts, train_labels, test_texts, test_labels, k=5, model_name="firqaaa/indo-sentence-bert-base", output_file="knn_results.csv"):
    model = SentenceTransformer(model_name)

    # Pisahkan data train berdasarkan label
    train_0 = [text for text, label in zip(train_texts, train_labels) if label == 0]
    train_1 = [text for text, label in zip(train_texts, train_labels) if label == 1]

    # Buat vector embedding
    train_0_embeddings = model.encode(train_0, convert_to_numpy=True)
    train_1_embeddings = model.encode(train_1, convert_to_numpy=True)
    test_embeddings = model.encode(test_texts, convert_to_numpy=True)

    # KNN untuk label 0
    knn_0 = NearestNeighbors(n_neighbors=min(k, len(train_0)), metric="cosine").fit(train_0_embeddings)
    
    # KNN untuk label 1
    knn_1 = NearestNeighbors(n_neighbors=min(k, len(train_1)), metric="cosine").fit(train_1_embeddings)

    data = []    
    for i, test_text in enumerate(test_texts):
        # Cari top-k sample dari train label 0
        distances_0, indices_0 = knn_0.kneighbors([test_embeddings[i]])
        top_0 = [train_0[idx] for idx in indices_0[0]]

        # Cari top-k sample dari train label 1
        distances_1, indices_1 = knn_1.kneighbors([test_embeddings[i]])
        top_1 = [train_1[idx] for idx in indices_1[0]]

        data.append({
            "Test Text": test_text,
            "True Label": test_labels[i],  # Langsung ambil label dari dataset test
            "Top 5 Similar Label 0": "\n".join(top_0),
            "Top 5 Similar Label 1": "\n".join(top_1)
        })
    
    # Simpan ke CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Hasil disimpan dalam file: {output_file}")

knn_sampling(train_texts, train_labels, test_texts, test_labels, k=5, output_file="result/knn-reddit.csv")
