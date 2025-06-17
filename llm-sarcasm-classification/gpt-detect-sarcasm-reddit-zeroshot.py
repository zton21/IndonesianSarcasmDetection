import openai
import pandas as pd
import time
import os
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Konfigurasi API Key OpenAI
api_key = open('../apikey').read().strip()
client = openai.OpenAI(api_key=api_key)

dataset_name = "enoubi/Reddit-Indonesian-Sarcastic-Fix-Text"
dataset = load_dataset(dataset_name)
data_test = dataset['test']
df = pd.DataFrame(data_test)

output_csv_file = f"result/gpt4omini-predictions-zeroshot-reddit.csv"

# Fungsi untuk melakukan query ke GPT-4o Mini
def detect_sarcasm(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=[
                {
                    "role": "developer", 
                    "content": (
                        "Anda adalah model bahasa yang ahli dalam mendeteksi sarkasme dalam teks berbahasa Indonesia."
                        "Sarkasme didefinisikan sebagai penggunaan ironi atau pernyataan yang secara eksplisit menyampaikan sesuatu yang berlawanan dengan maksud sebenarnya, biasanya dengan tujuan menyindir atau mengejek."
                        "Tugas Anda adalah mengklasifikasikan teks yang diberikan ke dalam dua kategori: ‘Sarkasme’ dan ‘Bukan Sarkasme’."
                        "Gunakan konteks, pilihan kata, dan nada yang tersirat dalam teks untuk mengidentifikasi sarkasme secara akurat."
                        "Anda hanya perlu menjawab dengan satu kata, yaitu 'Sarkasme' atau 'Bukan Sarkasme', tanpa tambahan teks lain."
                    )
                },
                {
                    "role": "user", 
                    "content": (
                        "Klasifikasikan teks berikut:\n"
                        f"Teks: \"{text}\"\n"
                        "Label:"
                    )
                }
            ],
            temperature=0
        )
        result = response.choices[0].message.content.strip()

        if result == "Sarkasme":
            return 1
        elif result == "Bukan Sarkasme":
            return 0
        else:
            return -1
    except Exception as e:
        print(f"Error: {e}")
        return -1

def process_batch(start, end):
    end = min(end, len(df))
    print(f"Processing rows {start} to {end-1}...")

    if os.path.exists(output_csv_file):
        df_existing = pd.read_csv(output_csv_file)
    else:
        df_existing = pd.DataFrame(columns=["Text", "True Label", "Prediction"])

    new_rows = []

    for index in range(start, end):
        row = df.iloc[index]
        text = row['text']
        true_label = row['label']

        # Cek apakah text ini sudah diproses sebelumnya
        if not df_existing.empty and (df_existing['Text'] == text).any():
            print(f"Skipping index {index}, already processed.")
            continue

        print(f"Processing index {index}...")

        # Query GPT
        prediction = detect_sarcasm(text)

        # Validasi hasil
        if prediction not in [0, 1]:
            print(f"Invalid result at index {index}: {prediction}")

        new_rows.append([text, true_label, prediction])

        time.sleep(0.5)

    df_new = pd.DataFrame(new_rows, columns=["Text", "True Label", "Prediction"])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)

    # Simpan ke file CSV
    df_combined.to_csv(output_csv_file, index=False, encoding="utf-8", line_terminator='\r\n')
    print(f"Batch processing selesai. Hasil disimpan di {output_csv_file}")

# Tentukan range data yang ingin diproses
start_index = 0
end_index = len(df)

# Jalankan batch processing
process_batch(start_index, end_index)

df = pd.read_csv(output_csv_file)

y_true = df["True Label"]
y_pred = df["Prediction"]

acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)

print("-" * 40)
print(f"  Accuracy : {acc:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall   : {recall:.4f}")
print(f"  F1 Score : {f1:.4f}")
print("-" * 40)