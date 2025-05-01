import openai
import pandas as pd
import time
import os

# Konfigurasi API Key OpenAI
api_key = open('input/apikey').read().strip()
client = openai.OpenAI(api_key=api_key)

# Load dataset kNN
knn_results_file = "result/knn-twitter.csv"
df_knn = pd.read_csv(knn_results_file)

# Fungsi untuk membuat prompt dengan jumlah shot yang dinamis
def generate_few_shot_user_prompt(test_text, similar_sarcasm, similar_non_sarcasm, num_shots):
    prompt_examples = ""
    for i in range(num_shots // 2):
        prompt_examples += (
            f"Contoh {2*i+1}:\n"
            f"Teks: \"{similar_sarcasm[i]}\"\n"
            f"Label: Sarkasme\n\n"
            f"Contoh {2*i+2}:\n"
            f"Teks: \"{similar_non_sarcasm[i]}\"\n"
            f"Label: Bukan Sarkasme\n\n"
        )
    
    return (
        f"{prompt_examples}"
        f"Klasifikasikan teks berikut:\n"
        f"Teks: \"{test_text}\"\n"
        "Label:"
    )

# Fungsi untuk melakukan query ke GPT-4o Mini
def detect_sarcasm_few_shot(test_text, similar_sarcasm, similar_non_sarcasm, num_shots):
    try:
        prompt = generate_few_shot_user_prompt(test_text, similar_sarcasm, similar_non_sarcasm, num_shots)
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
                        "Berikut adalah beberapa contoh untuk membantu Anda memahami pola deteksi:"
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )
        
        result = response.choices[0].message.content.strip()
        return 1 if result == "Sarkasme" else 0 if result == "Bukan Sarkasme" else -1
    except Exception as e:
        print(f"Error: {e}")
        return -1

def process_batch(start, end, num_shots, output_csv_file):
    print(num_shots)
    end = min(end, len(df_knn))
    print(f"Processing rows {start} to {end-1}...")

    if os.path.exists(output_csv_file):
        df_existing = pd.read_csv(output_csv_file)
    else:
        df_existing = pd.DataFrame(columns=["Test Text", "Prompt", "True Label", "Prediction"])

    new_rows = []

    for index in range(start, end):
        row = df_knn.iloc[index]
        test_text = row['Test Text']
        true_label = row.get('True Label', -1)
        
        similar_sarcasm = row['Top 5 Similar Label 1'].split('\n')[:num_shots//2]
        similar_non_sarcasm = row['Top 5 Similar Label 0'].split('\n')[:num_shots//2]
        
        if not df_existing.empty and (df_existing['Test Text'] == test_text).any():
            print(f"Skipping index {index}, already processed.")
            continue

        print(f"Processing index {index}...")
        prompt_text = generate_few_shot_user_prompt(test_text, similar_sarcasm, similar_non_sarcasm, num_shots)
        prediction = detect_sarcasm_few_shot(test_text, similar_sarcasm, similar_non_sarcasm, num_shots)
        new_rows.append([test_text, prompt_text, true_label, prediction])
        time.sleep(0.5)

    df_new = pd.DataFrame(new_rows, columns=["Test Text", "Prompt", "True Label", "Prediction"])
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(output_csv_file, index=False, encoding="utf-8", line_terminator='\r\n')
    print(f"Batch processing selesai. Hasil disimpan di {output_csv_file}")

shot_list = [4, 6, 8, 10]

for shots in shot_list:
    output_csv_file = f"result/gpt4omini-predictions-{shots}shot-twitter.csv"
    process_batch(0, len(df_knn), shots, output_csv_file)