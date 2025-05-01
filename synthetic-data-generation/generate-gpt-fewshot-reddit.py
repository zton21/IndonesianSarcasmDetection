import openai
import pandas as pd
import time
import os

# Konfigurasi API Key OpenAI
api_key = open('input/apikey').read().strip()
client = openai.OpenAI(api_key=api_key)

# Load file CSV yang berisi System Prompt dan User Prompt
input_csv_file = "prompt/prompt-fewshot-reddit.csv"
output_csv_file = "output/output-fewshot-reddit.csv"

# Load prompts
df = pd.read_csv(input_csv_file)

# Fungsi untuk melakukan query ke GPT-4o Mini
def process(system_prompt, user_prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

# Run in batches
def process_batch(start, end):
    end = min(end, len(df))  
    print(f"Processing rows {start} to {end - 1}...")

    # Load hasil terakhir
    if os.path.exists(output_csv_file):
        df_existing = pd.read_csv(output_csv_file)
    else:
        df_existing = pd.DataFrame(columns=df.columns.tolist())

    new_rows = []

    for index in range(start, end):
        row = df.iloc[index]
        system_prompt = row["System Prompt"]
        user_prompt = row["User Prompt"]

        # Skip jika data sudah ada
        if "Output" in df_existing.columns and index < len(df_existing) and pd.notna(df_existing.loc[index, "Output"]):
            print(f"Skipping index {index}")
            continue 

        print(f"Processing index {index}...")
        output = process(system_prompt, user_prompt)
        new_rows.append([system_prompt, user_prompt, output])

        # Agar tidak terkena Rate limit OpenAI
        time.sleep(1)

    # Buat DataFrame untuk hasil baru
    df_new = pd.DataFrame(new_rows, columns=["System Prompt", "User Prompt", "Output"])

    # Gabungkan dengan hasil sebelumnya dan simpan
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.to_csv(output_csv_file, index=False, encoding="utf-8", lineterminator='\r\n')

# Range select
for i in range(500):
    start_index = i
    end_index = i + 1

    # Jalankan processing
    process_batch(start_index, end_index)