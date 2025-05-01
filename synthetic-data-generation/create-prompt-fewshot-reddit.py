import random
import pandas as pd
from datasets import load_dataset

# parameters
n_prompts = 1500 # Jumlah synthetic data

# Load dataset
dataset_name = "enoubi/Reddit-Indonesian-Sarcastic-Fix-Text"
dataset = load_dataset(dataset_name)
dataset = dataset.shuffle(seed=42) 

# Template untuk System Prompt dan User Prompt
system_template = """
Anda adalah model bahasa yang ahli dalam memahami dan menghasilkan kalimat sarkasme dalam bahasa Indonesia, terutama dalam konteks media sosial seperti Reddit. Sarkasme adalah bentuk ekspresi yang menyampaikan makna berlawanan dari apa yang dikatakan, biasanya dengan tujuan menyindir, mengkritik, atau mengekspresikan ketidakpuasan secara tidak langsung.
Anda akan diberikan tiga contoh kalimat sarkasme sebagai referensi. Tugas Anda adalah pelajari struktur, pola bahasa, dan gaya penulisan dari contoh-contoh tersebut, lalu buat satu kalimat sarkasme baru.
Kalimat tersebut harus:
- Terdengar alami seperti komentar yang ditulis oleh pengguna Reddit di Indonesia.
- Menghindari penggunaan kata seru pada awal kalimat seperti ‘Wah’, ‘Wow’, atau ungkapan serupa.
- Disajikan tanpa tambahan penjelasan atau teks lain
""".strip()

user_template = """
Contoh:
1. [example]
2. [example]
3. [example]
""".strip()

# Filter data sarcastic
filtered_data = [item['text'] for item in dataset['train'] if item['label'] == 1]

ind = 0 
total_data = len(filtered_data)

# Replace [example]
def fill_example(template, data):
    global ind
    num_examples = template.count("[example]")
    selected_samples = []
    
    for _ in range(num_examples):
        selected_samples.append(data[ind]) 

        # All used, reshuffle
        ind += 1
        if ind >= total_data: 
            ind = 0
            random.seed(42)
            random.shuffle(data) 

    # Ganti <example> dengan contoh yang telah dipilih
    for sample in selected_samples:
        template = template.replace("[example]", sample, 1)
    
    return template


# Generate N prompt
def generate_prompts(system_template, user_template, data, n_prompts=5):
    system_prompts = []
    user_prompts = []
    outputs = []

    for _ in range(n_prompts):
        user_prompt = fill_example(user_template, data)
        system_prompts.append(system_template)
        user_prompts.append(user_prompt)
        outputs.append("")
    return system_prompts, user_prompts, outputs

system_prompts, user_prompts, outputs = generate_prompts(
    system_template, user_template, filtered_data, n_prompts=n_prompts
)

# Simpan ke file
df = pd.DataFrame({
    "System Prompt": system_prompts,
    "User Prompt": user_prompts,
    "Output": outputs
})
output_csv_file = "prompt/prompt-fewshot-reddit.csv"
df.to_csv(output_csv_file, index=False, encoding="utf-8")

print(f"Hasil generate telah disimpan ke dalam file: {output_csv_file}")
