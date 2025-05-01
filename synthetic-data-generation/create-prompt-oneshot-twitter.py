import random
import pandas as pd
from datasets import load_dataset

# parameters
n_prompts = 500 # Jumlah synthetic data

# Load dataset
dataset_name = "enoubi/Twitter-Indonesian-Sarcastic-Fix-Text"
dataset = load_dataset(dataset_name)
dataset = dataset.shuffle(seed=42) 

# Template untuk System Prompt dan User Prompt
system_template = """
Anda adalah model bahasa yang ahli dalam memahami dan menghasilkan kalimat sarkasme dalam bahasa Indonesia, terutama dalam konteks media sosial seperti Twitter. Sarkasme adalah bentuk ekspresi yang menyampaikan makna berlawanan dari apa yang dikatakan, biasanya dengan tujuan menyindir, mengkritik, atau mengekspresikan ketidakpuasan secara tidak langsung.
Anda akan diberikan satu contoh kalimat sarkasme sebagai referensi. Tugas Anda adalah pelajari struktur, pola bahasa, dan gaya penulisan dari contoh-contoh tersebut, lalu buat satu kalimat sarkasme baru.
Kalimat tersebut harus:
- Terdengar alami seperti tweet yang ditulis oleh pengguna Twitter di Indonesia.
- Menghindari penggunaan kata seru pada awal kalimat seperti ‘Wah’, ‘Wow’, atau ungkapan serupa.
- Disajikan tanpa tambahan penjelasan atau teks lain
""".strip()

user_template = """
Contoh:
[example]
""".strip()

# Filter data sarcastic
filtered_data = [item['tweet'] for item in dataset['train'] if item['label'] == 1]

ind = 0 
# Replace [example]
def fill_example(template, data):
    global ind
    num_examples = template.count("[example]")
    for i in range(num_examples):
        example = data[ind]

        # All used, restart
        ind = (ind + 1) % len(data)

        # Ganti hanya satu <example> setiap iterasi
        template = template.replace("[example]", example, 1)

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
output_csv_file = "prompt/prompt-oneshot-twitter.csv"
df.to_csv(output_csv_file, index=False, encoding="utf-8")

print(f"Hasil generate telah disimpan ke dalam file: {output_csv_file}")
