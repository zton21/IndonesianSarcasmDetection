import pandas as pd

# n : data per topic
n = 6
output_csv_file = "prompt/prompt-zeroshottopic-reddit.csv"

data = open("prompt/zeroshot-reddit-topics.txt").read().strip().splitlines()
x = len(data)

system_template = system_template = """
Anda adalah model bahasa yang ahli dalam memahami dan menghasilkan kalimat sarkasme dalam bahasa Indonesia, terutama dalam konteks media sosial seperti Reddit. Sarkasme adalah bentuk ekspresi yang menyampaikan makna berlawanan dari apa yang dikatakan, biasanya dengan tujuan menyindir, mengkritik, atau mengekspresikan ketidakpuasan secara tidak langsung.
Tugas Anda adalah menghasilkan satu kalimat sarkasme yang sesuai dengan topik yang diberikan.
Kalimat tersebut harus:
- Terdengar alami seperti komentar yang ditulis oleh pengguna Reddit di Indonesia.
- Menghindari penggunaan kata seru pada awal kalimat seperti ‘Wah’, ‘Wow’, atau ungkapan serupa.
- Disajikan tanpa tambahan penjelasan atau teks lain
""".strip()

df = pd.DataFrame({
    "System Prompt": [system_template] * x * n,
    "User Prompt": ["Topik: " + x for x in data] * n,
    "Generated Output": [""] * x * n
})

df.to_csv(output_csv_file, index=False, encoding="utf-8")