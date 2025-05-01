import streamlit as st
import pandas as pd
from PIL import Image

st.markdown(
    """

<style>
.element-container:nth-of-type(9) td:nth-of-type(4) * {
    text-decoration: none !important;
}
</style>
## PENGARUH GENERASI DATA SINTETIS MENGGUNAKAN GPT-4O MINI TERHADAP PERFORMA DETEKSI SARKASME BERBAHASA INDONESIA
""", unsafe_allow_html=True)

st.markdown(
"""
<h5 style="display:flex;justify-content:space-between;padding-bottom:1.5em;">
    <span>Jonathan Lee</span>
    <span>Kartono Apriliyandi</span>
    <span>Rachel Fanggian</span>
    <span>Derwin Suhartono</span>
</h5>
""", unsafe_allow_html=True
)

st.markdown("""
            
*Studi ini menganalisis pengaruh data sintetis berbasis prompting yang dihasilkan oleh model bahasa GPT, khususnya GPT-4o mini terhadap performa model deteksi sarkasme dalam bahasa Indonesia.*
#### Latar Belakang
Sarkasme merupakan bentuk komunikasi yang sering ditemukan di media sosial dan memiliki makna yang sering kali bertolak belakang dengan kata-kata yang digunakan. Kehadirannya menyulitkan teknologi NLP dalam memahami maksud sebenarnya, yang dapat menyebabkan kesalahan dalam analisis sentimen dan menurunkan akurasi dalam berbagai aplikasi seperti riset pasar dan evaluasi layanan. Tantangan ini semakin besar karena keterbatasan data sarkasme berbahasa Indonesia. Oleh karena itu, pendekatan baru seperti generasi data sintetis menggunakan Large Language Model (LLM), khususnya GPT-4o mini, mulai dimanfaatkan dengan menerapkan teknik prompting seperti zero-shot topic, one-shot, dan few-shot untuk memperkaya variasi data dan meningkatkan performa deteksi sarkasme.

#### Metodologi Penelitian
"""
, unsafe_allow_html=True)

st.image(Image.open("website/assets/Kerangka Berpikir.png"))

st.markdown(
    """
- **Data Preprocessing**: Memastikan bahwa setiap teks dapat dibaca, diproses, dan dipahami secara akurat oleh model bahasa. Melakukan perbaikan encoding html, emoji, dan unicode pada dataset asli yang rusak.
- **Generasi Data Sintetis**: Memperkaya dataset asli dengan data tambahan berlabel sarkasme.
Proses ini menggunakan teknik zero-shot topic, one-shot, dan few-shot
prompting dengan bantuan GPT-4o mini sebagai Large Language Model (LLM)
guna menghasilkan data sintetis yang relevan dalam konteks sarkasme
berbahasa Indonesia. 
- **Augmentasi Data**: Menggabungkan dataset sintetis yang dihasilkan dengan data
asli melalui proses augmentasi data, dengan proporsi bertingkat sebesar 10%,
20%, 30%, 40%, dan 50% dari jumlah data latih asli berlabel sarkasme.
- **Latih Model Klasifikasi**: Melatih beberapa model klasifikasi, yaitu dengan fine-tuning model IndoNLU
IndoBERT dan XLM-RoBERTa dengan dataset hasil augmentasi.
- **Evaluasi Model**: Mengevaluasi dengan mengukur kinerja model dalam mendeteksi sarkasme menggunakan metrik seperti akurasi, precision, recall, dan skor F1.
- **Analisis Hasil**: Analisis hasil yang tidak hanya membahas performa model, tetapi juga
menyertakan visualisasi data sintetis guna memahami karakteristik distribusi
dan variasi teks hasil generasi.
- **Implementasi Model:** Implementasi model deteksi sarkasme terbaik dari platform Twitter maupun Reddit dalam website ini, yang
memungkinkan pengguna untuk secara langsung menguji dan menganalisis
teks yang mengandung potensi sarkasme, baik dari Twitter maupun Reddit. 
#### Hasil Penelitian
Berdasarkan hasil penelitian yang telah dilakukan, berikut merupakan
kesimpulan yang diperoleh:

1. **Dampak Positif Data Sintetis**: Augmentasi data asli dengan data sintetis berbasis prompting berdampak positif dan signifikan terhadap performa model deteksi sarkasme. Peningkatan F1-score terlihat konsisten pada model IndoNLU IndoBERT dan XLM-R di dataset Twitter dan Reddit, dengan dampak paling signifikan pada dataset kecil seperti Twitter, sehingga manfaat data sintetis lebih terasa dalam kondisi low-resource, di mana jumlah data asli terbatas.

2. **Efektivitas Teknik Prompting**: Teknik few-shot prompting memberikan hasil terbaik dalam meningkatkan performa model deteksi sarkasme, diikuti oleh one-shot dan zero-shot. Efektivitas teknik sangat dipengaruhi oleh kualitas semantik data asli, dengan peningkatan kecil pada dataset Reddit yang memiliki konteks sarkasme lemah.

""")
st.image(Image.open("website/assets/result.png"), width=900)

st.markdown("""
3. **Proporsi Optimal Data Sintetis**: Penambahan data sintetis yang optimal berbeda di tiap kasus: 30% (IndoNLU IndoBERT) dan 10% (XLM-R) pada Twitter; 50% (IndoNLU IndoBERT) dan 40–50% (XLM-R) pada Reddit. Perbedaan ini mencerminkan kebutuhan data sintetis bergantung pada arsitektur model dan kualitas data asli.

#### Model 
Berikut adalah model dengan performa terbaik untuk mendeteksi sarkasme pada platform Twitter dan Reddit. Anda dapat mengakses model lengkap melalui link HuggingFace atau mencoba langsung kemampuan deteksi model melalui website.
""")

data = [
    {"Dataset": "Twitter", "Model": "XLM-R", "Teknik Prompting": "Few-Shot", "Link HuggingFace": "[XLM-RoBERTa-Twitter-Indonesian-Sarcastic-Few-Shot](https://huggingface.co/enoubi/XLM-RoBERTa-Twitter-Indonesian-Sarcastic-Few-Shot)", "": "[Coba Sekarang! 🚀](/twitter)"},
    {"Dataset": "Reddit", "Model": "XLM-R", "Teknik Prompting": "Few-Shot", "Link HuggingFace": "[XLM-RoBERTa-Reddit-Indonesian-Sarcastic-Few-Shot](https://huggingface.co/enoubi/XLM-RoBERTa-Reddit-Indonesian-Sarcastic-Few-Shot)", "": "[Coba Sekarang! 🚀](/reddit)"},
]

df = pd.DataFrame(data).set_index("Dataset")

st.table(df)


data = [
   {"Dataset":"Original (Preprocessed)", "Twitter": "[Twitter-Indonesian-Sarcastic-Fix-Text](https://huggingface.co/datasets/enoubi/Twitter-Indonesian-Sarcastic-Fix-Text)", "Reddit": "[Reddit-Indonesian-Sarcastic-Fix-Text](https://huggingface.co/datasets/enoubi/Reddit-Indonesian-Sarcastic-Fix-Text)"},
   {"Dataset":"Zero-Shot Topic", "Twitter":"[Twitter-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic](https://huggingface.co/datasets/enoubi/Twitter-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic)", "Reddit":"[Reddit-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic](https://huggingface.co/datasets/enoubi/Reddit-Indonesian-Sarcastic-Synthetic-Zero-Shot-Topic)"},
   {"Dataset":"One-Shot", "Twitter":"[Twitter-Indonesian-Sarcastic-Synthetic-One-Shot](https://huggingface.co/datasets/enoubi/Twitter-Indonesian-Sarcastic-Synthetic-One-Shot)", "Reddit":"[Reddit-Indonesian-Sarcastic-Synthetic-One-Shot](https://huggingface.co/datasets/enoubi/Reddit-Indonesian-Sarcastic-Synthetic-One-Shot)"},
   {"Dataset":"Few-Shot", "Twitter":"[Twitter-Indonesian-Sarcastic-Synthetic-Few-Shot](https://huggingface.co/datasets/enoubi/Twitter-Indonesian-Sarcastic-Synthetic-Few-Shot)", "Reddit":"[Reddit-Indonesian-Sarcastic-Synthetic-Few-Shot](https://huggingface.co/datasets/enoubi/Reddit-Indonesian-Sarcastic-Synthetic-Few-Shot)"}
]
df2 = pd.DataFrame(data).set_index("Dataset")

st.markdown(
    """
#### Dataset
Dataset tersedia untuk deteksi sarkasme di Twitter dan Reddit, mencakup data asli yang telah diproses dan data sintetis yang dihasilkan dengan teknik Zero-Shot Topic, One-Shot, serta Few-Shot. Semua link dapat diakses melalui Hugging Face.
""")

st.table(df2)

st.markdown("")
st.markdown("")