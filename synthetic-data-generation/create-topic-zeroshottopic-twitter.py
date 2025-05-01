import openai
import pandas as pd
import time
import os

# Fill OpenAI API Key
api_key = open('input/apikey').read().strip()
client = openai.OpenAI(api_key=api_key)

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

result = process("Anda adalah model bahasa yang ahli dalam menganalisis dan menghasilkan daftar topik yang sedang ramai dibicarakan di media sosial.",
    "Berikan daftar 250 topik di Twitter")

print(result)

f = open("prompt/zeroshot-twitter-topics.txt", "w")
f.write(result)