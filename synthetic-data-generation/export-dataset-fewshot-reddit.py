import pandas as pd
import re

def clean_text(text):
    text = re.sub(r"#\S+", "<hashtag>", text)  # Replace hashtags with <hashtag>
    return text

# Load the generated synthetic data
input_csv = "output/output-fewshot-reddit.csv"
output_csv = "output/dataset-fewshot-reddit.csv"
df = pd.read_csv(input_csv)

texts = []
if "Output" in df.columns:
    generated = df["Output"].dropna().tolist() 
    for x in generated:
        z = 0
        for row in x.replace('\n\n', '\n').split('\n'):
            y = row[row.find('.')+1:].strip()
            if y[0] == '"' and y[-1] == '"':
                y = y[1:-1].strip()
            if len(y) > 0:
                texts.append(y)
                z += 1
        print(z)
else:
    raise ValueError("Column 'Output' not found in CSV!")

texts = [clean_text(text) for text in texts]

# Create DataFrame with label = 1
df = pd.DataFrame({"text": texts, "label": 1})

# Save to CSV
df.to_csv(output_csv, index=False)

# Print sample
print(df.head())
print(f"Dataset saved to: {output_csv}")
