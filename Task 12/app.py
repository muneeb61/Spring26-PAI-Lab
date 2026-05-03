import glob
import os
import re
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

DATASET_PATH  = './data/**/*.csv'       
EMBEDDINGS_FILE = './hadith_embeddings.npy'
FAISS_INDEX_FILE = './faiss_index.index'
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# ── Column names from LK Hadith Corpus
COL_NAMES = [
    'Chapter_Number', 'Chapter_English', 'Chapter_Arabic',
    'Section_Number', 'Section_English', 'Section_Arabic',
    'Hadith_number', 'English_Hadith', 'English_Isnad', 'English_Matn',
    'Arabic_Hadith', 'Arabic_Isnad', 'Arabic_Matn', 'Arabic_Comment',
    'English_Grade', 'Arabic_Grade'
]

# ── Text Cleaning 
def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'[^A-Za-z\s]', '', text)
        text = text.lower().strip()
    else:
        text = ''
    return text

# ── Load Dataset 
def load_dataset():
    csv_files = glob.glob(DATASET_PATH, recursive=True)
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found at '{DATASET_PATH}'.\n"
            "Please clone the dataset: git clone https://github.com/ShathaTm/LK-Hadith-Corpus.git data"
        )

    print(f"📂 Found {len(csv_files)} CSV files...")
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, names=COL_NAMES, encoding='utf-8', skiprows=0)
        except Exception:
            df = pd.read_csv(f, names=COL_NAMES, encoding='latin-1', skiprows=0)
        dfs.append(df)

    hadith_df = pd.concat(dfs, ignore_index=True)

    # Clean
    hadith_df = hadith_df.dropna(subset=['English_Hadith'])
    hadith_df = hadith_df[hadith_df['English_Hadith'].str.strip() != '']
    hadith_df = hadith_df.reset_index(drop=True)
    hadith_df['Cleaned_Hadith'] = hadith_df['English_Hadith'].apply(clean_text)
    hadith_df = hadith_df[hadith_df['Cleaned_Hadith'] != ''].reset_index(drop=True)

    print(f"✅ Dataset loaded: {len(hadith_df)} Hadiths")
    return hadith_df

# ── Build or Load FAISS Index 
def build_or_load_index(hadith_df, model):
    if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(FAISS_INDEX_FILE):
        print("⚡ Loading saved embeddings & FAISS index...")
        embeddings = np.load(EMBEDDINGS_FILE)
        index = faiss.read_index(FAISS_INDEX_FILE)
        print(f"✅ FAISS index loaded — {index.ntotal} vectors")
    else:
        print("⏳ Generating embeddings (first run — may take a few minutes)...")
        embeddings = model.encode(
            hadith_df['Cleaned_Hadith'].tolist(),
            show_progress_bar=True,
            batch_size=64
        )
        embeddings = np.array(embeddings, dtype='float32')
        np.save(EMBEDDINGS_FILE, embeddings)

        dimensions = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimensions)
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_FILE)
        print(f"✅ FAISS index built & saved — {index.ntotal} vectors")

    return embeddings, index

# ── Initialise everything on startup
print("🚀 Starting Hadith Bot...")
print("⏳ Loading MiniLM model...")
model = SentenceTransformer(MODEL_NAME)

hadith_df = load_dataset()
embeddings, faiss_index = build_or_load_index(hadith_df, model)
print("✅ Ready! Open http://localhost:5000\n")

# ── Flask Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get('query', '').strip()
    count = int(data.get('count', 5))

    if not query:
        return jsonify({'results': []})

    cleaned_query = clean_text(query)
    query_embedding = model.encode([cleaned_query], convert_to_numpy=True).astype('float32')
    distances, indices = faiss_index.search(query_embedding, count)

    results = []
    for i in range(count):
        idx = indices[0][i]
        row = hadith_df.iloc[idx]
        results.append({
            'hadith':   row['English_Hadith'],
            'chapter':  str(row.get('Chapter_English', '')),
            'section':  str(row.get('Section_English', '')),
            'grade':    str(row.get('English_Grade', '')),
            'distance': float(distances[0][i])
        })

    return jsonify({'results': results})


if __name__ == '__main__':
    app.run(debug=False, port=5000)
