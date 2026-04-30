# Hadith QnA Bot — Lab 12
### MiniLM + FAISS + Flask Pipeline

---

## Setup & Run

### 1. Clone the Hadith Dataset
```bash
git clone https://github.com/ShathaTm/LK-Hadith-Corpus.git data
```
This creates a `data/` folder with all CSV files inside subfolders (Bukhari, AbuDaud, etc.)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
python app.py
```

### 4. Open in Browser
```
http://localhost:5000
```

---

## What happens on first run?
- Loads all 39,038 Hadiths from the CSV files
- Generates MiniLM embeddings (takes ~5–10 minutes)
- Builds FAISS index and saves it

## What happens on next runs?
- Loads saved embeddings & index instantly (seconds)

---

## Project Structure
```
hadith_flask/
├── app.py               # Flask backend (load, embed, FAISS, search)
├── requirements.txt     # Dependencies
├── templates/
│   └── index.html       # Frontend UI
├── data/                # ← Clone the dataset here
│   ├── Bukhari/
│   ├── AbuDaud/
│   └── ...
├── hadith_embeddings.npy   # Auto-generated on first run
└── faiss_index.index        # Auto-generated on first run
```
