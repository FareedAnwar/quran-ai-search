import numpy as np
import pandas as pd
from flask import Flask, app, request, jsonify
from fuzzywuzzy import process, fuzz
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import torch
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel


# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Flask app
app = Flask(__name__)

# Load dataset
df = pd.read_csv("quran2.csv")[['ayah_text', 'surrahname', 'ayah', 'RevelationPlace']]

def preprocessing(df):
    def preprocess_arabic_text(text):
        if isinstance(text, str):
            text = re.sub(r'[\u064B-\u065F]', '', text)  # Remove diacritics
            text = re.sub(r'[آٱأإ]', 'ا', text)  # Normalize alif
            text = re.sub(r'ة', 'ه', text)  # Normalize ta marbuta
            text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('arabic'))
            tokens = [word for word in tokens if word not in stop_words]
            return ' '.join(tokens)
        return ""
    
    df['ayah_text_processed'] = df['ayah_text'].apply(preprocess_arabic_text)
    df['surrahname_processed'] = df['surrahname'].apply(preprocess_arabic_text)
    return df

# Apply preprocessing
df = preprocessing(df)

# Load BERT model and tokenizer
model_name = "aubmindlab/bert-base-arabertv02"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

def get_embeddings(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: val.to(model.device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings

# Compute embeddings for ayah and surah
df['embeddings'] = df['ayah_text_processed'].apply(lambda x: get_embeddings([x])[0] if x else np.zeros(768))
df['embeddings_surrahname_processed'] = df['surrahname_processed'].apply(lambda x: get_embeddings([x])[0] if x else np.zeros(768))

# Save embeddings (optional)
np.save("ayah_embeddings.npy", np.vstack(df['embeddings'].values))
np.save("surah_embeddings.npy", np.vstack(df['embeddings_surrahname_processed'].values))

class SemanticSearch:
    def __init__(self, df, embeddings_column='embeddings', text_column='ayah_text_processed', 
                 original_text_column='ayah_text', surrah_name_column='surrahname', 
                 ayah_column='ayah', revelation_place_column='RevelationPlace', search_type='ayah'):
        self.df = df
        self.text_column = text_column
        self.original_text_column = original_text_column
        self.surrah_name_column = surrah_name_column
        self.ayah_column = ayah_column
        self.revelation_place_column = revelation_place_column
        self.embeddings_column = embeddings_column
        self.search_type = search_type
        self.embeddings = np.vstack(df[embeddings_column].dropna().values)
        self.embeddings = normalize(self.embeddings, axis=1)
        self.tokenizer = tokenizer
        self.model = model

    def preprocess_text(self, text):
        text = text.strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def get_embedding(self, text):
        text = self.preprocess_text(text)
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True, padding=True, max_length=512)
        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        return normalize(embedding, axis=1)

    def find_similar_conversations(self, user_input, threshold=0.6, top_n=7):
        if not isinstance(user_input, str) or not user_input.strip():
            return ["Cannot search with an empty input."]

        user_input = self.preprocess_text(user_input)
        results_with_info = []

        if self.search_type == 'ayah':
            def format_result(row):
                return f"({row[self.ayah_column]}) {row[self.surrah_name_column]}\n{row[self.original_text_column]}"
        else:  # surah
            def format_result(row):
                return f"Surah: {row[self.original_text_column]} (Revelation: {row[self.revelation_place_column]})"

        # Fuzzy matching
        all_texts = self.df[self.text_column].dropna().tolist()
        fuzzy_matches = process.extract(user_input, all_texts, limit=top_n, scorer=fuzz.partial_ratio)
        fuzzy_texts = [match[0] for match in fuzzy_matches if match[1] > 70]
        for text in fuzzy_texts:
            row = self.df[self.df[self.text_column] == text].iloc[0]
            results_with_info.append(format_result(row))

        # Keyword matching
        keyword_matches_df = self.df[self.df[self.text_column].str.contains(user_input, na=False, regex=False)]
        for _, row in keyword_matches_df.iterrows():
            results_with_info.append(format_result(row))

        # Semantic matching
        user_embedding = self.get_embedding(user_input)
        similarities = cosine_similarity(user_embedding, self.embeddings)[0]
        for idx in np.argsort(similarities)[::-1]:
            if similarities[idx] >= threshold:
                row = self.df.iloc[idx]
                results_with_info.append(format_result(row))

        # Deduplicate and limit results
        all_results = list(dict.fromkeys(results_with_info))[:top_n]
        return all_results if all_results else [f"No relevant {'ayahs' if self.search_type == 'ayah' else 'Surahs'} found."]

# Initialize SemanticSearch for ayah and surah
ayah_search = SemanticSearch(
    df,
    embeddings_column='embeddings',
    text_column='ayah_text_processed',
    original_text_column='ayah_text',
    surrah_name_column='surrahname',
    ayah_column='ayah',
    search_type='ayah'
)

surah_search = SemanticSearch(
    df,
    embeddings_column='embeddings_surrahname_processed',
    text_column='surrahname_processed',
    original_text_column='surrahname',
    revelation_place_column='RevelationPlace',
    search_type='surah'
)

def predict(user_query, search_type='ayah'):
    if search_type == 'ayah':
        results = ayah_search.find_similar_conversations(user_query, threshold=0.6, top_n=10)
    else:
        results = surah_search.find_similar_conversations(user_query, threshold=0.8, top_n=5)
    return results

# Flask API endpoint
@app.route('/search', methods=['GET'])
def search():
    user_query = request.args.get('query')
    search_type = request.args.get('type', 'ayah')  # Default to 'ayah'

    if not user_query:
        return jsonify({'error': 'Missing query parameter'}), 400

    if search_type not in ['ayah', 'surah']:
        return jsonify({'error': 'Invalid search type. Use "ayah" or "surah".'}), 400

    results = predict(user_query, search_type)

    return jsonify({
        'original_query': user_query,
        'search_type': search_type,
        'results': results
    })

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)