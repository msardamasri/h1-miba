import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

print("Downloading NLTK resources...")
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

def clean_text(text):
    """
    Text cleaning for Airbnb listings
    """
    if pd.isna(text):
        return "unknown"
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text if text else "unknown"


def advanced_preprocess_text(text, remove_stopwords=True, lemmatize=True):
    """
    Text preprocessing with lemmatization and stopword removal
    """
    if not text or text == "" or text == "unknown":
        return "unknown"
    
    tokens = word_tokenize(text)

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        keep_words = {'near', 'close', 'walk', 'distance', 'private', 'shared', 
                      'entire', 'room', 'apartment', 'house', 'studio', 'unknown'}
        stop_words = stop_words - keep_words
        tokens = [word for word in tokens if word not in stop_words]
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(word) for word in tokens]

    tokens = [word for word in tokens if len(word) > 2]

    return ' '.join(tokens) if tokens else "unknown"

def extract_amenities_text(amenities_str):
    """
    Extract and clean amenities from JSON-like string format
    """
    if pd.isna(amenities_str):
        return "unknown"
    text = str(amenities_str)
    text = re.sub(r'[\[\]"{}]', ' ', text)
    text = text.replace(',', ' ')
    cleaned = clean_text(text)
    return cleaned if cleaned != "unknown" else "unknown"


def generate_tfidf_features(df, text_column, n_features=50, 
                           ngram_range=(1, 2), max_df=0.8, min_df=2,
                           prefix='tfidf'):
    """
    Generate TF-IDF features from a text column
    """
    print(f"\n{'='*70}")
    print(f"Generating TF-IDF features for: {text_column}")
    print(f"Parameters: n_features={n_features}, ngram_range={ngram_range}")
    print(f"{'='*70}")
    
    tfidf = TfidfVectorizer(
        max_features=n_features,
        ngram_range=ngram_range,
        max_df=max_df,
        min_df=min_df,
        sublinear_tf=True,
        norm='l2'
    )
    
    tfidf_matrix = tfidf.fit_transform(df[text_column].fillna('unknown'))
    
    feature_names = [f"{prefix}_{name}" for name in tfidf.get_feature_names_out()]
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=feature_names,
        index=df.index
    )
    
    print(f"✓ Generated {tfidf_df.shape[1]} TF-IDF features")
    print(f"✓ Vocabulary size: {len(tfidf.vocabulary_)}")
    
    top_features = tfidf_df.mean().sort_values(ascending=False).head(10)
    print(f"\nTop 10 features by average TF-IDF score:")
    for feat, score in top_features.items():
        print(f"  {feat}: {score:.4f}")
    
    return tfidf_df, tfidf


def create_additional_text_features(df, text_columns):
    """
    Create additional statistical features from text columns
    """
    print(f"\n{'='*70}")
    print("CREATING ADDITIONAL TEXT FEATURES")
    print(f"{'='*70}")
    
    text_features = pd.DataFrame(index=df.index)
    for col in text_columns:
        if col not in df.columns:
            continue
        
        print(f"Processing: {col}")
        text_series = df[col].fillna('unknown')
        
        text_features[f'{col}_word_count'] = text_series.apply(
            lambda x: len(str(x).split()))
        text_features[f'{col}_char_count'] = text_series.apply(
            lambda x: len(str(x)))
        text_features[f'{col}_avg_word_length'] = text_series.apply(
            lambda x: np.mean([len(word) for word in str(x).split()]) if len(str(x).split()) > 0 else 0)
        text_features[f'{col}_uppercase_ratio'] = text_series.apply(
            lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0)
        text_features[f'{col}_exclamation_count'] = text_series.apply(
            lambda x: str(x).count('!'))
        text_features[f'{col}_is_missing'] = df[col].isna().astype(int)
    
    print(f"Created {text_features.shape[1]} additional text features")
    
    return text_features

def analyze_tfidf_importance(tfidf_df, tfidf_vectorizer, top_n=20):
    """
    Analyze and visualize TF-IDF feature importance
    """
    idf_scores = pd.DataFrame({
        'feature': tfidf_vectorizer.get_feature_names_out(),
        'idf_score': tfidf_vectorizer.idf_
    }).sort_values('idf_score', ascending=False)
    
    avg_tfidf = tfidf_df.mean().sort_values(ascending=False)
    doc_freq = (tfidf_df > 0).sum().sort_values(ascending=False)
    
    importance_df = pd.DataFrame({
        'feature': avg_tfidf.index,
        'avg_tfidf': avg_tfidf.values,
        'doc_frequency': [doc_freq.get(feat, 0) for feat in avg_tfidf.index]
    })
    
    print(f"\n{'='*70}")
    print(f"TF-IDF FEATURE IMPORTANCE ANALYSIS")
    print(f"{'='*70}")
    print(f"\nTop {top_n} features by average TF-IDF score:")
    print(importance_df.head(top_n).to_string(index=False))
    
    return importance_df

def process_airbnb_text_features(df, text_columns_config=None):
    """
    Complete pipeline to process text columns and generate TF-IDF features
    """
    print("\n" + "="*70)
    print("AIRBNB TEXT FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    if text_columns_config is None:
        text_columns_config = {
            'description': {
                'n_features': 50,
                'ngram_range': (1, 2),
                'max_df': 0.7,
                'min_df': 5,
                'prefix': 'desc'
            },
            'name': {
                'n_features': 30,
                'ngram_range': (1, 2),
                'max_df': 0.8,
                'min_df': 3,
                'prefix': 'name'
            },
            'neighborhood_overview': {
                'n_features': 30,
                'ngram_range': (1, 2),
                'max_df': 0.7,
                'min_df': 3,
                'prefix': 'neighborhood'
            },
            'amenities': {
                'n_features': 40,
                'ngram_range': (1, 1),
                'max_df': 0.8,
                'min_df': 5,
                'prefix': 'amenity'
            }
        }
    
    df_processed = df.copy()
    all_tfidf_features = []
    vectorizers = {}
    for col, config in text_columns_config.items():
        if col not in df.columns:
            print(f"\nColumn '{col}' not found in dataset, skipping...")
            continue
        
        print(f"\n{'─'*70}")
        print(f"Processing column: {col}")
        print(f"{'─'*70}")
        print("Step 1/3: Cleaning text...")
        if col == 'amenities':
            df_processed[f'{col}_clean'] = df[col].apply(extract_amenities_text)
        else:
            df_processed[f'{col}_clean'] = df[col].apply(clean_text)
        print("Step 2/3: Preprocessing (tokenization, lemmatization, stopword removal)...")
        df_processed[f'{col}_processed'] = df_processed[f'{col}_clean'].apply(
            lambda x: advanced_preprocess_text(x, remove_stopwords=True, lemmatize=True)
        )
        print("Step 3/3: Generating TF-IDF features...")
        tfidf_features, vectorizer = generate_tfidf_features(
            df_processed,
            f'{col}_processed',
            n_features=config['n_features'],
            ngram_range=config['ngram_range'],
            max_df=config['max_df'],
            min_df=config['min_df'],
            prefix=config['prefix']
        )
        
        all_tfidf_features.append(tfidf_features)
        vectorizers[col] = vectorizer
        analyze_tfidf_importance(tfidf_features, vectorizer, top_n=15)
    if all_tfidf_features:
        print(f"\n{'='*70}")
        print("COMBINING ALL TF-IDF FEATURES")
        print(f"{'='*70}")
        
        tfidf_combined = pd.concat(all_tfidf_features, axis=1)
        df_final = pd.concat([df_processed, tfidf_combined], axis=1)
        
        print(f"✓ Total TF-IDF features created: {tfidf_combined.shape[1]}")
        print(f"✓ Final dataset shape: {df_final.shape}")
        
        return df_final, vectorizers, tfidf_combined
    else:
        print("\n⚠ No text columns were processed")
        return df_processed, {}, pd.DataFrame()

def main_tfidf_pipeline(df):
    """
    Main TF-IDF pipeline for Airbnb hackathon
    """
    print("\n" + "="*70)
    print("STARTING COMPLETE TF-IDF PIPELINE FOR AIRBNB DATA")
    print("="*70)
    
    df_with_tfidf, vectorizers, tfidf_features = process_airbnb_text_features(df)
    
    text_columns = ['description', 'name', 'neighborhood_overview']
    additional_features = create_additional_text_features(
        df,
        [col for col in text_columns if col in df.columns]
    )

    df_final = pd.concat([df_with_tfidf, additional_features], axis=1)
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Final dataset shape: {df_final.shape}")
    print(f"✓ TF-IDF features: {tfidf_features.shape[1]}")
    print(f"✓ Additional text features: {additional_features.shape[1]}")
    print(f"✓ Total new features: {tfidf_features.shape[1] + additional_features.shape[1]}")
    
    return df_final, vectorizers