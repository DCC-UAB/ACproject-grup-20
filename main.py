#ELIMINAR LLETRES SUELTES RANDOM??
import re
import pandas as pd
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


import LR
import SVM
import RF
import KNN
import NB

# Selecció model: KNN, LR, NB, RF, SVM 
MODEL_CHOICE = 'LR'

# path carpeta
DATA_PATH = 'C:/Users/marti/OneDrive/Escriptori/datasets_AC/'  
#DATA_PATH = 
#DATA_PATH = 

# Normalització del text
def normalize_text(text):
    """
    Funció per normalitzar el text: convertir a minúscules, eliminar mencions,
    URLs, caràcters especials i eliminar paraules d'aturada.
    """
    text = text.lower()
    text = re.sub(r'@\w+', '', text)  # Eliminar mencions (@usuari)
    text = re.sub(r'http\S+', '', text)  # Eliminar URLs
    text = re.sub(r'[^\w\sáéíóúÁÉÍÓÚ]', '', text)  # Eliminar caràcters especials
    words = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    return " ".join(filtered_words)

# Inicialitzar el lematitzador i el stemmer
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def lemmatize_and_stem(text):
    """
    Funció per lematitzar i aplicar stemming.
    """
    tokens = word_tokenize(text)
    # Lematitzar
    lemmatized = [lemmatizer.lemmatize(word) for word in tokens]
    # Stemming
    lemmatized = [stemmer.stem(word) for word in lemmatized]
    return " ".join(lemmatized)

# Pipeline de preprocessament
def preprocess_pipeline(data, column_name):
    """
    Funció que aplica la normalització i la lematització/stemming al dataset.
    """
    data['cleaned_text'] = data[column_name].apply(normalize_text)
    data['processed_text'] = data['cleaned_text'].apply(lemmatize_and_stem)
    return data

def load_and_preprocess_data(data_path):
    """
    Càrrega i preprocessament de les dades.
    """
    # Carregar el dataset Train, Valid, Test
    X_train = pd.read_csv(f'{data_path}Train.csv')
    X_valid = pd.read_csv(f'{data_path}Valid.csv')
    X_test = pd.read_csv(f'{data_path}Test.csv')

    X_train = X_train.head(5000)
    X_valid = X_valid.head(1000)
    X_test = X_test.head(1000)

    # Preprocessament
    X_train = preprocess_pipeline(X_train, 'text')
    X_valid = preprocess_pipeline(X_valid, 'text')
    X_test = preprocess_pipeline(X_test, 'text')

    y_train = X_train['label']
    y_valid = X_valid['label']
    y_test = X_test['label']

    X_train = X_train[['processed_text']]
    X_valid = X_valid[['processed_text']]
    X_test = X_test[['processed_text']]

    return X_train, y_train, X_valid, y_valid, X_test, y_test

def convert_to_numeric_matrices(X_train, X_valid, X_test):
    """
    Converteix els textos preprocessats en matrius numèriques utilitzant TF-IDF.
    """
    vectorizer = TfidfVectorizer(max_features=5000)  # ajustar el nombre de features
    X_train_matrix = vectorizer.fit_transform(X_train['processed_text'])
    X_valid_matrix = vectorizer.transform(X_valid['processed_text'])
    X_test_matrix = vectorizer.transform(X_test['processed_text'])
    return X_train_matrix, X_valid_matrix, X_test_matrix, vectorizer

def evaluar_resultats(y_true, y_pred):
    """
    Calcula i mostra la matriu de confusió i l'accuracy a partir de les etiquetes reals i les prediccions.
    """
    # Calcula la matriu de confusió
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriu de confusió:")
    print(cm)
    
    # Calcula l'accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"\nAccuracy: {accuracy:.4f}")
    
    return cm, accuracy

def main():
    # Carregar i processar les dades
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_and_preprocess_data(DATA_PATH)
    
    # Convertir a matrius numèriques
    X_train_matrix, X_valid_matrix, X_test_matrix, vectorizer = convert_to_numeric_matrices(X_train, X_valid, X_test)
    
    # Seleccionar i executar el model segons la configuració
    print(f"Entrenant el model: {MODEL_CHOICE}")
    if MODEL_CHOICE == "LR":
        predictions_test = LR.entrena_i_prediu(X_train_matrix, y_train, X_test_matrix)
    elif MODEL_CHOICE == "SVM":
        predictions_test = SVM.entrena_i_prediu(X_train_matrix, y_train, X_test_matrix)
    elif MODEL_CHOICE == "KNN":
        predictions_test = KNN.entrena_i_prediu(X_train_matrix, y_train, X_test_matrix)
    elif MODEL_CHOICE == "RF":
        predictions_test = RF.entrena_i_prediu(X_train_matrix, y_train, X_test_matrix)
    elif MODEL_CHOICE == "NB":
        predictions_test = NB.entrena_i_prediu(X_train_matrix, y_train, X_test_matrix)
    else:
        raise ValueError(f"Model no reconegut: {MODEL_CHOICE}")

    # Avaluar els resultats
    evaluar_resultats(y_test, predictions_test)

if __name__ == "__main__":
    main()
