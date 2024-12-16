import pandas as pd
import langid

# Funció per detectar idiomes
def detect_languages(data, column_name):
    detected_languages = data[column_name].apply(lambda x: langid.classify(x)[0])
    return detected_languages

# Llegir les dades
test_data = pd.read_csv('C:/Users/marti/OneDrive/Escriptori/datasets_AC/Test.csv')
train_data = pd.read_csv('C:/Users/marti/OneDrive/Escriptori/datasets_AC/Train.csv')
valid_data = pd.read_csv('C:/Users/marti/OneDrive/Escriptori/datasets_AC/Valid.csv')

# Detectar idiomes en cada arxiu
test_languages = detect_languages(test_data, "text")
train_languages = detect_languages(train_data, "text")
valid_languages = detect_languages(valid_data, "text")

# Filtrar les línies que no són en anglès
test_non_english = test_data[test_languages != 'en']
train_non_english = train_data[train_languages != 'en']
valid_non_english = valid_data[valid_languages != 'en']

# Mostra els resultats
print("Línies no en anglès a Test.csv:")
print(test_non_english[['text']])
print("Línies no en anglès a Train.csv:")
print(train_non_english[['text']])
print("Línies no en anglès a Valid.csv:")
print(valid_non_english[['text']])
