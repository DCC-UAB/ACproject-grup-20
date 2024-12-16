import pandas as pd
import langid

# Función para detectar idiomas
def detect_languages(data, column_name):
    detected_languages = data[column_name].apply(lambda x: langid.classify(x)[0])
    return detected_languages.value_counts()


test_data = pd.read_csv('C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/Test.csv')
train_data = pd.read_csv('C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/Train.csv')
valid_data = pd.read_csv('C:/Users/Almoujtaba/Desktop/CARRERA/ACproject-grup-20/datasets_AC/Valid.csv')

# Detectar idiomas en cada archivo
test_languages = detect_languages(test_data, "text")
train_languages = detect_languages(train_data, "text")
valid_languages = detect_languages(valid_data, "text")

#Resultados
print("Idiomas en Test.csv:", test_languages)
print("Idiomas en Train.csv:", train_languages)
print("Idiomas en Valid.csv:", valid_languages)

####Casi todas son en ingles, 3 que no son.: italiano, noruego y indonesia.