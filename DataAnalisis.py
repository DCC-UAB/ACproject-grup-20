import pandas as pd
import langid
import matplotlib.pyplot as plt

# Funció per detectar idiomes
def detectar_idiomes(data, column_name):
    detected_languages = data[column_name].apply(lambda x: langid.classify(x)[0])
    
    # Mostrar les línies i els índexs dels textos no en anglès
    non_english_rows = data[detected_languages != 'en']
    if not non_english_rows.empty:
        print("\nLínies que no estan en anglès (índex del CSV):")
        print(non_english_rows.index.tolist())  # Mostra els índexs de les línies no angleses
    
    print('noenglish:', non_english_rows)
    
    return detected_languages.value_counts()

# Funció per mostrar la distribució de les etiquetes
def mostrar_distribucio_labels(data, column_name):
    distribucio = data[column_name].value_counts()
    return distribucio

# Funció per mostrar la distribució de les etiquetes com a gràfica
def mostrar_grafica_distribucio_labels(distribucio, title):
    distribucio.plot(kind='bar', color=['skyblue', 'lightcoral'])
    plt.title(title)
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.show()

# Funció principal
def main():
    # Carregar els fitxers CSV
    test_data = pd.read_csv('C:/Users/marti/OneDrive/Escriptori/datasets_AC/Test.csv')
    train_data = pd.read_csv('C:/Users/marti/OneDrive/Escriptori/datasets_AC/Train.csv')
    valid_data = pd.read_csv('C:/Users/marti/OneDrive/Escriptori/datasets_AC/Valid.csv')
    
    # Detectar idiomes en cada fitxer
    print("Detectant idiomes en Test.csv...")
    test_languages = detectar_idiomes(test_data, "text")
    print("Detectant idiomes en Train.csv...")
    train_languages = detectar_idiomes(train_data, "text")
    print("Detectant idiomes en Valid.csv...")
    valid_languages = detectar_idiomes(valid_data, "text")

    # Mostrar la distribució d'idiomes
    print("\nIdiomes en Test.csv:", test_languages)
    print("Idiomes en Train.csv:", train_languages)
    print("Idiomes en Valid.csv:", valid_languages)
    
    # Mostrar la distribució de positius i negatius en cada dataset
    print("\nDistribució de les etiquetes en Test.csv:")
    test_label_dist = mostrar_distribucio_labels(test_data, 'label')
    print(test_label_dist)
    mostrar_grafica_distribucio_labels(test_label_dist, "Distribució de les etiquetes en Test.csv")
    
    print("\nDistribució de les etiquetes en Train.csv:")
    train_label_dist = mostrar_distribucio_labels(train_data, 'label')
    print(train_label_dist)
    mostrar_grafica_distribucio_labels(train_label_dist, "Distribució de les etiquetes en Train.csv")
    
    print("\nDistribució de les etiquetes en Valid.csv:")
    valid_label_dist = mostrar_distribucio_labels(valid_data, 'label')
    print(valid_label_dist)
    mostrar_grafica_distribucio_labels(valid_label_dist, "Distribució de les etiquetes en Valid.csv")

# Cridar la funció principal
if __name__ == "__main__":
    main()
