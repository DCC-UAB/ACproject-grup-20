from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB, BernoulliNB, GaussianNB, CategoricalNB
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

EVALUATION_DIR = "C:/Users/marti/OneDrive/Escriptori/ProjecteAC/ACproject-grup-20/NB_evaluation"

def evaluar(y_true, y_pred, y_prob):
    """
    Calcula i mostra la matriu de confusió i altres mètriques d'avaluació.
    També guarda la matriu de confusió, la curva ROC i la curva Precision-Recall com a imatges.
    """
    # Matriu de confusió
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriu de confusió:")
    print(cm)

    # Mètriques
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Guardar la matriu de confusió com a imatge
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_true), yticklabels=np.unique(y_true))
    plt.title('Matriu de Confusió')
    plt.xlabel('Prediccions')
    plt.ylabel('Veritat')
    plt.savefig(f"{EVALUATION_DIR}/matriu_confusio.png")
    plt.close()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('Taxa de Falsos Positius')
    plt.ylabel('Taxa de Vertaders Positius')
    plt.title('Curva ROC')
    plt.legend(loc='lower right')
    plt.savefig(f"{EVALUATION_DIR}/curva_roc.png")
    plt.close()

    # Curva Precision-Recall
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_vals, precision_vals, color='b', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.savefig(f"{EVALUATION_DIR}/curva_precision_recall.png")
    plt.close()

def entrena_prediu_i_evalua(X_train, y_train, X_test, y_test):
    """
    Entrena un model Naive Bayes, genera les prediccions,
    i crida la funció 'evaluar' per mostrar els resultats.
    """
    # Definir el model: MULTINOMIAL (sutilitza en text)
    model = MultinomialNB()
    
    # Entrenar el model
    model.fit(X_train, y_train)
    
    # Generar prediccions
    predictions = model.predict(X_test)
    
    # Obtenir les probabilitats per a les curves
    y_prob = model.predict_proba(X_test)[:, 1]  # Seleccionar la probabilitat de la classe positiva
    
    # Avaluar resultats
    evaluar(y_test, predictions, y_prob)

    return predictions


##############################################################################################
#EVALUACIÓ % TRAIN
##############################################################################################
def comparar_accuracy_per_percentatge(X_train, y_train, X_test, y_test):
    """
    Compara l'accuracy del model Naive Bayes per diferents percentatges de les dades d'entrenament,
    tant pel conjunt d'entrenament com pel conjunt de test.
    Es seleccionen aleatòriament percentatges del 5%, 10%, 30%, 50%, 70%, 90%, 100% de les dades d'entrenament.
    """
    percentatges = [5, 10, 30, 50, 70, 90, 100]
    
    # Llistes per guardar els resultats de l'accuracy
    train_accuracies = []
    test_accuracies = []

    # Iterar sobre cada percentatge
    for percentatge in percentatges:
        # Calcular el percentatge de mostres
        train_size = percentatge / 100.0  # Convertir el percentatge a una fracció

        # Evitar que train_size sigui 1.0, ja que això vol dir que agafem tot el conjunt d'entrenament
        if train_size == 1.0:
            X_train_sub, y_train_sub = X_train, y_train
        else:
            # Seleccionar aleatòriament una part del conjunt d'entrenament
            X_train_sub, _, y_train_sub, _ = train_test_split(X_train, y_train, train_size=train_size, random_state=42)

        # Definir el model Naive Bayes (per exemple, MultinomialNB)
        model = MultinomialNB()
        
        # Entrenar el model amb el subset seleccionat
        model.fit(X_train_sub, y_train_sub)

        # Fer les prediccions pel conjunt d'entrenament
        y_train_pred = model.predict(X_train_sub)
        # Fer les prediccions pel conjunt de test
        y_test_pred = model.predict(X_test)

        # Calcular l'accuracy pel conjunt d'entrenament
        train_accuracy = accuracy_score(y_train_sub, y_train_pred)
        # Calcular l'accuracy pel conjunt de test
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Afegir els resultats a les llistes
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

    # Crear un DataFrame amb els resultats
    results_df = pd.DataFrame({
        'Percentatge': percentatges,
        'Train Accuracy': train_accuracies,
        'Test Accuracy': test_accuracies
    })

    # Mostrar els resultats
    print("\nResultats d'accuracy per diferents percentatges de dades d'entrenament:")
    print(results_df)

    # Generar gràfiques de comparació
    plt.figure(figsize=(10, 6))
    
    # Crear gràfic per a l'accuracy del conjunt d'entrenament i test
    plt.plot(percentatges, train_accuracies, marker='o', linestyle='-', label='Train Accuracy')
    plt.plot(percentatges, test_accuracies, marker='o', linestyle='-', label='Test Accuracy')
    
    # Afegir títol i etiquetes
    plt.title('Comparació de l\'Accuracy per percentatges de dades d\'entrenament')
    plt.xlabel('Percentatge del conjunt d\'entrenament')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # Modificar les etiquetes de l'eix X per mostrar percentatges
    plt.xticks(percentatges, [f'{x}%' for x in percentatges])

    # Guardar el gràfic a la carpeta d'evaluació
    plt.tight_layout()
    plt.savefig(f"{EVALUATION_DIR}/comparacio_accuracy_percentatge.png")
    plt.close()
    

##############################################################################################
#EVALUACIÓ PARÀMETRES
##############################################################################################

#MULTINOMIAL AMB ALTRES
def entrena_prediu_i_evaluaDiferentsNB(X_train, y_train, X_test, y_test):
    """
    Entrena 4 models Naive Bayes (MultinomialNB, BernoulliNB, GaussianNB, CategoricalNB),
    avalua les mètriques (accuracy, precision, recall, F1 score) i genera gràfiques de comparació.
    """
    # Convertir les matrius disperses a matrius denses (necessari per a alguns models com GaussianNB)
    X_train_dense = X_train.toarray() if hasattr(X_train, 'toarray') else X_train
    X_test_dense = X_test.toarray() if hasattr(X_test, 'toarray') else X_test

    # Models
    models = {
        'MultinomialNB': MultinomialNB(),
        'BernoulliNB': BernoulliNB(),
        'GaussianNB': GaussianNB(),
        'CategoricalNB': CategoricalNB()
    }

    # Llista per guardar els resultats
    results = []

    # Entrenar, predir i avaluar cada model
    for model_name, model in models.items():
        print(f"\nEntrenant i avaluant {model_name} amb valors per defecte...")

        # Entrenar el model
        model.fit(X_train_dense, y_train)
        
        # Generar prediccions
        y_pred = model.predict(X_test_dense)
        
        # Calcular les mètriques
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Afegir els resultats a la llista
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })

    # Crear un DataFrame amb els resultats
    results_df = pd.DataFrame(results)
    print("\nResultats finals:")
    print(results_df)

    # Generar gràfiques de comparació
    sns.set(style="whitegrid")

    # Crear una figura i eixos
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Mètriques a comparar
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        sns.barplot(x='Model', y=metric, data=results_df, ax=ax)
        ax.set_title(f'Comparació de {metric}')
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f"{EVALUATION_DIR}/comparacio_metricas.png")
    plt.close()
