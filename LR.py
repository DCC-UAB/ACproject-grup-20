#llista max iters esta be?
#tindria sentit fer un heatmap dels parametrees ? de quins? o tots

import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time

EVALUATION_DIR = "C:/Users/marti/OneDrive/Escriptori/ProjecteAC/ACproject-grup-20/LR_evaluation"

def evaluar(y_true, y_pred, y_proba):
    """
    Calcula i mostra la matriu de confusió i altres mètriques d'avaluació.
    També genera visualitzacions (heatmap de la matriu de confusió i la ROC curve).
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


    #MATRIU CONFUSIO
    # Visualització de la matriu de confusió com a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=set(y_true), yticklabels=set(y_true))
    plt.title('Matriu de confusió')
    plt.xlabel('Prediccions')
    plt.ylabel('Valors reals')

    # guardar matriu confusio
    training_plot_path = os.path.join(EVALUATION_DIR, "matriu_confusio.png")
    plt.savefig(training_plot_path)
    print(f"Matriu de confusió guardada a {training_plot_path}")
    plt.close()


    #ROC CURVE
    # Càlcul de la curva ROC
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    print(f"\nAUC (Area Under the Curve): {auc_score:.4f}")

    # Visualització de la curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Línia diagonal
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')

    # Guardar la ROC curve
    roc_plot_path = os.path.join(EVALUATION_DIR, "roc_curve.png")
    plt.savefig(roc_plot_path)
    print(f"ROC curve guardada a {roc_plot_path}")
    plt.close()


    #PRECISION - RECALL CURVE
    # Càlcul de la Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc_score = auc(recall, precision)
    print(f"\nAUC de Precision-Recall: {pr_auc_score:.4f}")

    # Visualització de la Precision-Recall Curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Precision-Recall Curve (AUC = {pr_auc_score:.2f})', color='green')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')

    # Guardar la Precision-Recall curve
    pr_plot_path = os.path.join(EVALUATION_DIR, "precision_recall_curve.png")
    plt.savefig(pr_plot_path)
    print(f"Precision-Recall curve guardada a {pr_plot_path}")
    plt.close()

def entrena_prediu_i_evalua(X_train, y_train, X_test, y_test):
    """
    Entrena un model de regressió logística, genera les prediccions,
    i crida la funció 'evaluar' per mostrar els resultats.
    """
    # Definir el model
    C_value = 1.0  # Valor fix de C
    model = LogisticRegression(C=C_value, solver='liblinear', max_iter=5000, penalty='l2')
    
    # Entrenar el model
    model.fit(X_train, y_train)
    
    # Generar prediccions i probabilitats
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]  # Probabilitats per a la classe positiva

    # Avaluar resultats
    evaluar(y_test, predictions, probabilities)

    return predictions

##############################################################################################
#EVALUACIÓ PARÀMETRES
##############################################################################################

#GRIDSEARCH
def entrena_prediu_i_evaluaGridSearch(X_train, y_train, X_test, y_test):
    """
    Entrena un model de regressió logística amb GridSearchCV per buscar els millors hiperparàmetres,
    avalua el millor model trobat i mostra els resultats, i els desa en un fitxer de text.
    """
    # Definir l'espai de cerca dels hiperparàmetres
    parameters = {
        'max_iter': [500, 1000, 5000],         # Límit màxim d'iteracions per ajustar el model.
        'C': [0.01, 0.1, 1, 10, 100],          # Paràmetre de regularització inversa (menor valor = més regularització).
        'penalty': ['l2', 'l1', 'elasticnet'], # Tipus de penalització utilitzada.
        'solver': ["saga", "liblinear"]        # Algorismes per optimitzar el model.
    }

    # Inicialitzar el model base de regressió logística
    lr = LogisticRegression()

    # Configurar GridSearchCV per buscar els millors hiperparàmetres
    clf_ = GridSearchCV(
        lr,                    # Model base
        parameters,            # Diccionari amb els paràmetres a provar
        n_jobs=-1,             # Número de processos paral·lels (ús màxim del sistema)
        cv=5,                  # 5 folds, iteracio entrenament-validacio, mitjana metriques
        scoring='accuracy'     # Mètrica d'avaluació
    )

    # Entrenar el model amb GridSearch utilitzant les dades d'entrenament
    clf_.fit(X_train, y_train)

    # Avaluar el millor model trobat sobre el conjunt de test i entrenament
    test_score = clf_.score(X_test, y_test)   # Accuracy sobre el conjunt de test
    train_score = clf_.score(X_train, y_train) # Accuracy sobre el conjunt d'entrenament

    # Resultats a imprimir
    results = [
        f"Accuracy sobre el conjunt de test: {test_score:.4f}",
        f"Accuracy sobre el conjunt d'entrenament: {train_score:.4f}",
        f"Millor accuracy en validacio creuada: {clf_.best_score_:.4f}",
        f"Millors hiperparametres trobats: {clf_.best_params_}"
    ]

    # Mostrar els resultats d'avaluació
    for result in results:
        print(result)
    
    # Guardar els resultats en un arxiu de text
    result_file_path = os.path.join(EVALUATION_DIR, "resultats_gridsearch.txt")
    with open(result_file_path, "w") as file:
        for result in results:
            file.write(result + "\n")

    # Exploració dels atributs interns del model GridSearchCV
    print("Claus disponibles a l'objecte GridSearchCV:")
    print(clf_.__dict__.keys())

    return clf_

#MAX_ITER
def entrena_prediu_i_evaluaMaxIter(X_train, y_train, X_test, y_test): 
    """
    Entrena el model de regressió logística amb diferents valors de max_iter
    i mostra com afecta al temps d'entrenament, l'accuracy, la precision, i genera matrius de confusió.
    """
    max_iter_values = [10, 25, 50, 100, 500, 1000, 2000, 5000]

    training_times = []
    accuracies = []
    precisions = []
    confusion_matrices = []

    for max_iter in max_iter_values:
        start_time = time.time()  # Mesura del temps d'entrenament
        model = LogisticRegression(max_iter=max_iter, solver='liblinear', C=1.0, penalty='l2')
        model.fit(X_train, y_train)
        end_time = time.time()
        
        # Temps d'entrenament
        training_time = end_time - start_time
        training_times.append(training_time)

        # Accuracy en test
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        # Precision
        precision = precision_score(y_test, y_pred, average='binary')  # Pots canviar a 'macro' o 'micro' si tens més de 2 classes
        precisions.append(precision)

        # Matriu de confusió
        conf_matrix = confusion_matrix(y_test, y_pred)
        confusion_matrices.append(conf_matrix)

        print(f"max_iter={max_iter}: temps={training_time:.2f} segons, accuracy={accuracy:.4f}, precision={precision:.4f}")

    # Gràfica del temps d'entrenament
    plt.figure(figsize=(10, 6))
    plt.plot(max_iter_values, training_times, marker='o', color='blue')
    plt.title('Impacte de max_iter en el temps d\'entrenament')
    plt.xlabel('max_iter')
    plt.ylabel('Temps d\'entrenament (s)')
    plt.grid(True)
    training_plot_path = os.path.join(EVALUATION_DIR, "MAX_ITER_temps_entrenament.png")
    plt.savefig(training_plot_path)
    plt.close()

    # Gràfica de l'accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(max_iter_values, accuracies, marker='o', color='orange')
    plt.title('Impacte de max_iter en l\'accuracy')
    plt.xlabel('max_iter')
    plt.ylabel('Accuracy')
    plt.grid(True)
    accuracy_plot_path = os.path.join(EVALUATION_DIR, "MAX_ITER_accuracy.png")
    plt.savefig(accuracy_plot_path)
    plt.close()

    # Gràfica de la precision
    plt.figure(figsize=(10, 6))
    plt.plot(max_iter_values, precisions, marker='o', color='green')
    plt.title('Impacte de max_iter en la precision')
    plt.xlabel('max_iter')
    plt.ylabel('Precision')
    plt.grid(True)
    precision_plot_path = os.path.join(EVALUATION_DIR, "MAX_ITER_precision.png")
    plt.savefig(precision_plot_path)
    plt.close()

    # Matrius de confusió
    plt.figure(figsize=(15, 12))

    # Configurar per 2 files i 4 columnes per a 8 subgràfics
    for i, conf_matrix in enumerate(confusion_matrices):
        plt.subplot(2, 4, i + 1)  # 2 files, 4 columnes
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', xticklabels=['Predicció 0', 'Predicció 1'], yticklabels=['Real 0', 'Real 1'])
        plt.title(f'Matriu de Confusió per max_iter={max_iter_values[i]}')

    # Desar la imatge amb les matrius de confusió
    confusion_matrix_plot_path = os.path.join(EVALUATION_DIR, "matrius_de_confusio.png")
    plt.tight_layout()
    plt.savefig(confusion_matrix_plot_path)
    plt.close()

    print(f"Gràfiques desades")

#C
def entrena_prediu_i_evaluaImpactC(X_train, y_train, X_test, y_test):
    """
    Entrena un model de regressió logística amb diferents valors de C, 
    mostra com afecta al temps d'entrenament, l'accuracy, precision, F1 score i recall,
    i genera les gràfiques de línies per mostrar aquests efectes en funció del valor de C.
    """
    # Definir els valors de C a provar
    C_values = [0.01, 0.1, 1, 10, 100, 1000]

    # Llistes per emmagatzemar els resultats
    training_times = []
    accuracies = []
    precisions = []
    f1_scores = []
    recalls = []

    # Entrenar i avaluar el model per cada valor de C
    for C in C_values:
        start_time = time.time()  # Mesura del temps d'entrenament
        model = LogisticRegression(max_iter=1000, C=C, penalty='l2', solver='liblinear')
        model.fit(X_train, y_train)
        end_time = time.time()

        # Temps d'entrenament
        training_time = end_time - start_time
        training_times.append(training_time)

        # Predicció i càlcul de mètriques
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')  # Precision per a classificació binària
        f1 = f1_score(y_test, y_pred, average='binary')  # F1 score per a classificació binària
        recall = recall_score(y_test, y_pred, average='binary')  # Recall per a classificació binària

        accuracies.append(accuracy)
        precisions.append(precision)
        f1_scores.append(f1)
        recalls.append(recall)

        print(f"C={C}: Temps={training_time:.2f} segons, Accuracy={accuracy:.4f}, Precision={precision:.4f}, F1={f1:.4f}, Recall={recall:.4f}")

    # Gràfica de les mètriques (Accuracy, Precision, F1 Score, Recall) en funció de C
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, accuracies, marker='o', label='Accuracy', color='blue')
    plt.plot(C_values, precisions, marker='o', label='Precision', color='red')
    plt.plot(C_values, f1_scores, marker='o', label='F1 Score', color='green')
    plt.plot(C_values, recalls, marker='o', label='Recall', color='purple')
    plt.xscale('log')
    plt.title('Impacte de C en Accuracy, Precision, F1 Score i Recall')
    plt.xlabel('Valor de C')
    plt.ylabel('Mètriques')
    plt.legend()
    plt.grid(True)
    metrics_plot_path = os.path.join(EVALUATION_DIR, "C_metrics.png")
    plt.savefig(metrics_plot_path)
    plt.close()

    # Gràfica del temps d'entrenament en funció de C
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, training_times, marker='o', color='orange')
    plt.xscale('log')
    plt.title('Impacte de C en el temps d\'entrenament')
    plt.xlabel('Valor de C')
    plt.ylabel('Temps d\'entrenament (s)')
    plt.grid(True)
    time_plot_path = os.path.join(EVALUATION_DIR, "C_temps.png")
    plt.savefig(time_plot_path)
    plt.close()

    print(f"Gràfiques desades")

###############################################################################################

#RESUTLATS GRIDSEARCH:
'''
Accuracy sobre el conjunt de test: 0.8900
Accuracy sobre el conjunt d'entrenament: 0.9100
Millor accuracy en validació creuada: 0.8853
Millors hiperparàmetres trobats: {'C': 1, 'max_iter': 500, 'penalty': 'l2', 'solver': 'liblinear'}
Claus disponibles a l'objecte GridSearchCV:
dict_keys(['scoring', 'estimator', 'n_jobs', 'refit', 'cv', 'verbose', 'pre_dispatch', 
    'error_score', 'return_train_score', 'param_grid', 'multimetric_', 'best_index_', 
    'best_score_', 'best_params_', 'best_estimator_', 'refit_time_', 'scorer_', 
    'cv_results_', 'n_splits_'])       
temps entrenament 4144.591024875641
'''


#a fer:
'''
Anàlisi de l'espai de cerca:
Heatmap que mostri l'accuracy obtinguda en funció de dos hiperparàmetres (p. ex., penalty i solver).
'''

