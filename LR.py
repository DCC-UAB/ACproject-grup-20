#si faig gridsearch, fa falta comprobar unicament amb valors diferents de c?

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import time

def evaluar(y_true, y_pred):
    """
    Calcula i mostra la matriu de confusió i altres mètriques d'avaluació.
    També genera una visualització (heatmap) de la matriu de confusió.
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

    '''
    # Visualització de la matriu de confusió com a heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=set(y_true), yticklabels=set(y_true))
    plt.title('Matriu de confusió')
    plt.xlabel('Prediccions')
    plt.ylabel('Valors reals')
    plt.show()
    '''

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
    
    # Generar prediccions
    predictions = model.predict(X_test)

    # Avaluar resultats
    evaluar(y_test, predictions)

    return predictions

def entrena_prediu_i_evaluaGridSearch(X_train, y_train, X_test, y_test):
    """
    Entrena un model de regressió logística amb GridSearchCV per buscar els millors hiperparàmetres,
    avalua el millor model trobat i mostra els resultats.
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
        cv=5,                  # Validació creuada amb 5 particions
        scoring='accuracy'     # Mètrica d'avaluació
    )

    # Entrenar el model amb GridSearch utilitzant les dades d'entrenament
    clf_.fit(X_train, y_train)

    # Avaluar el millor model trobat sobre el conjunt de test i entrenament
    test_score = clf_.score(X_test, y_test)   # Accuracy sobre el conjunt de test
    train_score = clf_.score(X_train, y_train) # Accuracy sobre el conjunt d'entrenament

    # Mostrar els resultats d'avaluació
    print(f"Accuracy sobre el conjunt de test: {test_score:.4f}")
    print(f"Accuracy sobre el conjunt d'entrenament: {train_score:.4f}")

    # Millor puntuació obtinguda durant la validació creuada
    print(f"Millor accuracy en validació creuada: {clf_.best_score_:.4f}")

    # Millor combinació d'hiperparàmetres trobada
    print("Millors hiperparàmetres trobats:", clf_.best_params_)

    # Exploració dels atributs interns del model GridSearchCV
    print("Claus disponibles a l'objecte GridSearchCV:")
    print(clf_.__dict__.keys())

    return clf_

def entrena_prediu_i_evaluaMaxIter(X_train, y_train, X_test, y_test): #MODIFICAR VALORS MAX ITER
    """
    Entrena el model de regressió logística amb diferents valors de max_iter
    i mostra com afecta al temps d'entrenament i a l'accuracy.
    """
    max_iter_values = [100, 500, 1000, 2000, 5000]

    training_times = []
    accuracies = []

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

        print(f"max_iter={max_iter}: temps={training_time:.2f} segons, accuracy={accuracy:.4f}")

    # Gràfica dels resultats
    plt.figure(figsize=(10, 6))
    plt.plot(max_iter_values, training_times, label='Temps d\'entrenament (s)', marker='o', color='blue')
    plt.plot(max_iter_values, accuracies, label='Accuracy', marker='o', color='orange')
    plt.title('Impacte de max_iter en el temps d\'entrenament i l\'accuracy')
    plt.xlabel('max_iter')
    plt.ylabel('Temps (s) / Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

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
Impacte del paràmetre C:
Gràfica de línies mostrant com varia l'accuracy, el F1 Score o altres mètriques en funció del valor de C.
Eix X: Valors de C.
Eix Y: Mètrica (accuracy, F1, etc.).

Anàlisi de l'espai de cerca:
Heatmap que mostri l'accuracy obtinguda en funció de dos hiperparàmetres (p. ex., penalty i solver).

Convergència del model:
Si varies max_iter, una gràfica per veure com afecta al temps d'entrenament o a la mètrica d'accuracy.
'''