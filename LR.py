#gridsearch, valors de c
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def evaluar(y_true, y_pred):
    """
    Calcula i mostra la matriu de confusió i altres mètriques d'avaluació.
    """
    # Matriu de confusió
    cm = confusion_matrix(y_true, y_pred)
    print("\nMatriu de confusió:")
    print(cm)

    # metriques
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

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

def entrena_prediu_i_evaluaG(X_train, y_train, X_test, y_test):
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


#RESUTLATS GRIDSEARCH:
'''
Accuracy sobre el conjunt de test: 0.8900
Accuracy sobre el conjunt d'entrenament: 0.9100
Millor accuracy en validació creuada: 0.8853
Millors hiperparàmetres trobats: {'C': 1, 'max_iter': 500, 'penalty': 'l2', 'solver': 'liblinear'}
Claus disponibles a l'objecte GridSearchCV:
dict_keys(['scoring', 'estimator', 'n_jobs', 'refit', 'cv', 'verbose', 'pre_dispatch', 'error_score', 'return_train_score', 'param_grid', 'multimetric_', 'best_index_', 'best_score_', 'best_params_', 'best_estimator_', 'refit_time_', 'scorer_', 'cv_results_', 'n_splits_'])       
temps entrenament 4144.591024875641
'''