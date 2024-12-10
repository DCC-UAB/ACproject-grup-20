# Sentiment Analysis : Movies

## 1. Models

### 1.1 KNN

Primera execució model KNN, n_neighbors=5 :

temps preocessar 124.6710159778595

Matriu de confusió:
[[1727  768]
 [ 481 2024]]

Accuracy: 0.7502
Precision: 0.7535
Recall: 0.7502
F1 Score: 0.7493
temps entrenament 22.811246871948242

Segona execució model KNN, n_neighbors=10 :
temps preocessar 256.92059874534607

Matriu de confusió:
[[1924  571]
 [ 588 1917]]

Accuracy: 0.7682
Precision: 0.7682
Recall: 0.7682
F1 Score: 0.7682
temps entrenament 48.164021253585815

Tercera execució model KNN, n_neighbors=5, plots inclosos :

temps preocessar 109.44386577606201

Matriu de confusió:
[[1727  768]
 [ 481 2024]]
![Matriu de confusió](KNN_evaluation/CM_KNN.png)
Accuracy: 0.7502
Precision: 0.7535
Recall: 0.7502
![Curva ROC + Precision-Recall](KNN_evaluation/ROC_PRECISIONRECALL_KNN.png)
F1 Score: 0.7493
temps entrenament 150.65066742897034

Quarta execució model KNN, n_neighbors=20, plots inclosos :
Temps trigat a processar les dades : 122.23539853096008

Matriu de confusió:
[[1907  588]
 [ 519 1986]]
![Matriu de confusió](KNN_evaluation/CM_KNN20.png)
Accuracy: 0.7786
Precision: 0.7788
Recall: 0.7786
![Curva ROC + Precision-Recall](KNN_evaluation/ROC_PRECISIONRECALL_KNN20.png)
F1 Score: 0.7786
temps entrenament 156.88142585754395
### 1.2 LR

### 1.3 NB

### 1.4 RF

### 1.5 SVM

