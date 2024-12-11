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

Cinquena execució model KNN, n_neighbors=200 (sqrt(n_mostres)), plots inclosos :
Temps trigat a processar les dades : 145.36676931381226
model utilitzat: KNN

Matriu de confusió:
[[2052  443]
 [ 471 2034]]
![Matriu de confusió](KNN_evaluation/CM_KNN200.png)
Accuracy: 0.8172
Precision: 0.8172
Recall: 0.8172
![Curva ROC + Precision-Recall](KNN_evaluation/ROC_PRECISIONRECALL_KNN200.png)
F1 Score: 0.8172
temps entrenament 817.1725895404816

Testing:
k=1: Accuracy mitjana=0.7045
k=2: Accuracy mitjana=0.7032
k=3: Accuracy mitjana=0.7259
k=4: Accuracy mitjana=0.7291
k=5: Accuracy mitjana=0.7346
k=6: Accuracy mitjana=0.7405
k=7: Accuracy mitjana=0.7447
k=8: Accuracy mitjana=0.7485
k=9: Accuracy mitjana=0.7514
k=10: Accuracy mitjana=0.7539
k=11: Accuracy mitjana=0.7554
k=12: Accuracy mitjana=0.7592
k=13: Accuracy mitjana=0.7592
k=14: Accuracy mitjana=0.7634
k=15: Accuracy mitjana=0.7622
k=16: Accuracy mitjana=0.7634
k=17: Accuracy mitjana=0.7642
k=18: Accuracy mitjana=0.7675
k=19: Accuracy mitjana=0.7672
k=20: Accuracy mitjana=0.7706
k=21: Accuracy mitjana=0.7709
k=22: Accuracy mitjana=0.7718
k=23: Accuracy mitjana=0.7721
k=24: Accuracy mitjana=0.7747
k=25: Accuracy mitjana=0.7742
k=26: Accuracy mitjana=0.7753
k=27: Accuracy mitjana=0.7758
k=28: Accuracy mitjana=0.7776
k=29: Accuracy mitjana=0.7773
k=30: Accuracy mitjana=0.7792
k=31: Accuracy mitjana=0.7802
k=32: Accuracy mitjana=0.7801
k=33: Accuracy mitjana=0.7801
k=34: Accuracy mitjana=0.7802
k=35: Accuracy mitjana=0.7821
k=36: Accuracy mitjana=0.7816
k=37: Accuracy mitjana=0.7823
k=38: Accuracy mitjana=0.7829
k=39: Accuracy mitjana=0.7831
k=40: Accuracy mitjana=0.7850
k=41: Accuracy mitjana=0.7849
k=42: Accuracy mitjana=0.7861
k=43: Accuracy mitjana=0.7854
k=44: Accuracy mitjana=0.7863
k=45: Accuracy mitjana=0.7867
k=46: Accuracy mitjana=0.7863
### 1.2 LR

### 1.3 NB

### 1.4 RF

### 1.5 SVM

