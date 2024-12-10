[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/USx538Ll)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-2e0aaae1b6195c2367325f4f02e2d04e9abb55f0b24a779b69b11b9e10269abc.svg)](https://classroom.github.com/online_ide?assignment_repo_id=17348992&assignment_repo_type=AssignmentRepo)

SETMANA 1 (respòs):
    - té sentit que utlitzem stemming i lemmatize alhora? 
    RESPOSTA: En el nostre cas no. 
    Un cop lemmatitzada una paraula, ja obtenim al seva forma base correcta. 
    També hem comprobat que es perd la qualitat del text al stemmatitzar un cop lemmatitzat. A ProbaMini es mostra com no aporta informació nova (escurça paraules ja en la seva base) i alhora produeix moltes formes incorrectes de les respectives paraules. 
    Per tant, l'ordre òptim és només utilitzar lemmatizer.

    - sense probar grans combinacions amb els respectius parametres de cada model, quin té una millor accuracy i/o precission?

    - s'eviten les paraules innecessàries en el preprocessament? 

SETMANA 2 (preguntes a respondre a partir de la seguent sessió):
    - TFIDF dona pes a les paraules més rellevants segons la frequencia d'una paraula i la seva importancia global, indiferentment del seu significat. Al tractar amb comentaris positius o negatius, milloraria el model si multipliquessim certes paraules que escollirirem no arbitrariament? Per ex., family pot ser una paraula important que surt en molts commentaris (positius sobretot, per ex). Tot i això no mostra sentiment. Si multipliquessim les paraules que si que sabem que mostren sentiment (com good), apart de fer-les mes rellevants no fariem tambe més irrellevants la resta de paraules? Em refereixo a tenir un aventatge, no sé si milloraria el model. Tot i això, family potser no apareix tant com bad, i per tant, el propi vectorizer ja els dona els pesos correctes i no s'ha de tocar res. 
    RESPOSTA:
    
    Creiem que el model NO milloraria si multipliquessim certes paraules escollides principalment per 2 motius:
    1. Introducció de biaix : Si multipliquem el pes de paraules manualment podriem introduir biaix en el model fent-lo dependre massa d'unes quantes paraules seleccionades, el qual podria portar a prediccions incorrectes en contextos on aquestes paraules tinguin un significat diferent.
    2. Irrevelancies : Donar més pes a algunes paraules fa que la resta siguin menys rellevants lo qual no sempre és desitjable.
    Per exemple, un text pot tenir una paraula clau (com good), però la resta del context també pot ser important per interpretar el sentiment global.

    FALTA REALITZAR TESTS PER CONFIRMAR!!

    Una alternativa que pensem que podria ser útil és l'ús de word embeddings. (?)

    - quins son els millors parametres en cada model i com influeixen en el temps/metriques d'avaluacio?

    - utilitzant la millor combinació de parametres, quin es el model mes rapid? i el que te millor accuracy?

    - si la matriu de confusio no varia respecte al parametre max_iteracions, què significa?
    RESPOSTA: el model convergeix abans d'aarribar al nombre màxim d'iteracions. Probar: El model és molt bo des de l'inici.El conjunt de dades és senzill. Problemes d'optimització del model.Regularització.Paràmetres del model i la qualitat del model.Desajust entre les mètriques de rendiment.