import langid

# Lista de frases para detectar el idioma
frases = [
    'Good evening, I hope you had a great day',
    'Hoy es un día perfecto para aprender algo nuevo',
    'Bon dia, com estàs avui?',
    'Bonjour, je suis heureux de vous rencontrer',
    'Guten Morgen, wie geht es Ihnen?'
]

# Diccionario de idiomas para traducir los códigos a nombres de idiomas
idiomas = {'ca': 'català', 'es': 'castellà', 'en': 'anglès', 'fr': 'francès', 'de': 'alemany'}

# Detectar y mostrar resultados para cada frase
for frase in frases:
    idioma, _ = langid.classify(frase)  
    idioma_nombre = idiomas.get(idioma, 'Idioma desconocido')
    print(f"Frase: {frase}")
    print(f"Idioma detectado: {idioma} ({idioma_nombre})")
    print()
