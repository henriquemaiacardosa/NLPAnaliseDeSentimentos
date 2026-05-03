import nltk

avaliacoes_imdb_treino = [
    ('Esse filme foi espetacular e muito emocionante', 'positivo'),
    ('A atuação do protagonista foi excelente', 'positivo'),
    ('Eu adorei a trilha sonora e a direção', 'positivo'),
    ('Um dos melhores filmes que já vi na vida', 'positivo'),
    ('O enredo é fantástico e prende a atenção', 'positivo'),
    ('Achei o filme péssimo e muito chato', 'negativo'),
    ('O roteiro é terrível e cheio de falhas', 'negativo'),
    ('Odiei a atuação de todos os atores', 'negativo'),
    ('Não recomendo esse filme para ninguém, perda de tempo', 'negativo'),
    ('Um desastre total, efeitos especiais horríveis', 'negativo')
]

avaliacoes_imdb_teste = [
    ('O filme foi excelente, muito bom', 'positivo'),
    ('Achei a história fantástica', 'positivo'),
    ('O filme é muito chato e terrível', 'negativo'),
    ('Péssimo roteiro e atores ruins', 'negativo')
]

stopwords_pt = nltk.corpus.stopwords.words('portuguese')

def processar_texto_imdb(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frases_processadas = []
    for (palavras, sentimento) in texto:
        com_stemming = [str(stemmer.stem(p)) for p in palavras.split() if p.lower() not in stopwords_pt]
        frases_processadas.append((com_stemming, sentimento))
    return frases_processadas

treino_processado = processar_texto_imdb(avaliacoes_imdb_treino)
teste_processado = processar_texto_imdb(avaliacoes_imdb_teste)

todas_palavras_imdb = []
for (palavras, sentimento) in treino_processado:
    todas_palavras_imdb.extend(palavras)

frequencia_imdb = nltk.FreqDist(todas_palavras_imdb)
palavras_unicas_imdb = frequencia_imdb.keys()

def extrair_features_imdb(documento):
    doc_set = set(documento)
    features = {}
    for palavra in palavras_unicas_imdb:
        features['contem(%s)' % palavra] = (palavra in doc_set)
    return features

base_treino_features = nltk.classify.apply_features(extrair_features_imdb, treino_processado)
base_teste_features = nltk.classify.apply_features(extrair_features_imdb, teste_processado)

classificador_imdb = nltk.NaiveBayesClassifier.train(base_treino_features)

print("\n--- Resultados do Classificador IMDb (Positivo/Negativo) ---")
print(f"Acurácia do modelo: {nltk.classify.accuracy(classificador_imdb, base_teste_features) * 100}%")

print("\nAnalisando frases soltas:")
frase_teste_1 = "Eu achei o filme excelente"
features_teste_1 = extrair_features_imdb([str(nltk.stem.RSLPStemmer().stem(p)) for p in frase_teste_1.split()])
print(f"A frase '{frase_teste_1}' foi classificada como: {classificador_imdb.classify(features_teste_1)}")

frase_teste_2 = "O roteiro era péssimo e chato"
features_teste_2 = extrair_features_imdb([str(nltk.stem.RSLPStemmer().stem(p)) for p in frase_teste_2.split()])
print(f"A frase '{frase_teste_2}' foi classificada como: {classificador_imdb.classify(features_teste_2)}")