import nltk
from nltk.metrics import ConfusionMatrix

nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')

basetreinamento = [
    ('me acho completamente amado', 'alegria'),
    ('amor é tremendo', 'alegria'),
    ('estou me sentindo muito animado novamente', 'alegria'),
    ('eu estou perfeito hoje', 'alegria'),
    ('o dia está muito bonito', 'alegria'),
    ('estou contente com o resultado do teste', 'alegria'),
    ('o amor é lindo', 'alegria'),
    ('nossa amizade e amor vai durar para sempre', 'alegria'),
    ('uma dor achada, meu deus', 'pavor'),
    ('receio de sair', 'pavor'),
    ('a noite é muito estranha', 'pavor'),
    ('estou estremecendo com esta casa', 'pavor'),
    ('isso me apavora', 'pavor'),
    ('estou apavorado', 'pavor'),
    ('ele esta me ameaçando a dias', 'pavor'),
    ('isso me deixa apavorada', 'pavor'),
    ('este lugar é apavorante', 'pavor'),
    ('estou abatida', 'tristeza'),
    ('a ansiedade tomou conta de mim', 'tristeza'),
    ('as pessoas não gostam do meu jeito', 'tristeza'),
    ('adeus passamos bons momentos juntos', 'tristeza'),
    ('acho sua falta', 'tristeza'),
    ('ele não gostou da minha comida', 'tristeza'),
    ('como me acho culpada', 'tristeza')
]

baseparatestes = [
    ('me acho completamente amado', 'alegria'),
    ('amor é tremendo', 'alegria'),
    ('estou me sentindo muito animado novamente', 'alegria'),
    ('estou apavorado', 'pavor'),
    ('ele esta me ameaçando a dias', 'pavor'),
    ('isso me deixa apavorada', 'pavor'),
    ('estou abatida', 'tristeza'),
    ('a dor tomou conta de mim', 'tristeza'),
    ('as pessoas não gostam do meu jeito', 'tristeza')
]

stopwordsnltk = nltk.corpus.stopwords.words('portuguese')

def aplicarstemmer(texto):
    stemmer = nltk.stem.RSLPStemmer()
    frasesstemming = []
    for (palavras, emocoes) in texto:
        comstemming = [str(stemmer.stem(p)) for p in palavras.split() if p not in stopwordsnltk]
        frasesstemming.append((comstemming, emocoes))
    return frasesstemming

frasescomstemmingtreinamento = aplicarstemmer(basetreinamento)
frasescomstemmingteste = aplicarstemmer(baseparatestes)

def buscapalavras(frases):
    todaspalavras = []
    for (palavras, emocao) in frases:
        todaspalavras.extend(palavras)
    return todaspalavras

def buscarfrequencia(palavras):
    frequencia = nltk.FreqDist(palavras)
    return frequencia

def procurapalavrasunicas(frequencia):
    freq = frequencia.keys()
    return freq

palavrastreina = buscapalavras(frasescomstemmingtreinamento)
frequenciatreina = buscarfrequencia(palavrastreina)
palavrasunicastreina = procurapalavrasunicas(frequenciatreina)

def analisarpalavras(documento):
    doc = set(documento)
    caracteristicas = {}
    for palavras in palavrasunicastreina:
        caracteristicas['%s' % palavras] = (palavras in doc)
    return caracteristicas

basecheiatreinamento = nltk.classify.apply_features(analisarpalavras, frasescomstemmingtreinamento)
basecheiateste = nltk.classify.apply_features(analisarpalavras, frasescomstemmingteste)

classificador = nltk.NaiveBayesClassifier.train(basecheiatreinamento)

esperado = []
previsto = []

for (frase, classe) in basecheiateste:
    resultado = classificador.classify(frase)
    previsto.append(resultado)
    esperado.append(classe)

matriz = ConfusionMatrix(esperado, previsto)
print("Matriz de Confusão (Treinamento do Capítulo):")
print(matriz)