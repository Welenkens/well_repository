import numpy
import re
from nltk import ngrams
from nltk.corpus import stopwords


def tokenize(sentences):
    words = []
    for sentence in sentences:
        w = word_extraction(sentence)
        words.extend(w)
    return words


def word_extraction(sentence):
    words = re.sub("[^\w]", " ", sentence).split()
    cleaned_text = [w.lower() for w in words if w not in stopwords.words()]
    return cleaned_text


def generate_bow(text):
    vocab = tokenize(text)
    vocab = list(dict.fromkeys(vocab))
    print("\n".join(vocab))
    print("\n")

    for sentence in text:
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1

        print("{0} \n{1}\n".format(sentence, numpy.array(bag_vector)))


def generate_bow_2gram(text):
    vocab = tokenize(text)
    vocab = list(dict.fromkeys(vocab))
    vocab_str = ' '.join(vocab)
    bigrams = ngrams(vocab_str.split(), 2)

    for grams in bigrams:
        print(grams)
    print("\n")

    for sentence in text:
        words = word_extraction(sentence)
        bag_vector = numpy.zeros(len(vocab))
        for w in words:
            for i, word in enumerate(vocab):
                if word == w:
                    bag_vector[i] += 1

        print("{0} \n{1}\n".format(sentence, numpy.array(bag_vector)))

text = ["Falar é fácil. Mostre-me o código.", "É fácil escrever código. Difícil é escrever código que funcione."]

print("1 - Vocabulário Completo (1-gram) ")
print("    N-Vetores de Palavras")
print("2 - Vocabulário Completo (2-gram); ")
print("    N-Vetores de Palavras")
option = input("Informe 1 ou 2: ")

if option == '1':
    generate_bow(text)
elif option == '2':
    generate_bow_2gram(text)
else:
    print("Opção inválida!")
