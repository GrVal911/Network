# coding: utf-8


# import pandas as pd
# import numpy as np
# import csv
import nltk
import pymorphy2

# import gensim
import string
# from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import re



# Интерактивный установщик
# nltk.download()




# text = open('./anna.txt', 'r', encoding='utf-8').read()

# print(f'Изначальный текст: \n{text}')
# print('--------------------------------------------------------')

def tokenization(text):
    tokens = word_tokenize(text) # Токенизация слов (разбиение текста на отдельные слова)
    tokens = [i.lower() for i in tokens] # Перевод всех слов в малый регистр
    tokens = [i for i in tokens if (i not in string.punctuation)] # Удаление пунктуации из текста
    return tokens


def delete_stop_word(tokens):

    stop_words = stopwords.words('russian')

    # Расширение списка стоп слов
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', '–', 'к', 'на', '...', "«", "»", '…'])
    tokens = [i for i in tokens if (i not in stop_words)]
    tokens = [i.replace('…', "").replace('…', "") for i in tokens]
    return tokens



def delete_symbols(tokens):
    # Создание паттернов для отсеивания смайликов и символов
    regrex_pattern = re.compile(pattern = "["
            u"\U0001F600-\U0001F64F"  # emoticons (смайлики)
            u"\U0001F300-\U0001F5FF"  # symbols & pictographs (символы и пиктограммы)
            u"\U0001F680-\U0001F6FF"  # transport & map symbols (символы транспорта и карты)
            u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                            "]+", flags = re.UNICODE)

    # Фильтрация текста от смайликов
    tokens = [regrex_pattern.sub(r'', i) for i in tokens]

    tokens
    return tokens



def convert_tokens_to_normal_form(tokens):
    # Преобразование слов в нормальную форму
    morph = pymorphy2.MorphAnalyzer(lang='ru')
    words_norlmal_form = []

    for i in tokens:
        p = morph.parse(i)[0]
        words_norlmal_form.append(p.normal_form)

    print(f'Обработанный текст: \n{words_norlmal_form}')
