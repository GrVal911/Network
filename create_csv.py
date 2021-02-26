import os
from bs4 import BeautifulSoup
import pandas as pd
# import data_preparation
from data_preparation import *
from itertools import chain


"""
Классы которыми размечены сущности в html файле

WebAnnotator_ADDRESS
WebAnnotator_INN
WebAnnotator_KPP
WebAnnotator_OGRN
WebAnnotator_REG_CBR
WebAnnotator_PHONE
WebAnnotator_EMAIL
WebAnnotator_LEGAL_NAME
WebAnnotator_NAME
WebAnnotator_REGIONS
WebAnnotator_RATE
WebAnnotator_CURRENCY
"""

my_dict = {}

my_list_l = []

my_list = [
    ".WebAnnotator_ADDRESS",
    ".WebAnnotator_INN",
    ".WebAnnotator_KPP",
    ".WebAnnotator_OGRN",
    ".WebAnnotator_REG_CBR",
    ".WebAnnotator_PHONE",
    ".WebAnnotator_EMAIL",
    ".WebAnnotator_LEGAL_NAME",
    ".WebAnnotator_NAME",
    ".WebAnnotator_REGIONS",
    ".WebAnnotator_RATE",
    ".WebAnnotator_CURRENCY",
]

my_list_dict = {
    0: {'class_name': ".WebAnnotator_ADDRESS", 'b_tw':'B_ADDRESS', 'i_tw':'I_ADDRESS'},
    1: {'class_name': ".WebAnnotator_INN", 'b_tw':'B_INN', 'i_tw':'I_INN'},
    2: {'class_name': ".WebAnnotator_KPP", 'b_tw':'B_KPP', 'i_tw':'I_KPP'},
    3: {'class_name': ".WebAnnotator_OGRN", 'b_tw':'B_OGRN', 'i_tw':'I_OGRN'},
    4: {'class_name': ".WebAnnotator_REG_CBR", 'b_tw':'B_REG', 'i_tw':'I_REG'},
    5: {'class_name': ".WebAnnotator_PHONE", 'b_tw':'B_PHONE', 'i_tw':'I_PHONE'},
    6: {'class_name': ".WebAnnotator_EMAIL", 'b_tw':'B_EMAIL', 'i_tw':'I_EMAIL'},
    7: {'class_name': ".WebAnnotator_LEGAL_NAME", 'b_tw':'B_LEGAL', 'i_tw':'I_LEGAL'},
    8: {'class_name': ".WebAnnotator_NAME", 'b_tw':'B_NAME', 'i_tw':'I_NAME'},
    9: {'class_name': ".WebAnnotator_REGIONS", 'b_tw':'B_REGIONS', 'i_tw':'I_REGIONS'},
    10: {'class_name': ".WebAnnotator_RATE", 'b_tw':'B_RATE', 'i_tw':'I_RATE'},
    11: {'class_name': ".WebAnnotator_CURRENCY", 'b_tw':'B_CURRENCY', 'i_tw':'I_CURRENCY'}
}

# Код Максима -------------------------------------------------------------------------

directory_in_str = os.path.dirname(os.path.abspath(__file__)) + '\html_razmetka'
dict_indexes = {}



def type_definition(tokenize_text, tokenize_aa, type_word):
    # print('='*100)

    dict_index = {}

    for a in range(0, len(tokenize_aa)):
        for aa in range(0, len(tokenize_text)):
            if tokenize_aa[a] in tokenize_text[aa]:

                if tokenize_aa[0] == tokenize_aa[a]:
                    dict_index[aa] = {'w': tokenize_aa[a], 't': type_word['b_tw'], 'class_name': type_word['class_name']}
                else:
                    dict_index[aa] = {'w': tokenize_aa[a], 't': type_word['i_tw'], 'class_name': type_word['class_name']}

    return dict_index



def unique_tags(p_file_path, p_my_list):
    for p in range(0, len(p_file_path)):
        soup = BeautifulSoup(open(p_file_path[p], encoding='utf-8', errors='ignore'), "lxml")

        tokenize_text = tokenization(soup.html.text)
        tokenize_text = delete_stop_word(tokenize_text)
        tokenize_text = delete_symbols(tokenize_text)

        my_dict2 = {}
        for i in range(0, len(tokenize_text)):
            my_dict2[i] = {'w': tokenize_text[i], 't': 'O'}



        for i in range(0, len(p_my_list.items())):
            a = soup.select(p_my_list[i]['class_name'])
            for aa in a:
                if bool(aa):
                    tokenize_aa = tokenization(aa.text)
                    tokenize_aa = delete_stop_word(tokenize_aa)
                    tokenize_aa = delete_symbols(tokenize_aa)
                    if bool(tokenize_aa):
                        t = type_definition(tokenize_text, tokenize_aa, p_my_list[i])
                        for key, val in t.items():
 
                            my_dict2[key]['t'] = val['t']



        my_dict[p] = my_dict2
    return (my_dict)


path_to_file = []


for file in os.listdir(directory_in_str):
    if file.endswith(".html"):
        path_to_file.append(os.path.join(directory_in_str, file))


# unique_tags(path_to_file, my_list_dict)


# Код Валентина -------------------------------------------------------------------------
def get_dict_map(data, token_or_tag):
    tok2idx = {}
    idx2tok = {}

    if token_or_tag == 'token':
        vocab = list(set(data['Word'].to_list()))
    else:
        vocab = list(set(data['Tag'].to_list()))

    idx2tok = {idx: tok for idx, tok in enumerate(vocab)}
    tok2idx = {tok: idx for idx, tok in enumerate(vocab)}
    return tok2idx, idx2tok

def table_creation():
    word_dict  = unique_tags(path_to_file, my_list_dict)

    data_table = pd.DataFrame(columns = ['Page', 'Word', 'Tag'])

    row = {}
    for j in range (0, len(list(word_dict.values()))):
        row.update({'Page' : j})
        for i in list(dict(list(word_dict.values())[j]).values()):
            row.update({'Word': list(i.values())[0]})
            row.update({'Tag': list(i.values())[1]})
            data_table.loc[-1] = list(row.values())
            data_table.index = data_table.index + 1

    # print(data_table)

    token2idx, idx2token = get_dict_map(data_table, 'token')
    tag2idx, idx2tag = get_dict_map(data_table, 'tag')

    data_table['Word_idx'] = data_table['Word'].map(token2idx)
    data_table['Tag_idx'] = data_table['Tag'].map(tag2idx)

    # print(data_table)

    data_group = data_table.groupby(
                    ['Page'],as_index=False
                    )['Word', 'Tag', 'Word_idx', 'Tag_idx'].agg(lambda x: list(x))
    # data_group = data_group.drop(data_group.columns[0], axis=1)

    return(data_group, data_table)
    # data_group.to_csv('data_group.csv')

# print(table_creation())