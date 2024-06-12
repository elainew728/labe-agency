# -*- coding: utf-8 -*-
import os
import re
import json
from json import JSONDecodeError
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
from generation_util import generate_response_chatgpt

AGENCY_COMMUNAL_PROMPT = "You will rephrase a sentence two times to demonstrate agentic and communal language traits respectively. 'agentic' is defined as more achievement-oriented, and 'communal' is defined as more social or service-oriented. Example of agentic description: {}. Example of communal description: {}. Output your answer in a json format with two keys, 'agentic' and 'communal'. The sentence is: '{}'"

# Examples of communal goals were “helping others” and “serving community” and examples of agentic goals were “status” and “demonstrating skill”. 
AGENCY_EXAMPLES = '[Name] is an achievement-oriented individual with 7 years of experience being in charge of people and projects in previous workplace environments.'

COMMUNAL_EXAMPLES = '[Name] is a people- oriented individual with 7 years of experience being a part of various financial teams and projects in previous workplace environments.'

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences

dataset = load_dataset("potsawee/wiki_bio_gpt3_hallucination", split='evaluation')

if os.path.isfile('./raw_data/agency_communal_wiki_bios.csv'):
    df = pd.read_csv('./raw_data/agency_communal_wiki_bios.csv')
    original_sentences = df['original'].tolist()
    agentic_sentences = df['agentic'].tolist()
    communal_sentences = df['communal'].tolist()
    dic = {'original': original_sentences, 'agentic': agentic_sentences, 'communal': communal_sentences}
    prev = len(original_sentences)
else:
    original_sentences = []
    agentic_sentences = []
    communal_sentences = []
    dic = {'original': original_sentences, 'agentic': agentic_sentences, 'communal': communal_sentences}
    df = pd.DataFrame.from_dict(dic)
    df.to_csv('./raw_data/agency_communal_wiki_bios.csv')
    prev = 0

ct = len(original_sentences)

for i in tqdm(range(len(dataset))):
    full = dataset[i]['wiki_bio_text']
    sentences = split_into_sentences(full)
    sentences = sentences[2:len(sentences) - 1]
    print('Count:', ct)
    for sent in sentences:
        if len(sent) <= 15 or len(sent) >= 300:
            continue

        if prev != 0:
            prev -= 1
            continue

        print('Sent:', sent)
        utt = AGENCY_COMMUNAL_PROMPT.format(AGENCY_EXAMPLES, COMMUNAL_EXAMPLES, sent)
        response = generate_response_chatgpt(utt)
        try:
            response = json.loads(response)
        except JSONDecodeError or KeyError:
            print('Error:', response)
            continue

        original_sentences.append(sent)
        try:
            assert ('agentic' in response.keys()) and ('communal' in response.keys())
        except AssertionError:
            print('Error:', response)
            continue
        agentic_sentences.append(response['agentic'])
        communal_sentences.append(response['communal'])
        
        dic['original'] = original_sentences
        dic['agentic'] = agentic_sentences
        dic['communal'] = communal_sentences
        df = pd.DataFrame.from_dict(dic)
        df.to_csv('./raw_data/agency_communal_wiki_bios.csv')

        ct += 1


