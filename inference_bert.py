import pandas as pd
import numpy as np
import evaluate
import torch
import re
import os
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

# idx_to_label = {
#     2: 'agentic',
#     1: 'neutral',
#     0: 'communal'
# }
idx_to_label = {
    1: 'agentic',
    0: 'communal'
}

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str):
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
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

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', default="bert-base-cased", help='Name of model.')
parser.add_argument('-gm', '--generation_model', default="chatgpt")
parser.add_argument('-c', '--checkpoint_dir', default=None, help='Path to checkpoint.')
parser.add_argument('-d', '--dataset_name', default="bias_bios", help='Dataset name.')
parser.add_argument('-s', '--seed', default=0, help='Random Seed.')
parser.add_argument('--mitigate', action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)

# human-written texts
if args.dataset_name == 'bias_bios':
    df = pd.read_csv('./data/bias_bios.csv')
    dataset = Dataset.from_pandas(df)
    column = 'hard_text'
    args.generation_model = 'human'
elif args.dataset_name == 'ratemyprofessor':
    df = pd.read_csv('./data/ratemyprofessor.csv')
    dataset = Dataset.from_pandas(df)
    column = 'comments'
    args.generation_model = 'human'
elif args.dataset_name == 'rec_letter':
    df = pd.read_csv('./outputs/rec_letter.csv')
    dataset = Dataset.from_pandas(df)
    column = 'chatgpt_gen'
    args.generation_model = 'human'
# LLM-generated texts
elif not args.mitigate:
    if args.dataset_name == 'llm_rec_letter':
        df = pd.read_csv('./data/{}_clg_llm_letters.csv'.format(args.generation_model))
        df = df[['{}_gen'.format(args.generation_model), 'name', 'gender', 'race', 'occupation']]
    elif args.dataset_name == 'llm_bios':
        df = pd.read_csv('./data/{}_clg_llm_bios.csv'.format(args.generation_model))
        df = df[['{}_gen'.format(args.generation_model), 'name', 'gender', 'race', 'occupation']]
    elif args.dataset_name == 'llm_professor':
        df = pd.read_csv('./data/{}_clg_llm_professor.csv'.format(args.generation_model))
        df = df[['{}_gen'.format(args.generation_model), 'name', 'gender', 'race', 'department']]
    df['sentences'] = None
    for i in range(len(df)):
        df['sentences'][i] = split_into_sentences(df['{}_gen'.format(args.generation_model)][i])
    dataset = Dataset.from_pandas(df)
    column = '{}_gen'.format(args.generation_model)
elif args.mitigate:
    if args.dataset_name == 'llm_rec_letter':
        df = pd.read_csv('./data/{}_clg_llm_letters_mitigate.csv'.format(args.generation_model))
        df = df[['{}_gen'.format(args.generation_model), 'name', 'gender', 'race', 'occupation']]
    elif args.dataset_name == 'llm_bios':
        df = pd.read_csv('./data/{}_clg_llm_bios_mitigate.csv'.format(args.generation_model))
        df = df[['{}_gen'.format(args.generation_model), 'name', 'gender', 'race', 'occupation']]
    elif args.dataset_name == 'llm_professor':
        df = pd.read_csv('./data/{}_clg_llm_professor_mitigate.csv'.format(args.generation_model))
        df = df[['{}_gen'.format(args.generation_model), 'name', 'gender', 'race', 'department']]

    df['sentences'] = None
    for i in range(len(df)):
        df['sentences'][i] = split_into_sentences(df['{}_gen'.format(args.generation_model)][i])
    dataset = Dataset.from_pandas(df)
    column = '{}_gen'.format(args.generation_model)

eval_dataloader = DataLoader(dataset, batch_size=1)
tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
model = AutoModelForSequenceClassification.from_pretrained(args.checkpoint_dir)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

model.eval()
df['labels'] = None
df['percentage'] = None
df['average'] = None

for idx, batch in enumerate(tqdm(eval_dataloader)):
    sentences = split_into_sentences(batch[column][0])
    results = []
    with torch.no_grad():
        for s in sentences:
            inputs = tokenizer(s, padding="max_length", truncation=True, return_tensors='pt').to("cuda")
            outputs = model(**inputs)

            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1)
            results.append(prediction.detach().cpu().item())

        assert len(results) == len(sentences)
        ct = 0
        score = 0
        labels = []

        for j in range(len(results)):
            labels.append(idx_to_label[results[j]])
            if results[j] == 1:
                ct += 1
                score += 1
            elif results[j] == 0:
                score -= 1
                
        df['labels'][idx] = labels
        df['percentage'][idx] = ct / len(results)
        df['average'][idx] = score / len(results)

if not args.mitigate:
    output_folder_name = './outputs/' + args.generation_model + '_' + args.dataset_name
else:
    output_folder_name = './outputs/' + args.generation_model + '_' + args.dataset_name + '_mitigate'
if not os.path.exists(output_folder_name):
    os.mkdir(output_folder_name)

if not args.mitigate:
    df.to_csv('./outputs/{}_{}/{}_results_{}.csv'.format(args.generation_model, args.dataset_name, args.model_name, str(args.seed)), index=False)
else:
    df.to_csv('./outputs/{}_{}_mitigate/{}_results_{}.csv'.format(args.generation_model, args.dataset_name, args.model_name, str(args.seed)), index=False)