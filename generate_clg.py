import random
from generation_util import *
from tqdm import tqdm
import pandas as pd
from argparse import ArgumentParser
from transformers import pipeline
from constants import male_names, female_names, occupations, departments, races, ages, rec_instruction, bio_instruction, prof_instruction

if __name__ == "__main__":
    # Configuration
    parser = ArgumentParser()
    parser.add_argument('-g', '--generation_type', default='rec')
    parser.add_argument('-b', '--bias', default='race')
    parser.add_argument('-m', '--model_type', default='chatgpt')
    parser.add_argument('-of', '--output_folder', default='./data')
    parser.add_argument('-mt', '--mitigate', action="store_true", help='Whether to do mitigation.')
    args = parser.parse_args()

    instructions = []
    if args.generation_type in ['rec', 'bio']:
        idxs = random.sample([i for i in range(5 * len(ages) * len(occupations))], 12)
    else:
        idxs = random.sample([i for i in range(5 * len(departments))], 12)
    for race in races:
        id = 0
        for name in female_names[race]:
            if args.generation_type in ['rec', 'bio']:
                for age in ages:
                    for occupation in occupations:
                        if args.generation_type == 'rec':
                            instruction = rec_instruction.format(name, age, race, "female", occupation).strip()
                            if args.mitigate:
                                instruction = instruction + ' When generating the reference letter, ensure that you display no biases in language agency across gender or race.'
                        elif args.generation_type == 'bio':
                            instruction = bio_instruction.format(name, age, race, "female", occupation).strip()
                            if args.mitigate:
                                instruction = instruction + ' When generating the biography, ensure that you display no biases in language agency across gender or race.'
                        
                        if (not args.mitigate) or (args.mitigate and id in idxs):
                            instructions.append(
                                (name, age, race, 'female', occupation, instruction)
                            )
                        id += 1
            elif args.generation_type == 'prof':
                for dpt in departments:
                    instruction = prof_instruction.format(name, race, "female", dpt).strip()
                    if args.mitigate:
                        instruction = instruction + ' When generating the professor review, ensure that you display no biases in language agency across gender or race.'
                    if (not args.mitigate) or (args.mitigate and id in idxs):
                        instructions.append(
                            (name, race, 'female', dpt, instruction)
                        )
                    id += 1

    for race in races:
        id = 0
        for name in male_names[race]:
            if args.generation_type in ['rec', 'bio']:
                for age in ages:
                    for occupation in occupations:
                        if args.generation_type == 'rec':
                            instruction = rec_instruction.format(name, age, race, "male", occupation).strip()
                            if args.mitigate:
                                instruction = instruction + ' When generating the reference letter, ensure that you display no biases in language agency across gender or race.'
                        elif args.generation_type == 'bio':
                            instruction = bio_instruction.format(name, age, race, "male", occupation).strip()
                            if args.mitigate:
                                instruction = instruction + ' When generating the biography, ensure that you display no biases in language agency across gender or race.'
                        
                        if (not args.mitigate) or (args.mitigate and id in idxs):
                            instructions.append(
                                (name, age, race, 'male', occupation, instruction)
                            )
                        id += 1
            elif args.generation_type == 'prof':
                for dpt in departments:
                    instruction = prof_instruction.format(name, race, "male", dpt).strip()
                    if args.mitigate:
                        instruction = instruction + ' When generating the professor review, ensure that you display no biases in language agency across gender or race.'
                    if (not args.mitigate) or (args.mitigate and id in idxs):
                        instructions.append(
                            (name, race, 'male', dpt, instruction)
                        )
                    id += 1

    random.shuffle(instructions)
    instructions = instructions
    print('Number of documents to be generated:', len(instructions))

    if args.generation_type in ['rec', 'bio']:
        output = {
                'name': [],
                'age': [],
                'race': [],
                'gender': [],
                'occupation': [],
                'prompts': [],
                '{}_gen'.format(args.model_type): []
                }
    elif args.generation_type == 'prof':
        output = {
                'name': [],
                'race': [],
                'gender': [],
                'department': [],
                'prompts': [],
                '{}_gen'.format(args.model_type): []
                }

    if args.model_type == 'mistral':
        pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device="cuda:0")
    elif args.model_type == 'llama3':
        pipe = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda:0")
    else:
        pipe=None

    if args.generation_type in ['rec', 'bio']:
        for name, age, race, gender, occupation, instruction in tqdm(instructions):
            generated_response = generate_response_fn(instruction, args.model_type, pipe)            

            generated_response = generated_response.replace("\n", "<return>")
            output['{}_gen'.format(args.model_type)].append(generated_response)
            output['prompts'].append(instruction)
            output['name'].append(name)
            output['race'].append(race)
            output['gender'].append(gender)
            output['occupation'].append(occupation)
            output['age'].append(age)
            
    elif args.generation_type == 'prof':
        for name, race, gender, department, instruction in tqdm(instructions):
            generated_response = generate_response_fn(instruction, args.model_type, pipe) 
            generated_response = generated_response.replace("\n", "<return>")
            output['{}_gen'.format(args.model_type)].append(generated_response)
            output['prompts'].append(instruction)
            output['name'].append(name)
            output['race'].append(race)
            output['gender'].append(gender)
            output['department'].append(department)

    df = pd.DataFrame.from_dict(output)
    if args.mitigate:
        postfix = '_mitigate.csv'
    else:
        postfix = '.csv'
    if args.generation_type == 'rec':
        df.to_csv('{}/{}_clg_letters'.format(args.output_folder, args.model_type) + postfix, index=False)
    elif args.generation_type == 'bio':
        df.to_csv('{}/{}_clg_bios'.format(args.output_folder, args.model_type) + postfix, index=False)
    elif args.generation_type == 'prof':
        df.to_csv('{}/{}_clg_professor'.format(args.output_folder, args.model_type) + postfix, index=False)

