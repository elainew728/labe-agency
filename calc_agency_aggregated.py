import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import random
import statistics 
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', help='Path for the generation csv file.')
parser.add_argument('-gm', '--generation_model', default="mistral")
parser.add_argument('--plot', action="store_true", help='Whether to plot.')
parser.add_argument('--mitigate', action="store_true")
args = parser.parse_args()

df1 = pd.read_csv(args.input_file) 
df2 = pd.read_csv(args.input_file.replace('_0.csv', '_1.csv'))
df3 = pd.read_csv(args.input_file.replace('_0.csv', '_2.csv'))

# Gender Bias
a_p_m, a_p_f, c_p_m, c_p_f = [], [], [], []
a_g_m, a_g_f = [], []
std = []
for df in [df1, df2, df3]:
    if 'llm_bios' in args.input_file:
        dataset_name = 'llm_bios_gender'
        df_m = df[df['gender'] == 'male']
        df_f = df[df['gender'] == 'female']
    elif 'llm_professor' in args.input_file:
        dataset_name = 'llm_professor_gender'
        df_m = df[df['gender'] == 'male']
        df_f = df[df['gender'] == 'female']
    elif 'llm_rec_letter' in args.input_file:
        dataset_name = 'llm_rec_letter_gender'
        df_m = df[df['gender'] == 'male']
        df_f = df[df['gender'] == 'female']

    print('\n\nEvaluating Gender Bias of Dataset: ', dataset_name)
    percentage_m = df_m['percentage'].tolist()
    percentage_communal_m = [1 - p for p in percentage_m]
    percentage_f = df_f['percentage'].tolist()
    percentage_communal_f = [1 - p for p in percentage_f]
    a_p_m.append(sum(percentage_m) / len(percentage_m) * 100)
    a_p_f.append(sum(percentage_f) / len(percentage_f) * 100)
    c_p_m.append(sum(percentage_communal_m) / len(percentage_communal_m) * 100)
    c_p_f.append(sum(percentage_communal_f) / len(percentage_communal_f) * 100)

    average_m = df_m['average'].tolist()
    average_f = df_f['average'].tolist()
    a_g_m.append(sum(average_m) / len(average_m) * 100)
    a_g_f.append(sum(average_f) / len(average_f) * 100)
    std.append(statistics.variance([sum(average_m) / len(average_m) * 100, sum(average_f) / len(average_f) * 100]))

print('male agentic percentage: ', sum(a_p_m) / len(a_p_m), 'female agentic percentage: ', sum(a_p_f) / len(a_p_f))
print('male communal percentage: ', sum(c_p_m) / len(c_p_m), 'female communal percentage: ', sum(c_p_f) / len(c_p_f))
print('male average: ', sum(a_g_m) / len(a_g_m), 'female average: ', sum(a_g_f) / len(a_g_f))
print('\n\nGender Var.: ', "{}".format(np.round(sum(std) / len(std), 2)))

# Racial Bias 
a_p_w, a_p_b, a_p_h, a_p_a = [], [], [], []
c_p_w, c_p_b, c_p_h, c_p_a = [], [], [], []
a_g_w, a_g_b, a_g_h, a_g_a = [], [], [], []
std = []
print('\n\nEvaluating Racial Bias of Dataset: ', dataset_name)
for df in [df1, df2, df3]:
    if 'race' in args.input_file:
        if 'llm_rec_letter' in args.input_file:
            dataset_name = 'llm_rec_letter'
        elif 'llm_professor' in args.input_file:
            dataset_name = 'llm_professor'
        elif 'llm_bios' in args.input_file:
            dataset_name = 'llm_bios'
        df_white = df[df['race'] == 'White']
        df_black = df[df['race'] == 'Black']
        df_hispanic = df[df['race'] == 'Hispanic']
        df_asian = df[df['race'] == 'Asian']
    
    percentage_white = df_white['percentage'].tolist()
    percentage_communal_white = [1 - p for p in percentage_white]
    percentage_black = df_black['percentage'].tolist()
    percentage_communal_black = [1 - p for p in percentage_black]
    percentage_hispanic = df_hispanic['percentage'].tolist()
    percentage_communal_hispanic = [1 - p for p in percentage_hispanic]
    percentage_asian = df_asian['percentage'].tolist()
    percentage_communal_asian = [1 - p for p in percentage_asian]
    a_p_w.append(sum(percentage_white) / len(percentage_white) * 100)
    a_p_b.append(sum(percentage_black) / len(percentage_black) * 100)
    a_p_h.append(sum(percentage_hispanic) / len(percentage_hispanic) * 100)
    a_p_a.append(sum(percentage_asian) / len(percentage_asian) * 100)
    c_p_w.append(sum(percentage_communal_white) / len(percentage_communal_white) * 100)
    c_p_b.append(sum(percentage_communal_black) / len(percentage_communal_black) * 100)
    c_p_h.append(sum(percentage_communal_hispanic) / len(percentage_communal_hispanic) * 100)
    c_p_a.append(sum(percentage_communal_asian) / len(percentage_communal_asian) * 100)

    average_white = df_white['average'].tolist()
    average_black = df_black['average'].tolist()
    average_hispanic = df_hispanic['average'].tolist()
    average_asian = df_asian['average'].tolist()
    a_g_w.append(sum(average_white) / len(average_white) * 100)
    a_g_b.append(sum(average_black) / len(average_black) * 100)
    a_g_h.append(sum(average_hispanic) / len(average_hispanic) * 100)
    a_g_a.append(sum(average_asian) / len(average_asian) * 100)
    std.append(statistics.variance([sum(average_white) / len(average_white) * 100, sum(average_black) / len(average_black) * 100, sum(average_hispanic) / len(average_hispanic) * 100, sum(average_asian) / len(average_asian) * 100]))

print('White agentic percentage: ', sum(a_p_w) / len(a_p_w))
print('Black agentic percentage: ', sum(a_p_b) / len(a_p_b))
print('Hispanic agentic percentage: ', sum(a_p_h) / len(a_p_h))
print('Asian agentic percentage: ', sum(a_p_a) / len(a_p_a))
print('White average: ', sum(a_g_w) / len(a_g_w))
print('Black average: ', sum(a_g_b) / len(a_g_b))
print('Hispanic average: ', sum(a_g_h) / len(a_g_h))
print('Asian average: ', sum(a_g_a) / len(a_g_a))
print('\n\nRacial Var.: ', "{}".format(np.round(sum(std) / len(std), 2)))

# intersectional
print('\n\nEvaluating Intersectional Bias of Dataset: ', dataset_name)
race_f, race_m = [], []
race_m_dic, race_f_dic = {0: [], 1: [], 2: [],}, {0: [], 1: [], 2: [],}
for r in ['White', 'Black', 'Hispanic', 'Asian']:
    tmp_race_f, tmp_race_m = [], []
    p_m, p_f, p_c_m, p_c_f, a_m, a_f = [], [], [], [], [], []

    for i, df in enumerate([df1, df2, df3]):
        if 'llm' in args.input_file:
            if 'llm_' in args.input_file:
                dataset_name = 'llm_rec_letter'
            elif 'llm_rofessor' in args.input_file:
                dataset_name = 'llm_professor'
            elif 'llm_bios' in args.input_file:
                dataset_name = 'llm_bios'
            tmp_df = df[df['race'] == r]

        tmp_df_m = tmp_df[tmp_df['gender'] == 'male']
        tmp_df_f = tmp_df[tmp_df['gender'] == 'female']

        tmp_average_m = tmp_df_m['average'].tolist()
        tmp_average_f = tmp_df_f['average'].tolist()

        a_m.append((sum(tmp_average_m) / len(tmp_average_m)) * 100)
        a_f.append((sum(tmp_average_f) / len(tmp_average_f)) * 100)
        tmp_race_m.append((sum(tmp_average_m) / len(tmp_average_m)) * 100)
        tmp_race_f.append((sum(tmp_average_f) / len(tmp_average_f)) * 100)
        race_m_dic[i].append((sum(tmp_average_m) / len(tmp_average_m)) * 100)
        race_f_dic[i].append((sum(tmp_average_f) / len(tmp_average_f)) * 100)

        tmp_percentage_m = tmp_df_m['percentage'].tolist()
        tmp_percentage_f = tmp_df_f['percentage'].tolist()
        tmp_percentage_communal_m = [1 - p for p in tmp_percentage_m]
        tmp_percentage_communal_f = [1 - p for p in tmp_percentage_f]

        p_m.append((sum(tmp_percentage_m) / len(tmp_percentage_m)) * 100) 
        p_f.append((sum(tmp_percentage_f) / len(tmp_percentage_f)) * 100)
        p_c_m.append((sum(tmp_percentage_communal_m) / len(tmp_percentage_communal_m)) * 100)
        p_c_f.append((sum(tmp_percentage_communal_f) / len(tmp_percentage_communal_f)) * 100) 
    
    race_m.append(sum(tmp_race_m) / len(tmp_race_m)) # avg agentic gap of males in the race
    race_f.append(sum(tmp_race_f) / len(tmp_race_f))
    assert len(a_m) == 3
    print('Race:', r, 'Avg Male: ', sum(a_m) / len(a_m), 'Avg Female: ', sum(a_f) / len(a_f), 
          'Per Agentic Male: ', sum(p_m) / len(p_m), 'Per Agentic Female: ', sum(p_f) / len(p_f), 
          'Per Communal Male: ', sum(p_c_m) / len(p_c_m), 'Per Communal Female: ', sum(p_c_f) / len(p_c_f))

std = []
for i in range(3):
    std.append(statistics.variance(race_f_dic[i] + race_m_dic[i]))
print('\n\nIntersectional Var.: ', "{}".format(np.round(sum(std) / len(std), 2)))

races = ['White', 'Black', 'Hispanic', 'Asian']

if args.plot:
    # Choosing colors (shades of blue and purple)
    colors = ['#a5eef2' for _ in range(4)] 
    colors2 = ['#89afe8' for _ in range(4)] 

    plt.figure(figsize=(10, 6))
    plt.ylim(-2, 2)
    plt.xlim(-40,40)

    # Positions for the bars
    y_positions = np.arange(len(races))
    # Bar width
    bar_width = 0.5

    if (race_m[0] < 0) and (race_f[0] < 0):
        plt.barh(y_positions, race_f, bar_width, edgecolor ='grey', label='Female', color=colors2, alpha=0.7)
        plt.barh(y_positions, race_m, bar_width, edgecolor ='grey', label='Male', color=colors, alpha=0.7)
    else:
        plt.barh(y_positions, race_m, bar_width, edgecolor ='grey', label='Male', color=colors, alpha=0.7)
        plt.barh(y_positions, race_f, bar_width, edgecolor ='grey', label='Female', color=colors2, alpha=0.7)

    plt.axvline(x=0.00, color='#e2e2e2', linestyle='--')
    # Adding labels, title and legend
    plt.xlabel('Gap (%Agentic - %Communal)', fontsize=20)
    plt.ylabel('Races', fontsize=20)
    plt.yticks(y_positions, races, fontsize=16)
    if args.generation_model == 'mistral':
        title = 'Mistral '
    elif args.generation_model == 'llama3':
        title = 'Llama3 '
    elif args.generation_model == 'chatgpt':
        title = 'ChatGPT '
    if dataset_name == 'llm_bios':
        title += 'Biography'
    elif dataset_name == 'llm_professor':
        title += 'Prof Review'
    elif dataset_name == 'llm_rec_letter':
        title += 'Rec Letter'
    
    plt.title(title, fontsize=22, pad=10) # Rec_Letter
    plt.legend(fontsize=16)
    # Show the plot
    if not args.mitigate:
        output_subfolder = args.generation_model
    else:
        output_subfolder = args.generation_model + '_mitigate'
    if not os.path.exists('./llm_outputs/{}'.format(output_subfolder)):
        os.mkdir('./llm_outputs/{}'.format(output_subfolder))
    plt.savefig('./llm_outputs/{}/{}_intersectional_bert_visualization.pdf'.format(output_subfolder, dataset_name), dpi=300,bbox_inches='tight')
    plt.close()