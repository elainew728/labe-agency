import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
import random
import statistics 

parser = argparse.ArgumentParser()
parser.add_argument('--input_file', help='Path for the generation csv file.')
args = parser.parse_args()

df1 = pd.read_csv(args.input_file) 
df2 = pd.read_csv(args.input_file.replace('_0.csv', '_1.csv'))
df3 = pd.read_csv(args.input_file.replace('_0.csv', '_2.csv'))

# Gender Bias
a_p_m, a_p_f, c_p_m, c_p_f = [], [], [], []
a_g_m, a_g_f = [], []
std = []
for df in [df1, df2, df3]:
    if 'bias_bios' in args.input_file:
        dataset_name = 'bias_bios'
        df_m = df[df['gender'] == 'male']
        df_f = df[df['gender'] == 'female']
    elif 'resume' in args.input_file:
        dataset_name = 'resume'
        df_m = df[df['gender'] == 'Male']
        df_f = df[df['gender'] == 'Female']
    elif 'ratemyprofessor' in args.input_file:
        dataset_name = 'ratemyprofessor'
        departments = ['Communication department', 'Fine Arts department', 'Chemistry department', 'Mathematics department', 
                'Biology department', 'English department', 'Computer Science department', 'Sociology department', 'Economics department', 'Humanities department', 'Science department', 'Languages department', 'Education department', 'Accounting department', 'Philosophy department']
        excl_departments = []
        for d in list(df['department_name'].unique()):
            if d not in departments:
                excl_departments.append(d)
        for n in excl_departments:
            df = df[df['department_name'] != n]
        df.reset_index(drop=True, inplace=True)

        df_m = df[df['gender'] != 'female']
        df_m = df_m[df_m['gender'] != 'mostly_female']
        df_f = df[df['gender'] != 'male']
        df_f = df_f[df_f['gender'] != 'mostly_male']

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
