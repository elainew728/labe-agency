# White Men Lead, Black Women Help: Uncovering Gender, Racial, and Intersectional Bias in Language Agency
*Yixin Wan, Kai-Wei Chang*

The official code repository for the paper: White Men Lead, Black Women Help: Uncovering Gender, Racial, and Intersectional Bias in Language Agency

### Language Agency Classification (LAC) Dataset Construction
We provide the code for generating the raw dataset, and the final version of the cleaned, labeled, and splitted LAC dataset for training LAC classifiers.
* To run data generation, first go to the folder for generation scripts:
```
cd lac_dataset_construction
```
* Then, add in your OpenAI account configurations in ```generation_util.py``` and run:
```
python generate_dataset.py
```
