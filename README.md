# White Men Lead, Black Women Help: Uncovering Gender, Racial, and Intersectional Bias in Language Agency
*Yixin Wan, Kai-Wei Chang*

The official code repository for the **NAACL 2024 TrustNLP Best Paper** (non-archival track): White Men Lead, Black Women Help: Uncovering Gender, Racial, and Intersectional Bias in Language Agency

### Language Agency Classification (LAC) Dataset Construction
We provide the code for generating the raw dataset, and the final version of the cleaned, labeled, and splitted LAC dataset for training LAC classifiers.
* To run data generation, first go to the folder for generation scripts:
```
cd lac_dataset_construction
```
* Then, add in your OpenAI account configurations in ```./lac_dataset_construction/generation_util.py``` and run:
```
python generate_dataset.py
```

### Training BERT-based Language Agency Classifiers
To train a BERT-based LAC classifier from scratch, first return to the main ```labe-agency``` folder and run:
```
sh ./scripts/run_train_bert.sh
```

Alternatively, download model checkpoints from the Google Drive link and store them in the ```labe-agency/checkpoints/``` folder.

### Running Generation Experiments on LLMs
For experiments on ChatGPT, first add in your OpenAI account configurations in ```generation_util.py```.
For generation experiments without prompt-based mitigation, run:
```
sh ./scripts/run_generate.sh
```

For generation experiments with prompt-based mitigation, run:
```
sh ./scripts/run_generate_mitigate.sh
```

### Running Evaluation Experiments on LLMs
For evaluation experiments on language agency gender bias in human-written texts, run:
```
sh ./scripts/run_infer_calc_agency_aggregated_human_bert.sh
```

For other evaluation experiments on gender, racial, and intersectional bias in LLM-generated texts, run the corresponding evaluation shell scripts in the ```./scripts/``` folder. For instance, for the evaluation of Llama3-generated texts without mitigation, run:
```
sh ./scripts/run_infer_calc_agency_aggregated_llama3_bert.sh
```
