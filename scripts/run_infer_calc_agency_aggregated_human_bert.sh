# bios
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_0/checkpoint-4500 -d bias_bios --seed 0
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_1/checkpoint-4500 -d bias_bios --seed 1
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_2/checkpoint-4500 -d bias_bios --seed 2
python calc_agency_gender.py --input_file ./outputs/human_bias_bios/bert-base-cased_results_0.csv
# professor reviews
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_0/checkpoint-4500 -d ratemyprofessor --seed 0
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_1/checkpoint-4500 -d ratemyprofessor --seed 1
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_2/checkpoint-4500 -d ratemyprofessor --seed 2
python calc_agency_gender.py --input_file ./outputs/human_ratemyprofessor/bert-base-cased_results_0.csv
# rec letters
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_0/checkpoint-4500 -d rec_letter --seed 0
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_1/checkpoint-4500 -d rec_letter --seed 1
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_2/checkpoint-4500 -d rec_letter --seed 2
python calc_agency_gender.py --input_file ./outputs/human_rec_letter/bert-base-cased_results_0.csv