# bios
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_0/checkpoint-4500 -d llm_bios --seed 0 -gm llama3 --mitigate
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_1/checkpoint-4500 -d llm_bios --seed 1 -gm llama3 --mitigate
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_2/checkpoint-4500 -d llm_bios --seed 2 -gm llama3 --mitigate
python calc_agency_aggregated.py --input_file ./outputs/llama3_llm_bios_mitigate/bert-base-cased_results_0.csv -gm llama3 --mitigate --plot
# professor reviews
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_0/checkpoint-4500 -d llm_professor --seed 0 -gm llama3 --mitigate
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_1/checkpoint-4500 -d llm_professor --seed 1 -gm llama3 --mitigate
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_2/checkpoint-4500 -d llm_professor --seed 2 -gm llama3 --mitigate
python calc_agency_aggregated.py --input_file ./outputs/llama3_llm_professor_mitigate/bert-base-cased_results_0.csv -gm llama3 --mitigate --plot 
# rec letters
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_0/checkpoint-4500 -d llm_rec_letter --seed 0 -gm llama3 --mitigate
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_1/checkpoint-4500 -d llm_rec_letter --seed 1 -gm llama3 --mitigate
CUDA_VISIBLE_DEVICES=0 python inference_bert.py \
    -c ./checkpoints/bert-base-cased_2/checkpoint-4500 -d llm_rec_letter --seed 2 -gm llama3 --mitigate
python calc_agency_aggregated.py --input_file ./outputs/llama3_llm_rec_letter_mitigate/bert-base-cased_results_0.csv -gm llama3 --mitigate --plot