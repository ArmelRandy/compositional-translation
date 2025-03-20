ARGS="\
    --data_dir /path/to/compositional-translation/out/GENERATIONS/FLORES/bm25s/gemma-2-9b-it\
    --dataset_name_or_path flores\
    --comet_model_path Unbabel/wmt22-comet-da\
    --model_name_or_path prometheus-eval/prometheus-7b-v2.0\
    --request_batch_size 16\
    --number_of_predictions 1012\
    --num_return_sequences 1\
    --num_beams 1\
    --max_new_tokens 1024\
    --temperature 0.0\
    --top_p 1.0\
    --repetition_penalty 1.0\
    --ensembling\
    --verbose\
    "

cd /path/to/compositional-translation
export PYTHONPATH="/path/to/compositional-translation"

torchrun \
    --rdzv-backend=c10d\
    --rdzv-endpoint=localhost:29500\
    /path/to/compositional-translation/comptra/evaluate/evaluate_p.py\
    $ARGS \