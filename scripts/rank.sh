ARGS="
    --data_dir /path/to/compositional-translation/out/GENERATIONS/FLORES/bm25s/Meta-Llama-3.1-70B-Instruct-AWQ-INT4\
    --dataset_name_or_path flores\
    --comet_model_path Unbabel/wmt22-comet-da\
    --model_name_or_path ibraheemmoosa/mt-ranker-xxl\
    --request_batch_size 16\
    --number_of_predictions 1012\
    --verbose\
    "

cd /path/to/compositional-translation
export PYTHONPATH="/path/to/compositional-translation"
torchrun \
    --rdzv-backend=c10d\
    --rdzv-endpoint=localhost:29400\
    /path/to/compositional-translation/comptra/evaluate/mtranker.py\
    $ARGS\