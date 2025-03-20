ARGS="
    --tokenizer google/mt5-xl\
    --model_name_or_path google/metricx-23-qe-xxl-v2p0\
    --dataset_name_or_path flores\
    --max_input_length 1024\
    --batch_size 1\
    --input_dir /path/to/compositional-translation/out/GENERATIONS/FLORES/bm25s/Meta-Llama-3.1-70B-Instruct-AWQ-INT4\
    --number_of_predictions 1012\
    "

cd /path/to/compositional-translation
export PYTHONPATH="/path/to/compositional-translation"
torchrun \
    --rdzv-backend=c10d\
    --rdzv-endpoint=localhost:29400\
    comptra/evaluate/metricx23/predict.py\
    $ARGS\