ARGS="
    --dataset_name_or_path flores\
    --request_batch_size 16\
    --data_dir /path/to/compositional-translation/out/GENERATIONS/FLORES/TEaR/Meta-Llama-3.1-70B-Instruct-AWQ-INT4\
    --number_of_predictions 10\
    --is_qe\
    "

cd /path/to/compositional-translation
export PYTHONPATH="/path/to/compositional-translation"
torchrun \
    --rdzv-backend=c10d\
    --rdzv-endpoint=localhost:29500\
    comptra/evaluate/gemba/prompt.py\
    $ARGS\