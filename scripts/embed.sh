cd /path/to/compositional-translation
export PYTHONPATH="/path/to/compositional-translation"
torchrun \
    --rdzv-backend=c10d\
    --rdzv-endpoint=localhost:29500\
    comptra/data/embed.py\
    --dataset_name_or_path ntrex\
    --batch_size 16\