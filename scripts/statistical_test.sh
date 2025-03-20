ARGS="
    --model_name_or_path google/metricx-23-xxl-v2p0\ # Unbabel/XCOMET-XXL
    --dataset_name_or_path flores\
    --max_input_length 1024\
    --batch_size 1\
    --data_dir /path/to/compositional-translation/out/GENERATIONS/FLORES\
    --number_of_predictions 1012\
    --seed 122\
    --num_workers 8\
    --metric metricx\ # comet
    --languages Amharic Burmese Fijian Khmer Lao Samoan Sinhala Tsonga Turkmen Uyghur N\'ko\
    "

cd /path/to/compositional-translation
export PYTHONPATH="/path/to/compositional-translation"
torchrun \
    --rdzv-backend=c10d\
    --rdzv-endpoint=localhost:29500\
    comptra/evaluate/test.py\
    $ARGS\