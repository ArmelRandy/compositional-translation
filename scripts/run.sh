MODEL_NAME_OR_PATH=$1
TOKENIZER_NAME_OR_PATH=$2
SRC=$3
TGT=$4
K=$5
SEED=$6
METHOD_DIVIDE=$7
MERGE_PROMPT=$8
NUMBER_OF_SUBPROBLEMS=$9
STEPS=${10}
NUMBER_OF_REFINING_STEPS=${11}
PORT_ID=${12}

ARGS="\
    --model_name_or_path $MODEL_NAME_OR_PATH\
    --tokenizer_name_or_path $TOKENIZER_NAME_OR_PATH\
    --src $SRC\
    --tgt $TGT\
    --request_batch_size 16\
    --inference_api vllm\
    --api_key <YOUR_API_KEY>\ # Only necessary for API-served models
    --max_samples 10000\
    --num_return_sequences 1\
    --num_beams 1\
    --max_new_tokens 2000\
    --temperature 0.0\
    --top_p 1.0\
    --repetition_penalty 1.0\
    --output_dir ./out/GENERATIONS/FLORES/bm25s\
    --k $K\
    --seed $SEED\
    --method_divide $METHOD_DIVIDE\
    --merge_prompt $MERGE_PROMPT\
    --method_translate vanilla\
    --selection_method greedy\
    --steps $STEPS\
    --verbose\
    --number_of_subproblems $NUMBER_OF_SUBPROBLEMS\
    --number_of_refining_steps $NUMBER_OF_REFINING_STEPS\
    --template_key 11\
    --retriever_type bm25s\
    --dataset_name_or_path flores\
    --number_of_merge_demonstrations 0\
    "

torchrun \
    --rdzv-backend=c10d\
    --rdzv-endpoint=localhost:$PORT_ID\
    main.py\
    $ARGS \
