#!/bin/bash

export PYTHONPATH="..":$PYTHONPATH


MODEL_IDs=("large-v3-turbo" "large-v3")
MODEL_IDs=("large-v3")
# MODEL_IDs=("large-v3-turbo")
DEVICE_INDEX=0
BATCH_SIZE=64
# BATCH_SIZE=256
MODEL_IDs=("large-v3")
num_models=${#MODEL_IDs[@]}

# pip install -r ../requirements/requirements_trtllm.txt

for (( i=0; i<${num_models}; i++ ));
do
    MODEL_ID=${MODEL_IDs[$i]}
    # download_model $MODEL_ID
    # build_model $MODEL_ID
    # build_model_decoder_int8 $MODEL_ID

    # python3 run_eval.py \
    #     --model_id=whisper_${MODEL_ID} \
    #     --dataset_path="yuekai/aishell" \
    #     --dataset="test" \
    #     --split="test" \
    #     --device=${DEVICE_INDEX} \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1
    python3 run_faster_whisper_eval.py \
        --model_id=$MODEL_ID \
        --dataset_path="yuekai/speechio" \
        --dataset="SPEECHIO_ASR_ZH00011" \
        --split="test" \
        --device=${DEVICE_INDEX} \
        --batch_size=${BATCH_SIZE} \
        --max_eval_samples=-1

    # python3 run_eval.py \
    #     --model_id=whisper_${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="ami" \
    #     --split="test" \
    #     --device=${DEVICE_INDEX} \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1

    # python3 run_eval.py \
    #     --model_id=whisper_${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="earnings22" \
    #     --split="test" \
    #     --device=${DEVICE_INDEX} \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1

    # python3 run_eval.py \
    #     --model_id=whisper_${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="gigaspeech" \
    #     --split="test" \
    #     --device=${DEVICE_INDEX} \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1

    # python3 run_eval.py \
    #     --model_id=whisper_${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="librispeech" \
    #     --split="test.clean" \
    #     --device=${DEVICE_INDEX} \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1

    # python3 run_eval.py \
    #     --model_id=whisper_${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="librispeech" \
    #     --split="test.other" \
    #     --device=${DEVICE_INDEX} \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1

    # python3 run_eval.py \
    #     --model_id=whisper_${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="spgispeech" \
    #     --split="test" \
    #     --device=${DEVICE_INDEX} \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1

    # python3 run_eval.py \
    #     --model_id=whisper_${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="tedlium" \
    #     --split="test" \
    #     --device=${DEVICE_INDEX} \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1

    # python3 run_eval.py \
    #     --model_id=whisper_${MODEL_ID} \
    #     --dataset_path="hf-audio/esb-datasets-test-only-sorted" \
    #     --dataset="voxpopuli" \
    #     --split="test" \
    #     --device=${DEVICE_INDEX} \
    #     --batch_size=${BATCH_SIZE} \
    #     --max_eval_samples=-1

    # Evaluate results
    RUNDIR=`pwd` && \
    cd ../normalizer && \
    python3 -c "import eval_utils; eval_utils.score_results('${RUNDIR}/results', '${MODEL_ID}')" > $RUNDIR/log_${MODEL_ID}.txt && \
    cd $RUNDIR
    echo "Done evaluating $MODEL_ID"

done
