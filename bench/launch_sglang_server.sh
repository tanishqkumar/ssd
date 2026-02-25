#!/bin/bash
# launch_sglang_server.sh — Launch SGLang server with spec decode for benchmarking
# Run on a compute node (RS-06 or RS-20)

source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec

MODEL=/data/shared/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
DRAFT=/data/shared/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6

python3 -m sglang.launch_server \
    --model-path $MODEL \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path $DRAFT \
    --speculative-num-steps 5 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 \
    --tp 4 \
    --mem-fraction-static 0.7 \
    --disable-radix-cache \
    --max-running-requests 1 \
    --trust-remote-code \
    --port 40010
