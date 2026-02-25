#!/bin/bash
# launch_vllm_server.sh — Launch vLLM server with spec decode for benchmarking
# Run on a compute node (RS-06 or RS-20)

source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate vllm

MODEL=/data/shared/huggingface/hub/models--meta-llama--Llama-3.1-70B-Instruct/snapshots/1605565b47bb9346c5515c34102e054115b4f98b
DRAFT=/data/shared/huggingface/hub/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6

VLLM_USE_V1=0 python3 -m vllm.entrypoints.openai.api_server \
    --model $MODEL \
    --speculative-config "{\"method\": \"draft_model\", \"model\": \"$DRAFT\", \"num_speculative_tokens\": 5}" \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.7 \
    --max-num-seqs 1 \
    --disable-log-requests \
    --trust-remote-code \
    --port 40020
