#!/bin/bash
# Comprehensive test suite for SSD async speculative decoding.
# Launches all benchmarks as separate SLURM jobs (avoids OOM from sequential runs).
# Usage: bash tests/launch_tests.sh
# Results: /home/tkumar/ssd/test_results/suite_<timestamp>/
# After all jobs complete: bash tests/collect_results.sh <suite_dir>

set -e
cd /home/tkumar/ssd

TS=$(date +%Y%m%d_%H%M%S)
OUTDIR="/home/tkumar/ssd/test_results/suite_${TS}"
mkdir -p "$OUTDIR"
echo "Output dir: $OUTDIR"

BENCH="python -O bench/bench.py"

# ============================================================
# Job 0: Unified mask unit test (fast)
# ============================================================
cat > "${OUTDIR}/job0_unittest.sh" << 'JOBEOF'
#!/bin/bash
#SBATCH --exclusive
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:6
#SBATCH --time=00:05:00
cd /home/tkumar/ssd
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOBID
export PYTHONPATH=/home/tkumar/ssd
source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec
echo "Node: $(hostname), Job: $SLURM_JOBID"
echo '=== UNIT TEST: unified mask correctness ==='
python /home/tkumar/ssd/test_unified_mask.py
JOBEOF

# ============================================================
# Job 1: Llama 70B AR + Sync Spec (4 GPUs)
# ============================================================
cat > "${OUTDIR}/job1_ar_sync.sh" << 'JOBEOF'
#!/bin/bash
#SBATCH --exclusive
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:6
#SBATCH --time=00:30:00
cd /home/tkumar/ssd
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOBID
export PYTHONPATH=/home/tkumar/ssd
source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec
echo "Node: $(hostname), Job: $SLURM_JOBID"
echo '=== Llama 70B AR temp=0 ==='
python -O bench/bench.py --llama --size 70 --gpus 4 --b 1 --numseqs 8 --output_len 512 --temp 0
echo '=== Llama 70B Sync Spec K=6 f=3 temp=0 ==='
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 4 --spec --k 6 --f 3 --b 1 --numseqs 8 --output_len 512 --temp 0
JOBEOF

# ============================================================
# Job 2: Llama 70B Async B=1 (ns=8 and ns=128)
# ============================================================
cat > "${OUTDIR}/job2_async_b1.sh" << 'JOBEOF'
#!/bin/bash
#SBATCH --exclusive
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:6
#SBATCH --time=00:30:00
cd /home/tkumar/ssd
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOBID
export PYTHONPATH=/home/tkumar/ssd
source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec
echo "Node: $(hostname), Job: $SLURM_JOBID"
echo '=== Llama 70B Async K=6 f=3 B=1 ns=8 temp=0 ==='
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 0
echo '=== Llama 70B Async K=6 f=3 B=1 ns=128 temp=0 ==='
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 128 --output_len 512 --temp 0
JOBEOF

# ============================================================
# Job 3: Llama 70B Async batch size sweep (B=2,4,8)
# ============================================================
cat > "${OUTDIR}/job3_bsz_sweep.sh" << 'JOBEOF'
#!/bin/bash
#SBATCH --exclusive
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:6
#SBATCH --time=00:45:00
cd /home/tkumar/ssd
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOBID
export PYTHONPATH=/home/tkumar/ssd
source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec
echo "Node: $(hostname), Job: $SLURM_JOBID"
echo '=== Llama 70B Async K=6 f=3 B=2 temp=0 ==='
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 2 --jit --numseqs 8 --output_len 512 --temp 0
echo '=== Llama 70B Async K=6 f=3 B=4 temp=0 ==='
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 4 --jit --numseqs 16 --output_len 512 --temp 0
echo '=== Llama 70B Async K=6 f=3 B=8 temp=0 ==='
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 8 --jit --numseqs 32 --output_len 512 --temp 0
JOBEOF

# ============================================================
# Job 4: Llama 70B Async B=16
# ============================================================
cat > "${OUTDIR}/job4_b16.sh" << 'JOBEOF'
#!/bin/bash
#SBATCH --exclusive
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:6
#SBATCH --time=00:30:00
cd /home/tkumar/ssd
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOBID
export PYTHONPATH=/home/tkumar/ssd
source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec
echo "Node: $(hostname), Job: $SLURM_JOBID"
echo '=== Llama 70B Async K=6 f=3 B=16 temp=0 ==='
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 16 --jit --numseqs 64 --output_len 512 --temp 0
JOBEOF

# ============================================================
# Job 5: Llama 70B Async temp sweep (B=1)
# ============================================================
cat > "${OUTDIR}/job5_temp_sweep.sh" << 'JOBEOF'
#!/bin/bash
#SBATCH --exclusive
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:6
#SBATCH --time=00:45:00
cd /home/tkumar/ssd
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOBID
export PYTHONPATH=/home/tkumar/ssd
source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec
echo "Node: $(hostname), Job: $SLURM_JOBID"
echo '=== Llama 70B Async K=6 f=3 B=1 temp=0.7 ==='
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 0.7
echo '=== Llama 70B Async K=6 f=3 B=1 temp=1.0 ==='
python -O bench/bench.py --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 1.0
JOBEOF

# ============================================================
# Job 6: Llama 70B Ablations (NO-JIT, EAGER, JIT, sampler_x, custom FOL)
# ============================================================
cat > "${OUTDIR}/job6_ablations.sh" << 'JOBEOF'
#!/bin/bash
#SBATCH --exclusive
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:6
#SBATCH --time=01:30:00
cd /home/tkumar/ssd
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOBID
export PYTHONPATH=/home/tkumar/ssd
source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec
echo "Node: $(hostname), Job: $SLURM_JOBID"

BENCH="python -O bench/bench.py"
BASE="--llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --numseqs 8 --output_len 512 --temp 0"

echo '=== Llama 70B Async K=6 f=3 NO-JIT temp=0 ==='
$BENCH $BASE

echo '=== Llama 70B Async K=6 f=3 EAGER temp=0 ==='
$BENCH $BASE --eager

echo '=== Llama 70B Async K=6 f=3 JIT temp=0 ==='
$BENCH $BASE --jit

echo '=== Llama 70B Async K=6 f=3 JIT temp=1.3 x=0.05 ==='
$BENCH --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 1.3 --x 0.05

echo '=== Llama 70B Async K=6 f=3 JIT temp=1.3 x=0.25 ==='
$BENCH --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 1.3 --x 0.25

echo '=== Llama 70B Async K=6 f=3 JIT temp=0 custom-FOL ==='
$BENCH --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 1 --jit --numseqs 8 --output_len 512 --temp 0 --flh 3 3 3 3 3 3 3 --flm 15 5 1 0 0 0 0

echo '=== Llama 70B Async K=6 f=3 B=4 nonunif temp=0 ==='
$BENCH --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 4 --jit --numseqs 16 --output_len 512 --temp 0 --flh 3 3 3 3 3 3 3 --flm 15 5 1 0 0 0 0

echo '=== Llama 70B Async K=6 f=3 B=16 nonunif temp=0 ==='
$BENCH --llama --size 70 --draft 1 --gpus 5 --spec --async --k 6 --f 3 --b 16 --jit --numseqs 64 --output_len 512 --temp 0 --flh 3 3 3 3 3 3 3 --flm 15 5 1 0 0 0 0

echo '=== ALL DONE ==='
JOBEOF

# ============================================================
# Job 7: Qwen 32B (AR + Sync + Async)
# ============================================================
cat > "${OUTDIR}/job7_qwen.sh" << 'JOBEOF'
#!/bin/bash
#SBATCH --exclusive
#SBATCH --cpus-per-task=22
#SBATCH --gres=gpu:6
#SBATCH --time=00:45:00
cd /home/tkumar/ssd
export TRITON_CACHE_DIR=/tmp/triton_cache_$SLURM_JOBID
export PYTHONPATH=/home/tkumar/ssd
source /home/tkumar/miniconda3/etc/profile.d/conda.sh
conda activate async-spec
echo "Node: $(hostname), Job: $SLURM_JOBID"
echo '=== Qwen 32B AR temp=0 ==='
python -O bench/bench.py --size 32 --gpus 4 --b 1 --numseqs 8 --output_len 512 --temp 0
echo '=== Qwen 32B Sync Spec K=4 f=2 temp=0 ==='
python -O bench/bench.py --size 32 --draft 0.6 --gpus 4 --spec --k 4 --f 2 --b 1 --numseqs 8 --output_len 512 --temp 0
echo '=== Qwen 32B Async Spec K=4 f=2 temp=0 ==='
python -O bench/bench.py --size 32 --draft 0.6 --gpus 5 --spec --async --k 4 --f 2 --b 1 --jit --numseqs 8 --output_len 512 --temp 0
JOBEOF

# ============================================================
# Submit all jobs
# ============================================================
for script in "${OUTDIR}"/job*.sh; do
    NAME=$(basename "${script%.sh}")
    JOB=$(sbatch --output="${OUTDIR}/${NAME}_%j.log" "$script" 2>&1)
    echo "$JOB ($NAME)"
    echo "$JOB" >> "${OUTDIR}/jobids.txt"
done

echo ""
echo "All jobs submitted. Monitor: squeue -u tkumar"
echo "Results: $OUTDIR"
echo ""
echo "Once all jobs complete, run:"
echo "  bash tests/collect_results.sh $OUTDIR"
