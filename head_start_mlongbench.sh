#!/usr/bin/env bash
# Run the multilingual LongBench (XQuAD + MGSM in EN/DE/ZH/SW/...).
#
# Step 1 — build the data once (requires HF `datasets` or a --source_dir):
#
#   python3 data/build_multilingual_longbench.py \
#       --tasks xquad_en xquad_de xquad_zh mgsm_en mgsm_de mgsm_sw mgsm_zh \
#       --target_length 6000 --num_examples 150 \
#       --out_dir data/MultilingualLongBench
#
# Step 2 — run KV-cache compression with fused RTH heads.
#
# We compare three head_choice modes head-to-head, all with beta=1.5:
#   * `reason`        - original retrieval-reasoning heads (paper baseline)
#   * `fuse`          - average of [retrieval, RTH-ende, RTH-ensw]  (default)
#   * `rth_only`      - average of [RTH-ende, RTH-ensw]              (ablation)
#
# The exact components for `fuse` can be customised with --fuse_heads, e.g.
#   fuse_heads='llama_copy,trans_ende,trans_ensw,trans_enzh'
#
set -euo pipefail

mkdir -p ./longbench_logs/

max_capacity_prompts=128
attn_implementation=flash_attention_2
model_path=meta-llama/Meta-Llama-3.1-8B-Instruct
beta=1.5
temp=1
data_dir=./data/MultilingualLongBench
datasets='xquad_en,xquad_de,xquad_zh,mgsm_en,mgsm_de,mgsm_sw,mgsm_zh'

# (head_choice, fuse_heads) pairs to sweep.
configs=(
    "reason||"
    "fuse|llama_copy,trans_ende,trans_ensw"
    "rth_only|"
    "fuse|llama_copy,trans_ende,trans_ensw,trans_enzh,trans_zhen"
)

devices=(0 1 2 3 4 5 6 7)
counter=0

for cfg in "${configs[@]}"; do
    head_choice=${cfg%%|*}
    fuse_heads=${cfg##*|}
    device=${devices[counter]}
    log_tag=${head_choice}
    if [ -n "${fuse_heads}" ]; then
        log_tag="${head_choice}_$(echo ${fuse_heads} | tr ',' '+')"
    fi
    nohup bash head_base.sh \
        ${device} \
        ReasonKV \
        ${max_capacity_prompts} \
        ${attn_implementation} \
        ${model_path} \
        ${head_choice} \
        ${beta} \
        ${temp} \
        "${fuse_heads}" \
        "${datasets}" \
        "${data_dir}" \
        > ./longbench_logs/mlongbench_${log_tag}_base${max_capacity_prompts}_beta${beta}_temp${temp}.txt 2>&1 &
    ((counter+=1))
done

wait
echo "All multilingual LongBench runs finished."
