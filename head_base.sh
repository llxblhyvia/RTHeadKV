export CUDA_VISIBLE_DEVICES=$1

method=$2 # Support AdativeKV, ReasonKV
max_capacity_prompts=$3 # 128,2048 in paper
attn_implementation=$4 # Support "flash_attention_2"
model_path=$5
head_choice=$6
beta=$7
temp=$8
# Optional positional args (kept backwards-compatible):
#   $9  fuse_heads     - comma-separated keys, only used when head_choice=fuse
#   $10 datasets       - comma-separated dataset list (overrides default LongBench)
#   $11 data_dir       - directory holding <dataset>.jsonl
fuse_heads=${9:-}
datasets=${10:-}
data_dir=${11:-./data/LongBench}

tag="${head_choice}"
if [ -n "${fuse_heads}" ]; then
    tag="${head_choice}_$(echo ${fuse_heads} | tr ',' '+')"
fi
save_dir="./results/results_long_bench_${tag}_base${max_capacity_prompts}_beta${beta}_temp${temp}"

extra_args=""
if [ -n "${fuse_heads}" ]; then
    extra_args="${extra_args} --fuse_heads ${fuse_heads}"
fi
if [ -n "${datasets}" ]; then
    extra_args="${extra_args} --datasets ${datasets}"
fi

python3 run_longbench.py \
    --method ${method} \
    --model_path ${model_path} \
    --max_capacity_prompts ${max_capacity_prompts} \
    --head_choice ${head_choice} \
    --beta ${beta} \
    --temp ${temp} \
    --attn_implementation ${attn_implementation} \
    --save_dir ${save_dir} \
    --data_dir ${data_dir} \
    ${extra_args} \
    --use_cache True
