export PYTHONPATH="${PYTHONPATH}:../../LIBERO"
export CHECKPOINT_DIR=../../../openvla_checkpoints
# export CUDA_VISIBLE_DEVICES=7

TASK_SUITE_NAME=${1:-}
USE_SAFE_PRUNER=${2:-}
[ -n "$TASK_SUITE_NAME" ] || { printf "TASK_SUITE_NAME: "; read -r TASK_SUITE_NAME; }
[ -n "$USE_SAFE_PRUNER" ] || { printf "USE_SAFE_PRUNER (True/False): "; read -r USE_SAFE_PRUNER; }
export TASK_SUITE_NAME USE_SAFE_PRUNER

case "$TASK_SUITE_NAME" in
  libero_spatial) CHECKPOINT_SUFFIX=spatial; SAFE_FASTV_R=0.8 ;;
  libero_object) CHECKPOINT_SUFFIX=object; SAFE_FASTV_R=0.75 ;;
  libero_goal) CHECKPOINT_SUFFIX=goal; SAFE_FASTV_R=0.9 ;;
  libero_10) CHECKPOINT_SUFFIX=10; SAFE_FASTV_R=0.7 ;;
  *) echo "Unsupported TASK_SUITE_NAME: $TASK_SUITE_NAME"; exit 1 ;;
esac

case "$USE_SAFE_PRUNER" in
  True|true|1|yes|Yes) export USE_SAFE_PRUNER=True FASTV_R="$SAFE_FASTV_R" ;;
  False|false|0|no|No|None|none|null|NULL) export USE_SAFE_PRUNER=False FASTV_R=0 ;;
  *) echo "USE_SAFE_PRUNER must be True or False"; exit 1 ;;
esac

export PRETRAINED_CHECKPOINT="${CHECKPOINT_DIR}/openvla-7b-oft-finetuned-libero-${CHECKPOINT_SUFFIX}"

export FASTV_K=3
export ATTN_SLICE="-57:-1"
export AVG_LAYERS="0,31"
export LAMBDA=1
export KEYFRAME_SIMILARITY_THRESHOLD=0.92

echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
echo "USE_SAFE_PRUNER=${USE_SAFE_PRUNER}"

python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint "$PRETRAINED_CHECKPOINT" \
  --task_suite_name "$TASK_SUITE_NAME" \
  --use_safe_pruner "$USE_SAFE_PRUNER"
