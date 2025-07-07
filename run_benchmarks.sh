# # # torchrun \
# # #   --nproc_per_node=8 \
# # #   --nnodes=1 \
# # # python3 src/old_main.py \
# # #     --seed 100 \
# # #     --strategy multimodal \
# # #     --model multimodal \
# # #     --patterns_per_exp 1 \
# # #     --benchmark rotated-mnist \
# # #     --n_experiences 1 \
# # #     --train_mb_size 60000 \
# # #     --eval_mb_size 60000 \
# # #     --train_epochs 1 \
# # #     --lr 0.01 \
# # #     --momentum 0 \
# # #     --output_dir ./eval_results \
# # #     --result_filename single_test.csv
# #     #  --memory_strength 0.5 \
# #     #  --plugin sgem \

# #!/usr/bin/env bash

# distributed=false
# debug=false

# for arg in "$@"; do
#   if [ "$arg" = "--distributed" ]; then
#     distributed=true
#   elif [ "$arg" = "--debug" ]; then
#     debug=true
#   else
#     echo "Unknown option: $arg"
#     echo "Usage: $0 [--distributed] [--debug]"
#     exit 1
#   fi
# done

# if [ "$distributed" = true ]; then
#   launcher="NCCL_SOCKET_IFNAME=lo NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT --rdzv_id=sgem_experiment --nproc_per_node=4 --nnodes=1 src/main.py"
# else
#   launcher="python3 -W ignore src/main.py"
# fi

# if [ "$debug" = true ]; then
#   launcher="$launcher --debug"
# fi

# eval $launcher


# # env vars
# # export NCCL_DEBUG=INFO
# # export NCCL_DEBUG_SUBSYS=ALL
# # export NCCL_ALGO=Ring
# # export NCCL_MIN_NCHANNELS=1     # use a single channel so Ring is chosen
# # export MASTER_ADDR=127.0.0.1
# # export MASTER_PORT=29500
# # export TORCH_DISTRIBUTED_DEBUG=DETAIL


#!/usr/bin/env bash
nohup python3 -u ./new_src/main.py > benchmark.log 2>&1 &
disown
