export CUDA_VISIBLE_DEVICES=0

set -e  # exit when error happen
source activate DVQA_torch38


wandb agent dang/uncategorized/mxm4tupb


#python -m compete_slack.py

