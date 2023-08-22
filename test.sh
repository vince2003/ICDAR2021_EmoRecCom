export CUDA_VISIBLE_DEVICES=0 # GPU RTX3090
source activate DVQA_torch38

#ck_testing="./checkpoint/1og1vdxs/latest.pth"
ck_testing="./211.pth"
test_files="test_processing_embedding50d_correct.json"


#-----------------------train-----------------------------

python testing.py --ck_testing $ck_testing --test_files $test_files --batch_size 32


