export CUDA_VISIBLE_DEVICES=0 # GPU RTX3090
source activate DVQA_torch38

name_project=comics_competition
name_graph=1bce_1mse
max_epoch=3
chay_thu=False
ck="./checkpoint/"
resume_ck_path="./checkpoint/1gim08fm/latest.pth"



#-----------------------train-----------------------------

python run_cqa.py --project $name_project --name_graph $name_graph --max_epochs $max_epoch --trial_mode $chay_thu --folder_ck $ck --batch_size 4 --lr 5e-4 --bce_w 1 --mse_w 1 --resume False --resume_ck $resume_ck_path


