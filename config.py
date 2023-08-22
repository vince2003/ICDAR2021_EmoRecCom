import argparse
import pdb
import torch
from torchvision import transforms
import model


#----------Config truc tiep tu terminal------------
def str2bool(v):
  return v.lower() in ('true', '1')

def get_config():
  config = parser.parse_args()
  return config  # Training settings

#----init argument
parser = argparse.ArgumentParser()

#------------------------------------------------------------------------------
# Operation
operate_arg = parser.add_argument_group('Operation')
operate_arg.add_argument('--evaluate', default=False, type=str2bool)
operate_arg.add_argument('--resume', default=True, type=str2bool)
operate_arg.add_argument('--expt_name', default='FigureQA', type=str)
operate_arg.add_argument('--trial_mode', default=False, type=str2bool)
#operate_arg.add_argument('--data_root', default='data', type=str)
operate_arg.add_argument('--ck_testing', default='data', type=str)


#------------------------------------------------------------------------------
#Name_project
project_arg = parser.add_argument_group('Name_project')
project_arg.add_argument('--project', default='run_spyder', type=str)
project_arg.add_argument('--name_graph', default='No_name', type=str)
project_arg.add_argument('--folder_ck', default='./ck_spyder/', type=str)
project_arg.add_argument('--resume_ck', default='./latest.pth', type=str)


# Dataset Store Definitions
dataset_arg = parser.add_argument_group('Dataset')
dataset_arg.add_argument('--train_file', default='train_MSE_BCE_processing_embedding50d_2labels_correct.json', type=str)
dataset_arg.add_argument('--val_files', default='val_BCE_processing_embedding50d_1label_correct.json', type=str)
dataset_arg.add_argument('--test_files', default='', type=str)
dataset_arg.add_argument('--dataset', default='FigureQA', type=str)
dataset_arg.add_argument('--high_img', default=384, type=int)
dataset_arg.add_argument('--wide_img', default=384, type=int)


# Data and Preprocessing
preprocess_arg = parser.add_argument_group('Preprocessing')
preprocess_arg.add_argument('--root', default='data', type=str)
preprocess_arg.add_argument('--data_subset', default=1, type=int)
preprocess_arg.add_argument('--batch_size', default=64, type=int)
preprocess_arg.add_argument('--lut_location', default='', type=str)


#------------------------------------------------------------------------------
# Network
net_arg = parser.add_argument_group('Network')
net_arg.add_argument('--word_emb_dim', default=50, type=int)
net_arg.add_argument('--ques_lstm_out', default=256, type=int)
net_arg.add_argument('--num_hidden_act', default=1024, type=int)
net_arg.add_argument('--num_rf_out', default=256, type=int)
net_arg.add_argument('--num_bimodal_units', default=256, type=int)
net_arg.add_argument('--loss', default='BCE', type=str)
net_arg.add_argument('--image_encoder', default='dense', type=str)
net_arg.add_argument('--dropout_classifier', default=True, type=str2bool)


# Training/Optimization
training_arg = parser.add_argument_group('Training')
training_arg.add_argument('--test_interval', default=1, type=int)
training_arg.add_argument('--test_every_epoch_after', default=20, type=int)
training_arg.add_argument('--max_epochs', default=100, type=int)
training_arg.add_argument('--overwrite_expt_dir', default=False, type=str2bool)
training_arg.add_argument('--grad_clip', default=50, type=int)

# Parameters for learning rate schedule
lr_arg = parser.add_argument_group('lr')
lr_arg.add_argument('--epoch_interval', default=10, type=int)
lr_arg.add_argument('--lr', default=5e-3, type=float)
lr_arg.add_argument('--lr_decay_step', default=2, type=int)
lr_arg.add_argument('--lr_decay_rate', default=.9, type=float)
lr_arg.add_argument('--warm_up_to_epoch', default=15, type=int)
lr_arg.add_argument('--warm_down_from_epoch', default=20, type=int)

# Loss function
loss_arg = parser.add_argument_group('loss')
loss_arg.add_argument('--bce_w', type=int, default=1)
loss_arg.add_argument('--mse_w', type=int, default=1)


#------------------------------------------------------------------------------
#============Config gian tiep==================================================
# Dataset Store Definitions
config_indirect = get_config()

dataset = config_indirect.dataset 

train_file = dict()
#train_file['FigureQA'] = 'train_MSE_BCE_processing_embedding50d_2labels_correct.json'
train_file[dataset] = config_indirect.train_file

val_files = dict()
#val_files['FigureQA'] = {'val1': '....', val2': '....', val3': '....'}  # Sample structure of validation
#val_files['FigureQA'] = {}
#val_files['FigureQA'] = {'val1': 'val_BCE_processing_embedding50d_1label_correct.json'}

if config_indirect.val_files:
    val_files[dataset] = {'val1': config_indirect.val_files}
else:
    val_files[dataset] = {}
    

test_files = dict()
#test_files['FigureQA'] = {'test1': '....', test2': '....', test3': '....'}  # Sample structure of test

if config_indirect.test_files:
    test_files[dataset] = {'test1': config_indirect.test_files}
else:    
    test_files[dataset] = {}
#test_files[dataset] = {'test1': 'FigureQA_test1_qa.json'}

transform_combo_train = dict()
transform_combo_test = dict()

transform_combo_train[dataset] = transforms.Compose([
    transforms.Resize((config_indirect.high_img, config_indirect.wide_img)),
    #transforms.RandomCrop(size=(448, 448), padding=8),
    #transforms.RandomRotation(2.8),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.9365, 0.9303, 0.9295],
                         std=[1, 1, 1])
])

# transform_combo_train['DVQA'] = transforms.Compose([
#     transforms.Resize(256),
#     #transforms.RandomCrop(size=(256, 256), padding=8),
#     #transforms.RandomRotation(2.8),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.8744, 0.8792, 0.8836],
#                          std=[1, 1, 1])
# ])

transform_combo_test[dataset] = transforms.Compose([
    transforms.Resize((config_indirect.high_img, config_indirect.wide_img)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.9365, 0.9303, 0.9295],
                         std=[1, 1, 1])

])

# transform_combo_test['DVQA'] = transforms.Compose([
#     transforms.Resize(256),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.8744, 0.8792, 0.8836],
#                          std=[1, 1, 1])
# ])

train_filename = train_file[dataset]
val_filenames = val_files[dataset]
test_filenames = test_files[dataset]

train_transform = transform_combo_train[dataset]
test_transform = transform_combo_test[dataset]

use_model = model.PReFIL
#==============================================================================
evaluate=config_indirect.evaluate
resume=config_indirect.resume
expt_name=config_indirect.expt_name

#------------------------------------------------------------------------------

# Data and Preprocessing

root = config_indirect.root  # This will be overwritten by command line argument
 # Should be defined above in the datastore section

data_subset = config_indirect.data_subset  # Random Fraction of data to use for training
batch_size = config_indirect.batch_size
lut_location = config_indirect.lut_location  # When training, LUT for transcript and answer token to idx is computed from scratch if left empty, or
# if your data specification has not changed, you can copy previously computed LUT.json and point to it to save time
# When resuming or evaluating, this is ignored and LUT computed for that experiment will be used

# Model Details


word_emb_dim = config_indirect.word_emb_dim
ques_lstm_out = config_indirect.ques_lstm_out
num_hidden_act = config_indirect.num_hidden_act
num_rf_out = config_indirect.num_rf_out
num_bimodal_units = config_indirect.num_bimodal_units

loss=config_indirect.loss
#loss='MSE'

image_encoder = config_indirect.image_encoder

if image_encoder == 'dense':
    densenet_config = (6, 6, 6)
    densenet_dim = [128, 160, 352] # Might be nice to compute according to densenet_config

# Training/Optimization

optimizer = torch.optim.Adamax
test_interval = config_indirect.test_interval  # In epochs
test_every_epoch_after = config_indirect.test_every_epoch_after
max_epochs = config_indirect.max_epochs
overwrite_expt_dir = config_indirect.overwrite_expt_dir  # For convenience, set to True while debugging
grad_clip = config_indirect.grad_clip

# Parameters for learning rate schedule
epoch_interval=config_indirect.epoch_interval

lr = config_indirect.lr
#lr_decay_step = config_indirect.lr_decay_step  # Decay every this many epochs
lr_decay_rate = config_indirect.lr_decay_rate


lr_decay_epochs = range(config_indirect.warm_up_to_epoch, config_indirect.warm_down_from_epoch, config_indirect.lr_decay_step)
lr_warmup_steps = [0.5 * config_indirect.lr, 1.0 * config_indirect.lr, 1.0 * config_indirect.lr, 1.5 * config_indirect.lr, 2.0 * config_indirect.lr]
dropout_classifier = config_indirect.dropout_classifier










