#import argparse
import json
import os
import shutil
import sys
import numpy as np
import random
#import configs.config_template as CONFIG  # Allows use of autocomplete, this is overwritten by cmd line argument
import torch
import torch.nn as nn
import pandas as pd
from sklearn.metrics import roc_auc_score
import torchvision
from PIL import Image
from cqa_dataloader import build_dataloaders
import pdb
# GPU_ID = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
#os.environ['WANDB_MODE'] = 'dryrun'

#--------------------------------
from trial_mode import trial_arg

#---------Fix model-----------
def seed_torch(seed=0):
    random.seed(seed)#
    os.environ['PYTHONHASHSEED'] = str(seed)#
    np.random.seed(seed)#
    torch.manual_seed(seed)#
    torch.cuda.manual_seed(seed)#
    torch.cuda.manual_seed_all(seed)# # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False#
    torch.backends.cudnn.deterministic = True#
seed_torch()



#------------config-------------------------
import config as CONFIG
from config import get_config
args = get_config()

name_project, trial_ck, max_epochs, data_subset, shuffle, batch_size, resume=trial_arg()

#---------wandb-----------------------------
import wandb
wandb.init(config=args, project=name_project, name=args.name_graph, save_code=True)
api = wandb.Api()
id_wandb=wandb.run.id
#pdb.set_trace()

#-----------Folder to save checkpoint--------
folder_ck=args.folder_ck
path_ck=folder_ck+trial_ck+id_wandb
if not os.path.exists(path_ck):
    os.makedirs(path_ck)  
#-------------------------------------------


#EXPT_DIR = os.path.join(CONFIG.root, 'experiments', CONFIG.expt_name)



# if args.evaluate or args.resume:
#     pdb.set_trace()
#     if os.path.exists(EXPT_DIR):
#         shutil.copy(os.path.join(EXPT_DIR, f'config_{args.expt_name}.py'), f'configs/config_{args.expt_name}.py')
#     else:
#         sys.exit("Experiment Folder does not exist")


#exec(f'import configs.config_{args.expt_name} as CONFIG')
#CONFIG.root = args.data_root

#print('--------------CONFIG.root-------------',CONFIG.root)

#pdb.set_trace()


def inline_print(text):
    """
    A simple helper to print text inline. Helpful for displaying training progress among other things.
    Args:
        text: Text to print inline
    """
    sys.stdout.write('\r' + text)
    sys.stdout.flush()

def show_dataloader(tensor,name="gt_",normalize=True):
    img_grid=torchvision.utils.make_grid(tensor,nrow=len(tensor),padding=0,normalize=normalize)
    print("min value:",torch.min(img_grid))
    print("max value:",torch.max(img_grid))
    npimg = img_grid.cpu().data.numpy()
    npimg_hwc=np.transpose(npimg, (1, 2, 0))
    #plt.imshow(npimg_hwc)
    #plt.xticks(range(0,img_grid.shape[2]+img_grid.shape[1],img_grid.shape[1]))
    #plt.yticks([0,img_grid.shape[1]])    
    #plt.show()
    #plt.close()
    image_alb_save = Image.fromarray((npimg_hwc* 255).astype(np.uint8))
    image_alb_save.save(name+"_.png")
    print("saving images ...........")
    
    
def fit(net, dataloader, criterion_bnr, criterion_con, optimizer, epoch, config):
    """
    Train 1 epoch on the given dataloader and model

    Args:
        net: Model instance to train
        dataloader: Dataloader to use
        criterion: Training objective
        optimizer: Optimizer to use
        epoch: Current Epoch
        config: config
    """

    net.train()
    #correct = 0
    total = 0
    total_loss = 0
    
    results_con=[]
    results_bnr=[]
    #id_image=[]
    label=[]    
    
    for ques1, ans_bnr, ans_con, i, imgid in dataloader:
        #pdb.set_trace()
        
        ques1 = ques1.cuda()
        i = i.cuda()
        
        ql=(torch.ones(ques1.shape[0])*(ques1.shape[1])).cuda()
        ans_bnr = ans_bnr.cuda()  
        ans_con = ans_con.cuda() 
        #pdb.set_trace()
        #show_dataloader(i, 'image_', normalize=True) 
        
        
        p_bnr, p_con = net(i, ques1, ql, config)
        loss_bnr = criterion_bnr(p_bnr, ans_bnr)
        loss_con = criterion_con(p_con, ans_con)
        #pdb.set_trace()
        loss=args.bce_w*loss_bnr+args.mse_w*loss_con
        optimizer.zero_grad()
        loss.backward()
        #nn.utils.clip_grad_norm_(net.parameters(), config.grad_clip)
        optimizer.step()        
        #print(p)
        #pdb.set_trace()
        # if config.dataset == 'FigureQA':
        #     p_scale = torch.sigmoid(p)
        #     pred_class = p_scale >= 0.5
        #     c = float(torch.sum(pred_class.float() == a))
        # else:
        #     _, idx = p.max(dim=1)
        #     c = torch.sum(idx == a).item()
        #correct += c
        
        #-------continuous------
        batch_pred_con = p_con.tolist()
        results_con.append(batch_pred_con)   
        
        #--------bnr----------
        p_bnr = torch.sigmoid(p_bnr)
        batch_pred_bnr = p_bnr.tolist()
        results_bnr.append(batch_pred_bnr)
        
        
        
        batch_label = ans_bnr.tolist()
        label.append(batch_label)        
        
        
        total += len(ql)
        total_loss += loss * len(ql)
        #pdb.set_trace()
        inline_print(
            f'Running {dataloader.dataset.split}, Processed {total} of {len(dataloader) * dataloader.batch_size} '
            f'Loss: {total_loss / total}')
        
    res_con=np.vstack(results_con)
    res_bnr=np.vstack(results_bnr)
    
    lbl=np.vstack(label)        
    #pdb.set_trace()  
    auc_metric_con=auc_roc(lbl,res_con) 
    auc_metric_bnr=auc_roc(lbl,res_bnr)       
        
    print("---train auc con: %3f"%(auc_metric_con))
    print("---train auc bnr: %3f"%(auc_metric_bnr))
    wandb.log({"loss":(total_loss / total)}, step=epoch)
    
    wandb.log({"AUC_train_con":auc_metric_con}, step=epoch)
    wandb.log({"AUC_train_bnr":auc_metric_bnr}, step=epoch)

    #print(f'\nTrain Accuracy for Epoch {epoch + 1}: {correct / total}')

def auc_roc(label, result):
    auc=[]
    for i in range(7):               
        label_array=label[:,i] 
        result_array=result[:,i]
        #pdb.set_trace()
        auc_each=roc_auc_score(label_array, result_array)
        auc.append(auc_each)    
    return sum(auc)/len(auc)
    
    
    
def predict(net, dataloaders, epoch, config):
    """
    Evaluate 1 epoch on the given list of dataloaders and model, prints accuracy and saves predictions

    Args:
        net: Model instance to train
        dataloaders: List of dataloaders to use
        epoch: Current Epoch
        config: config
    """
    net.eval()
    for data in dataloaders:
        results_con=[]
        results_bnr=[]
        #id_image=[]
        label=[]           
        with torch.no_grad():
            for q, ans_bnr, ans_con, i, imgid in data:
                q = q.cuda()
                i = i.cuda()
                ql = (torch.ones(q.shape[0])*(q.shape[1])).cuda()
                ans_bnr = ans_bnr.cuda()
                #pdb.set_trace()
                
                
                p_bnr, p_con = net(i, q, ql, config)
                
                #------continous--------
                batch_pred_con=p_con.tolist()
                results_con.append(batch_pred_con)
                
                #---------binary------------                
                p_bnr = torch.sigmoid(p_bnr)
                batch_pred_bnr=p_bnr.tolist()
                results_bnr.append(batch_pred_bnr)
                
                batch_label=ans_bnr.tolist()
                label.append(batch_label)
                
                #id_image.extend(imgid)                

                # if 'test' not in data.dataset.split:
                #     #pdb.set_trace()
                #     print('AUC-ROC for validation')
                # else:
                #     print("AUC-ROC for testing") # test ko co label nen tao gia la: "answer":"0"
        
        #pdb.set_trace()                   
        res_con=np.vstack(results_con)
        res_bnr=np.vstack(results_bnr)
        
        lbl=np.vstack(label)        
        #pdb.set_trace()  
        auc_metric_con=auc_roc(lbl,res_con)
        auc_metric_bnr=auc_roc(lbl,res_bnr)
        print("---val auc_con: %3f "%(auc_metric_con))
        print("---val auc_bnr: %3f --- epoch:%3i/%3i: "%(auc_metric_bnr, epoch, max_epochs))
        
        wandb.log({"AUC_val_con":auc_metric_con}, step=epoch)
        wandb.log({"AUC_val_bnr":auc_metric_bnr}, step=epoch)
        #auc_roc=roc_auc_score(lbl, res, average='macro')
        
        # id_img=np.vstack(id_image)        
        # pre_df = pd.DataFrame({'id_imgs':id_img[:,0], 'Angry': res[:,0], 'Disgust': res[:,1],\
        #                         'Fear': res[:,2], 'Happy': res[:,3],'Sad': res[:,4],\
        #                         'Suprise': res[:,5], 'Neutral': res[:,6], 'Other': res[:,7]})
            
        # #pdb.set_trace()

        # pre_df.to_csv('./result_csv/results_epoch'+str(epoch)+'.csv', index=True, header=True)
        # print("Saved result CSV")



# def make_experiment_directory(config):
#     if not config.evaluate and not resume and not config.overwrite_expt_dir:
#         if os.path.exists(EXPT_DIR):
#             #raise RuntimeError(f'Experiment directory {EXPT_DIR} already exists, '
#                                #f'and the config is set to do not overwrite')
#             print("Ton tai folder experiment/FigureQA")

#     if not os.path.exists(EXPT_DIR):
#         os.makedirs(EXPT_DIR)


def update_learning_rate(epoch, optimizer, config):
    if epoch < len(config.lr_warmup_steps):
        optimizer.param_groups[0]['lr'] = config.lr_warmup_steps[epoch]
    elif epoch in config.lr_decay_epochs:
        optimizer.param_groups[0]['lr'] *= config.lr_decay_rate    
    return optimizer.param_groups[0]['lr']


def training_loop(config, net, train_loader, val_loaders, test_loaders, optimizer, criterion_bnr, criterion_con, start_epoch=0):
    for epoch in range(start_epoch, max_epochs):
        up_lr=update_learning_rate(epoch, optimizer, config)
        fit(net, train_loader, criterion_bnr, criterion_con, optimizer, epoch, config)
        curr_epoch_path = os.path.join(path_ck, str(epoch + 1) + '.pth')
        #pdb.set_trace()
        latest_path = os.path.join(path_ck, 'latest.pth')
        data = {'model_state_dict': net.state_dict(),
                'optim_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'lr': optimizer.param_groups[0]['lr']}
        if epoch % config.epoch_interval == 0:
            torch.save(data, curr_epoch_path)
            torch.save(data, latest_path)
            #print("save checkpoint at ",curr_epoch_path)

        if epoch % config.test_interval == 0:
            #pdb.set_trace()
            predict(net, val_loaders, epoch, config)
            predict(net, test_loaders, epoch, config)
            
        wandb.log({"Learning rate":up_lr}, step=epoch)



def evaluate_saved(net, dataloader, config):
    weights_path = os.path.join(path_ck, 'latest.pth')
    saved = torch.load(weights_path)
    net.eval()
    net.load_state_dict(saved['model_state_dict'])
    predict(net, dataloader, saved['epoch'], config)


# %%
def main():
    seed_torch()
    #pdb.set_trace()
    #make_experiment_directory(CONFIG)
    print('Building Dataloaders according to configuration')

    if CONFIG.evaluate or resume:
        #CONFIG.lut_location = os.path.join(EXPT_DIR, 'LUT.json')
        train_data, val_data, test_data = build_dataloaders(CONFIG)
    else:
        train_data, val_data, test_data = build_dataloaders(CONFIG)
        # lut_dict = {'ques2idx': train_data.dataset.ques2idx,
        #             'maxlen': train_data.dataset.maxlen}
        # json.dump(lut_dict, open(os.path.join(EXPT_DIR, 'LUT.json'), 'w'))
        # #pdb.set_trace()
        # shutil.copy(f'configs/config_{args.expt_name}.py',
        #             os.path.join(EXPT_DIR, 'config_' + args.expt_name + '.py'))

    #print('Building model to train: ')
    if CONFIG.dataset == 'FigureQA':
        net = CONFIG.use_model(44444, 8, CONFIG)
    # else:
    #     net = CONFIG.use_model(n1, n2, CONFIG)

    #print("Model Overview: ")
    #print(net)
    net.cuda()
    start_epoch = 0
    if not CONFIG.evaluate:
        print('Training...')
        optimizer = CONFIG.optimizer(net.parameters(), lr=CONFIG.lr)
        #criterion = torch.nn.CrossEntropyLoss()
        #if CONFIG.loss == 'BCE':
        criterion_bnr = nn.BCEWithLogitsLoss()
        print("------------BCE LOSS------------")
        #else:
        criterion_con = nn.MSELoss()
        print("------------MSE LOSS------------")

        if resume:            
            #resumed_data = torch.load(os.path.join(path_ck, 'latest.pth'))
            resumed_data = torch.load(args.resume_ck)
            print(f"Resuming from epoch {resumed_data['epoch'] + 1}")
            net.load_state_dict(resumed_data['model_state_dict'])
            optimizer = CONFIG.optimizer(net.parameters(), lr=resumed_data['lr'])
            optimizer.load_state_dict(resumed_data['optim_state_dict'])
            start_epoch = resumed_data['epoch']
        training_loop(CONFIG, net, train_data, val_data, test_data, optimizer, criterion_bnr, criterion_con, start_epoch)

    else:
        print('Evaluating...')
        evaluate_saved(net, test_data, CONFIG)
        evaluate_saved(net, val_data, CONFIG)


if __name__ == "__main__":
    main()
