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
from cqa_dataloader_test import build_dataloaders_test
import pdb
# GPU_ID = '0'
# os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID
#os.environ['WANDB_MODE'] = 'dryrun'

#--------------------------------
#from trial_mode import trial_arg

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

result_path="./result_csv/"
if not os.path.exists(result_path):
    os.makedirs(result_path)  

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
    image_alb_save = Image.fromarray((npimg_hwc* 255).astype(np.uint8))
    image_alb_save.save(name+"_.png")
    print("saving images ...........")
    
    
def predict_test(net, dataloaders, epoch, config):
    """
    Evaluate 1 epoch on the given list of dataloaders and model, prints accuracy and saves predictions

    Args:
        net: Model instance to train
        dataloaders: List of dataloaders to use
        epoch: Current Epoch
        config: config
    """
    net.eval()
    total=0
    for data in dataloaders:
        results=[]
        id_image=[]
        #label=[]
        with torch.no_grad():
            for q, ans_bnr, ans_con, i, imgid in data:
                q = q.cuda()
                i = i.cuda()
                ql = (torch.ones(q.shape[0])*(q.shape[1])).cuda()
                #ans_bnr = ans_bnr.cuda()
                #pdb.set_trace()
                #show_dataloader(i,name="im_",normalize=True)
                
                
                p_bnr, p_con = net(i, q, ql, config)
                
                
                batch_pred=p_con.tolist()
                results.append(batch_pred)
                
                #batch_label=ans_bnr.tolist()
                #label.append(batch_label)
                total += len(p_bnr)
                
                id_image.extend(imgid)                

                inline_print(f'Processed {total} of {len(data) * data.batch_size} ')
                
        #pdb.set_trace()                   
        res=np.vstack(results)        
        id_img=np.vstack(id_image)        
        pre_df = pd.DataFrame({'id_imgs':id_img[:,0], 'Angry': res[:,0], 'Disgust': res[:,1],\
                                'Fear': res[:,2], 'Happy': res[:,3],'Sad': res[:,4],\
                                'Suprise': res[:,5], 'Neutral': res[:,6], 'Other': res[:,7]})
            
        # #pdb.set_trace()

        pre_df.to_csv('./result_csv/results_epoch'+str(epoch)+'.csv', index=True, header=True)
        print("\nDone")



def evaluate_saved(net, dataloader, config):
    ck_testing=args.ck_testing
    weights_path = os.path.join(ck_testing)
    saved = torch.load(weights_path)
    net.eval()
    net.load_state_dict(saved['model_state_dict'])
    predict_test(net, dataloader, saved['epoch'], config)


# %%
def main():
    seed_torch()

  
    test_data = build_dataloaders_test(CONFIG)

    print('Building model to train: ')
    #if CONFIG.dataset == 'FigureQA':
    net = CONFIG.use_model(44444, 8, CONFIG)
 

    print("Model Overview: ")
    print(net)
    net.cuda()
    print('Testing...')
    evaluate_saved(net, test_data, CONFIG)



if __name__ == "__main__":
    main()
