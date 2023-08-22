from config import get_config
import pdb
args = get_config()

def trial_arg():
    
    if args.trial_mode:
        print('---chay che do trial_mode---')
        name_project='debug_comics'  # name project on wandb
        trial_ck='trial_ck/' 
        max_epochs=20
        data_subset=1
        shuffle=False
        batch_size=2
        resume=False
    else:
        print('---chay che do binh thuong---')
        name_project=args.project
        trial_ck=''
        max_epochs=args.max_epochs
        data_subset=1
        shuffle=True
        batch_size=args.batch_size
        resume=args.resume
        
    return name_project, trial_ck, max_epochs, data_subset, shuffle, batch_size, resume