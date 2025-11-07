import pdb
import os
import pandas as pd
from datasets.dataset_generic import Generic_WSI_Classification_Dataset, Generic_MIL_Dataset, save_splits
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Creating splits for whole slide classification')
parser.add_argument('--label_frac', type=float, default= 1.0,
                    help='fraction of labels (default: 1)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--k', type=int, default=10,
                    help='number of splits (default: 10)')
parser.add_argument('--task', type=str, choices=['task_1_tumor_vs_normal', 'task_2_tumor_subtyping'])
parser.add_argument('--val_frac', type=float, default= 0.1,
                    help='fraction of labels for validation (default: 0.1)')
parser.add_argument('--test_frac', type=float, default= 0.1,
                    help='fraction of labels for test (default: 0.1)')
parser.add_argument('--classification_csv', type=str, default="",
                    help='classification information of the WSIs in csv format')

parser.add_argument('--label_file', type=str, default="",
                    help='File name of WSI label information.')
parser.add_argument('--split_folder', type=str, default="",
                    help='Folder name for split outputs.')
args = parser.parse_args()



def parse_label_dict(label_file):
    with open(label_file,'r') as label_file_open:
        label_file_cont=label_file_open.readlines()
        out_dic={}
        for line in label_file_cont:
            line=line.strip()
            if line!="":
                line=line.split()
                if len(line) == 2:
                    out_dic[str(line[0])]=int(line[1])
                else:
                    print ("\n\nERROR!:Label format is not correct. Check the label file and try again.\n\n ")
                    quit()
    return out_dic



label_data = parse_label_dict(args.label_file)

print ("\n\nLabel Parsing Results: ", label_data, "\n\n")

if args.task == 'task_1_tumor_vs_normal':
    args.n_classes=len(label_data.keys())
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.classification_csv, #'dataset_csv/tumor_vs_normal_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_data, # {'normal_tissue':0, 'tumor_tissue':1},
                            patient_strat=True,
                            ignore=[])

elif args.task == 'task_2_tumor_subtyping':
    args.n_classes=len(label_data.keys())
    dataset = Generic_WSI_Classification_Dataset(csv_path = args.classification_csv, #'dataset_csv/tumor_subtyping_dummy_clean.csv',
                            shuffle = False, 
                            seed = args.seed, 
                            print_info = True,
                            label_dict = label_data, #{'subtype_1':0, 'subtype_2':1, 'subtype_3':2},
                            patient_strat= True,
                            patient_voting='maj',
                            ignore=[])

else:
    raise NotImplementedError

num_slides_cls = np.array([len(cls_ids) for cls_ids in dataset.patient_cls_ids])
val_num = np.round(num_slides_cls * args.val_frac).astype(int)
test_num = np.round(num_slides_cls * args.test_frac).astype(int)

if __name__ == '__main__':
    if args.label_frac > 0:
        label_fracs = [args.label_frac]
    else:
        label_fracs = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for lf in label_fracs:
        split_dir = args.split_folder#+'/'+ str(args.task) + '_{}'.format(int(lf * 100))
        os.makedirs(split_dir, exist_ok=True)
        dataset.create_splits(k = args.k, val_num = val_num, test_num = test_num, label_frac=lf)
        for i in range(args.k):
            dataset.set_splits()
            descriptor_df = dataset.test_split_gen(return_descriptor=True)
            splits = dataset.return_splits(from_id=True)
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}.csv'.format(i)))
            save_splits(splits, ['train', 'val', 'test'], os.path.join(split_dir, 'splits_{}_bool.csv'.format(i)), boolean_style=True)
            descriptor_df.to_csv(os.path.join(split_dir, 'splits_{}_descriptor.csv'.format(i)))



