#!/usr/bin/env python

# test evaluation__________________________________
import torch
import pandas as pd
from Multimodal_utils import data_loader, data_loader_for_multimodal, parse_sample_classification, feature_from_h5, get_feature_from_wsi_model, get_feature_from_omics_model, EarlyStopping, min_max_norm_TPM
import torch.nn as nn
import numpy as np
from models.model_clam import CLAM_SB
from multimodal_models.Transcriptome_model import TranscriptomeModel
from multimodal_models.Multimodal_model import MultimodalModel_MLP
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score 
import torch.backends.cudnn as cudnn
import csv

import os
import warnings
warnings.filterwarnings('ignore')

"""
Created on Mon Jan 18 17:41:10 2021

@author: minkj
"""
#________________ option parse _______________________________
import sys 

args = sys.argv[1:]

option_dict={'--multimodal_training_folder':'z_multimodal_chk', '--cuda_vis_dev':'0','--patch_level':'1', '--wsi_ext':'svs', '--encoder':"1", '--task':'1', '--num_split':'10', '--val_frac':'0.1', '--test_frac':'0.1', '--embed_dim':'1', '--drop_out':'0.25', '--model_type':'1', '--num_classes':'2','--create_heatmap':'3', '--restart':'1', '--patch_size':'256', '--num_cv_fold':'10', '--task':'1', '--multimodal_type':"MLP", '--wsi_folder':"", '--TPM_file':"",'--CV_split_number':'all'}
help="""
===================================================================================================================================
To run this pipeline, you must prepare the following two files first:

1. WSI classification file in CSV format.
Example (important: slide ID must be without extension): 

case_id,slide_id,label
patient_1,slide_1,Sensitive
patient_2,slide_2,Resistant
......

2. label_format

Example:
Sensitive  0
Resistant  1
......


Example to run in Singularity:

singularity exec --nv --bind /host/DATA/folder:/DATA WSIomics.sif WSIomics_inference.py --help


Example to run in Docker:

docker run -it -v /host/DATA/folder:/DATA managene7/wsiomics:v.2.0 bash

WSIomics_inference.py --help


* '/host/DATA/folder' => the host folder path that has the folder for input WSI files.
** Use options instead of '-help'

If you find any bugs or have a recommendation, send an email to: minkju3003@gmail.com
___________________________________________________________________________________________________________________________________

--task                  (required)          1: Validation with input and labeling data (default is 1) 
                                            2: Get prediction value with input data only.

--classification_csv    (only for task 1)   Classification csv file name [format (header): case_id, slide_id (no ext), label]
--label_format          (only for task 1)   Label file name. (Can be omitted for prediction.)
--CV_split_number       (only for task 2)   Cross-validation split numbers separated with a comma without a space. 
                                            (Example: --CV_split_number 1,3,6 ) (Default is 'all' splits) 

--wsi_folder            (for both tasks)    Name of the folder containing raw WSI files. (Can be omitted for transcriptome model only.)
--TPM_file              (for both tasks)    Name of the transcriptome TPM file.          (Can be omitted for WSI model only.)
--wsi_training_folder   (for both tasks)    Full folder name of the trained WSI models.  It ends with "_4_TRAINING".
--multimodal_chk_folder (for both tasks)    Folder name containing trained transcriptome and multimodal models. 
--cuda_vis_dev          (option)            Number for cuda_visible_device (default is '0')

--wsi_ext               (option)            Extension of image (default is 'svs')                       <== required for heatmap
--patch_level           (option)            Patch level of WSIs. 0, 1 (default), or 2                   <== required for heatmap
--patch_size            (option)            Patch size for creating patches (default is 256)
___________________________________________________________________________________________________________________________________
"""


if args==[]:
    print (help)
    sys.exit()
for i in range(len(args)):
    if args[i].startswith("-"):
        try:
            option_dict[args[i]]=args[i+1]
        except:
            if args[0]=="--help" or args[0]=="-h":
                print (help)
                sys.exit()

#__option check_____

curr_dir="/home/WSIomics"

##__output folder names__
created_patch_folder=option_dict['--wsi_folder']+"_1_CREATED_PATCHES"
feature_output=option_dict['--wsi_folder']+"_2_FEATURE_OUTPUTS"


#____________________________________<-------------

##__output command log file__
out_log=open(option_dict['--wsi_folder']+"__command_logs.txt",'w')

process_list_autogen_csv=created_patch_folder+"/process_list_autogen.csv"

##_______________________________

if option_dict['--task']=="1":
    task="task_1_tumor_vs_normal"
elif option_dict['--task']=="2":
    task="task_2_tumor_subtyping"
else:
    print ("ERROR!!: you chose the wrong '--task' option. Choose the right option and try it again.")
    sys.exit()


if option_dict['--encoder']=="1":
    model_name="uni_v1"
    embed_dim="1024"

if option_dict['--model_type']=="1":
    model_type="clam_sb"

if option_dict['--restart']=="6":
    option_dict['--create_heatmap']="3"
#__________________

if option_dict['--wsi_folder']!="":
    if option_dict['--create_heatmap'] in ['1','3']:
        # 1. create patches
    
        if 1 >= int(option_dict['--restart']):
            command_1=f"python {curr_dir}/create_patches_fp.py --source {option_dict['--wsi_folder']} --save_dir {created_patch_folder} --patch_level {str(option_dict['--patch_level'])} --patch_size {option_dict['--patch_size']} --seg --patch --stitch "
            out_log.write(command_1+"\n")
            print ("\n\n1/6___Create patches started..\n", command_1, "\n\n")
            create_patches=os.system(command_1)
        else:
           create_patches=0 
        
        if create_patches !=0:
            print ("ERROR!!: Patch creation failed. Please check input files or options and try it again.\n\n")
            sys.exit()
    
        # 2. feature extraction
        else:
            img_ext="."+option_dict['--wsi_ext']
            if 2 >= int(option_dict['--restart']):
                command_2=f"CUDA_VISIBLE_DEVICES={option_dict['--cuda_vis_dev']} python {curr_dir}/extract_features_fp.py --data_h5_dir {created_patch_folder} --data_slide_dir {option_dict['--wsi_folder']} --csv_path {process_list_autogen_csv} --feat_dir {feature_output} --batch_size 512 --slide_ext {img_ext} --model_name {model_name}"
                out_log.write(command_2+"\n")
                print ("\n\n2/6___Feature extraction started..\n",command_2, "\n\n")
                feature_extraction=os.system(command_2)
            else:
                feature_extraction=0
    
        if feature_extraction != 0:
            print ("ERROR!!: Feature extraction failed. Please check input files or options and try it again.\n\n")
            sys.exit()

device=f"cuda:{option_dict['--cuda_vis_dev']}" if torch.cuda.is_available() else "CPU"




# def Accuracy(target, probability):
#     softmax = nn.Softmax(dim=1)
#     y_true=target.squeeze()
#     y_pred=softmax(probability).argmax(dim=1)

#     accuracy=accuracy_score(y_true,y_pred)
    
#     return accuracy





# prediction____________________

def get_prediction_from_wsi_model(wsi_parameter_path, wsi_feature):
    model_dict = {"dropout": 0.25, 'n_classes': 2, "embed_dim": 1024}
    wsi_model=CLAM_SB(**model_dict).to(device)

    ckpt = torch.load(wsi_parameter_path, weights_only=True)
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    wsi_model.load_state_dict(ckpt_clean, strict=True)        

    with torch.no_grad():
        _, Y_prob, Y_hat, _, _ = wsi_model(wsi_feature.to(device), instance_eval=False, return_features=True) #Y_prob => softmax(probability)
    
    return Y_prob.cpu(),Y_hat.cpu()

def get_prediction_from_omics_model(transcriptome_model, sample_tpm, tdim):
    with torch.no_grad():
        logits, Y_prob, Y_hat, feature=transcriptome_model(sample_tpm.to(device))
    return Y_prob.cpu(), Y_hat.cpu()

def get_prediction_from_multimodal_model(multimodal_model, wsi_feature, transcriptome_feature, multimodal_type):
    with torch.no_grad():
        logits, Y_prob, Y_hat, feature=multimodal_model(wsi_feature.to(device), transcriptome_feature.to(device))
    return Y_prob.cpu(), Y_hat.cpu()

#_____________________________________

# validation______________________

def parse_sample_classification_wsi_validation(class_info_file, classification_format_file=""): #return [[wsi_id, label],...], {case_id: label}
    import csv
    class_info=csv.reader(open(class_info_file,'r'))
    
    if classification_format_file!="":
        classification_file=open(classification_format_file,'r')
        classification_file_lines=classification_file.readlines()
        label_conversion_dict={}
        for line in classification_file_lines:
            line=line.strip().split()
            try:
                if len(line)==2:
                    label_conversion_dict[line[0]]=int(line[1])
            except:
                print("""
                Error: classification format should contain a label and a number.
                      
                For example:
                Resistant   0
                Sensitive   1
    
                The provided classification format raised an error in the number part.
                Check the format and run again.
                """)
                sys.exit()
        init=0
        wsi_classification_list=[]
        case_classification_dic={}
        wsi_classification_list_for_multimodal=[]
    
        for row in class_info:
            
            if init==0:
                init=1
                pass
            else:
                wsi_classification_list.append([row[1], label_conversion_dict[row[2]]])
                wsi_classification_list_for_multimodal.append([row[1], row[0], label_conversion_dict[row[2]]])
                if row[0] not in case_classification_dic:
                    case_classification_dic[row[0]]=label_conversion_dict[row[2]]
                else:
                    if label_conversion_dict[row[2]] != case_classification_dic[row[0]]:
                        print(f"\n\nAlert!: {row[0]} has two different labels. Please check it and try again.\n\n")
                        sys.exit()
                
        return wsi_classification_list, case_classification_dic, wsi_classification_list_for_multimodal
    else:
        wsi_classification_list=[]
        case_classification_dic={}
        wsi_classification_list_for_multimodal=[]
    
        for row in class_info:
            
            if init==0:
                init=1
                pass
            else:
                wsi_classification_list.append([row[1]])
                wsi_classification_list_for_multimodal.append([row[1], row[0]])
                if row[0] not in case_classification_dic:
                    case_classification_dic[row[0]]=""
                
        return wsi_classification_list, case_classification_dic, wsi_classification_list_for_multimodal


def wsi_validation(wsi_class_info_list, feature_h5_folder, WSI_chkpoint_path): # return (accuracy, auc)
    wsi_predictions=[]
    targets=[] 
    for data in wsi_class_info_list: # multimodal_target_list = [wsi_id, case_id, label(number)]
        h5_file_path=os.path.join(feature_h5_folder,f'{data[0]}.h5')

        wsi_raw_feature=feature_from_h5(h5_file_path)
        wsi_prediction, wsi_hat=get_prediction_from_wsi_model(WSI_chkpoint_path, wsi_raw_feature)

        wsi_predictions.append(wsi_prediction[0])
        targets.append(data[1])
    prediction=torch.Tensor(np.array(wsi_predictions))
    targets=torch.from_numpy(np.reshape(np.array(targets),(-1,1)))

    softmax = nn.Softmax(dim=1)
    y_pred=softmax(prediction).argmax(dim=1)
    
    accuracy=accuracy_score(targets.cpu(), y_pred.cpu())
    f1=f1_score(targets.cpu().numpy(), y_pred.cpu().numpy())

    
    try:
        auc=roc_auc_score(torch.squeeze(targets.cpu()), prediction[:,1]) # <========== modified
    except:
        auc=0

    return accuracy, auc, f1


#######################################
def wsi_prediction(wsi_class_info_list, feature_h5_folder, WSI_chkpoint_path):
    wsi_prediction_list=[]
    for data in wsi_class_info_list: # multimodal_target_list = [wsi_id, case_id, label(number)]
        h5_file_path=os.path.join(feature_h5_folder,f'{data[0]}.h5')

        wsi_raw_feature=feature_from_h5(h5_file_path)
        wsi_prediction, wsi_hat=get_prediction_from_wsi_model(WSI_chkpoint_path, wsi_raw_feature).tolist()
        wsi_prediction_list.append([data[0], wsi_prediction])


    return wsi_prediction_list
#######################################        


def transcriptome_validation(tpm_df, case_classification_dic, transcriptome_model, tdim): # return (wsi_features, tpm, label)
    tpm_data=tpm_df
    tpm_data=tpm_data.to_dict('list')
    transcriptome_predictions=[]
    targets=[] 
    count=0
    for data in case_classification_dic.items(): # multimodal_target_list = [wsi_id, case_id, label(number)]

        if data[0] in tpm_data:
            sample_tpm=torch.FloatTensor(np.array(tpm_data[data[0]]))
            Y_prob, Y_hat=get_prediction_from_omics_model(transcriptome_model, sample_tpm, tdim)

            transcriptome_predictions.append(Y_prob)

            targets.append(data[1])

    prediction=torch.from_numpy(np.array(transcriptome_predictions))

    targets=torch.from_numpy(np.reshape(np.array(targets),(-1,1)))

    
    softmax = nn.Softmax(dim=1)
    y_pred=softmax(prediction).argmax(dim=1)    
    
    accuracy=accuracy_score(targets.cpu(), y_pred.cpu())
    f1=f1_score(targets.cpu().numpy(), y_pred.cpu().numpy())
    
    try:
        auc=roc_auc_score(targets.cpu(), prediction.cpu()[:,1]) # <======= modified
    except:
        auc=0

    return accuracy, auc, f1 # export data and label as numpy format.

#######################################
def transcriptome_prediction(tpm_df, case_classification_dic, transcriptome_model, tdim): # return (wsi_features, tpm, label)
    
    tpm_data=tpm_df
    tpm_data=tpm_data.to_dict('list')
    transcriptome_prediction_list=[]
    count=0
    for data in case_classification_dic.items(): # multimodal_target_list = [wsi_id, case_id, label(number)]

        if data[0] in tpm_data:
            sample_tpm=torch.FloatTensor(np.array(tpm_data[data[0]]))
            Y_prob, Y_hat=get_prediction_from_omics_model(transcriptome_model, sample_tpm, tdim)

            transcriptome_prediction_list.append([data[0], Y_prob])
    
    
    return transcriptome_prediction_list 
#######################################


def data_loader_for_multimodal_prediction(tpm_data, multimodal_input_list, feature_h5_folder, wsi_parameter_path, omics_parameter_path, cuda_vis_dev): # return (wsi_features, tpm, label)              
    
    tdim=tpm_data.shape[0]
    
    tpm_data_dic=tpm_data.to_dict('list')

    case_id_list=list(tpm_data_dic.keys())
    
    wsi_features=[]
    sample_tpms=[]
    id_list=[]
    count=0
    for data in multimodal_input_list: # multimodal_input_list = [wsi_id, case_id]
        if data[0][-4:]==".svs" or data[0][-4:]==".tif":
            data[0]=data[0][:-4]
            
        h5_file_path=os.path.join(feature_h5_folder,f'{data[0]}.h5')
        
        if data[1] in tpm_data_dic:
            sample_tpm=torch.FloatTensor(np.array(tpm_data_dic[data[1]]))

            tpm_feature=get_feature_from_omics_model(omics_parameter_path, sample_tpm, tdim, cuda_vis_dev)
            sample_tpms.append(torch.tensor(tpm_feature))

            wsi_raw_feature=feature_from_h5(h5_file_path)
            wsi_feature=get_feature_from_wsi_model(wsi_parameter_path, wsi_raw_feature, cuda_vis_dev)
            wsi_features.append(torch.tensor(wsi_feature))

            id_list.append(data)
            
    return np.array(wsi_features), np.array(sample_tpms), id_list # export data in numpy format.



def multimodal_validation(wsi_feature, transcriptome_feature, targets, multimodal_model, multimodal_type): # return (wsi_features, tpm, label)
    
    x1, x2, y=torch.from_numpy(wsi_feature).float().to(device), torch.from_numpy(transcriptome_feature).float().to(device),torch.from_numpy(targets).float().to(device)
    multimodal_Y_prob, multimodal_Y_hat =get_prediction_from_multimodal_model(multimodal_model, x1, x2, multimodal_type)

    prediction=torch.Tensor(multimodal_Y_prob)
    targets=torch.from_numpy(np.reshape(np.array(targets),(-1,1)))

    softmax = nn.Softmax(dim=1)
    y_pred=softmax(prediction).argmax(dim=1) 

    # accuracy=Accuracy(prediction,targets) 
    accuracy=accuracy_score(targets.cpu(), y_pred.cpu())
    f1=f1_score(targets.cpu().numpy(), y_pred.cpu().numpy())
    
    try:
        auc=roc_auc_score(targets.cpu(), multimodal_Y_prob.cpu()[:,1]) # <======= modified
    except:
        auc=0

    return accuracy, auc, f1

def multimodal_prediction(wsi_feature, transcriptome_feature, multimodal_model, multimodal_type): # return (wsi_features, tpm, label)
    
    x1, x2=torch.from_numpy(wsi_feature).float().to(device), torch.from_numpy(transcriptome_feature).float().to(device)
    multimodal_Y_prob, multimodal_Y_hat=get_prediction_from_multimodal_model(multimodal_model, x1, x2, multimodal_type)

    return multimodal_Y_prob


def parsing_label_format(label_format_file):
    file_open=open(label_format_file,'r')
    contents=file_open.readlines()
    out_dic={}
    for line in contents:
        line=line.strip().split()
        if line!=[]:
            out_dic[int(line[1])]=line[0]
    return out_dic

# prediction
def main():
    WSI_h5_feature_path=f"./{feature_output}/h5_files"
    WSI_chkpoint_path=f"./{option_dict['--wsi_training_folder']}"
    chk_point_pre=os.listdir(WSI_chkpoint_path)

    chk_point_list=[file for file in chk_point_pre if "checkpoint.pt" in file]
    if option_dict["--CV_split_number"]!="all":
        split_num=option_dict["--CV_split_number"].split(",")
        chk_point_list=[file for file in chk_point_list if file.split("_")[1] in split_num]
        
    chk_point_list.sort()
    num_chk_point=len(chk_point_list)
    
    
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(3000) 
    torch.manual_seed(3000)

    
    if option_dict['--task']=="2":
        option_dict['--label_format']==""

    label_dic=parsing_label_format(option_dict['--label_format'])
        
    wsi_class_info_list, case_classification_dic, wsi_class_info_list_for_multimodal =parse_sample_classification_wsi_validation(option_dict['--classification_csv'], option_dict['--label_format']) #return [[wsi_id, label],...]
    
    TPM_df=pd.read_csv(option_dict['--TPM_file'], index_col=0)
    
    init=0

    if option_dict['--task']=="1":
        out_csv=csv.writer(open(option_dict['--classification_csv'][:-4]+"_"+option_dict["--multimodal_chk_folder"]+"_validation.csv",'w'))
        out_csv.writerow(["Cross Validation Splits", "WSI Accuracy","WSI F1", "WSI AUC", "Transcriptome Accuracy","Transcriptome F1",  "Transcriptome AUC", "Multimodal Accuracy","Multimodal F1", "Multimodal AUC" ])
    elif option_dict['--task']=="2":
        if option_dict["--wsi_folder"]=="":
            out_csv=csv.writer(open(option_dict['--TPM_file'][:-4]+"_"+option_dict["--multimodal_chk_folder"]+"_prediction.csv",'w'))
            
        elif option_dict["--TPM_file"]=="":
            out_csv=csv.writer(open(option_dict['--wsi_folder']+"_"+option_dict["--multimodal_chk_folder"]+"_prediction.csv",'w'))
        elif option_dict["--wsi_folder"]!="" and option_dict["--TPM_file"]!="":
            out_csv=csv.writer(open(option_dict['--wsi_folder']+"_"+option_dict["--multimodal_chk_folder"]+"_Multimodal_prediction.csv",'w'))
        out_csv.writerow(["Modality", "Patient_ID", "WSI_ID", "Classification_result", "prediction_value"])
    
    
    for chk_point in chk_point_list:
        #WSI_validation
        WSI_chkpoint_file=WSI_chkpoint_path+f"/{chk_point}"
        split=chk_point.split("_")[1]
        
        transcriptome_chkpoint_file=option_dict["--multimodal_chk_folder"]+f"/transcriptome_model_splits_{split}.pth"
        multimodal_chkpoint_file=option_dict["--multimodal_chk_folder"]+f"/multimodal_model_splits_{split}.pth"
    
        ## load transcriptome model
        transcriptome_model=TranscriptomeModel().to(device)
        transcriptome_model.eval()
    
        transcriptome_state_dict=torch.load(transcriptome_chkpoint_file, weights_only=True, map_location="cuda")
        
        transcriptome_model_state_dict = {k: v for k, v in transcriptome_state_dict.items() if k in transcriptome_model.state_dict()} 
        
        norm_min_max_dic = {k: v for k, v in transcriptome_state_dict.items() if k not in transcriptome_model.state_dict()}
        
        tdim=len(norm_min_max_dic)
    
        transcriptome_model=TranscriptomeModel(tdim=tdim).to(device)
        transcriptome_model.eval()
    
        
        transcriptome_model.load_state_dict(transcriptome_model_state_dict)
        
    
        ## load multimodal model
        multimodal_model=MultimodalModel_MLP(transcriptome_dim=tdim).to(device)
        multimodal_model.eval()
    
        multimodal_state_dict=torch.load(multimodal_chkpoint_file , weights_only=True, map_location="cuda")
        multimodal_model_state_dict = {k: v for k, v in multimodal_state_dict.items() if k in multimodal_model.state_dict()}
        
        multimodal_model.load_state_dict(multimodal_model_state_dict)   
    
        # marker filtering and min-max_scaling of marker TPM data
        if init==0:
            init=1
            
            marker_IDs=list(norm_min_max_dic.keys())
            marker_TPM_df=TPM_df.filter(marker_IDs, axis=0)
    
            # Min-Max scaling of TPM
            index_val=marker_TPM_df.index
            column_val=marker_TPM_df.columns
        
            marker_TPM_np_values=marker_TPM_df.values
            norm_tpm_list=[]
            for posi, data in enumerate(marker_TPM_np_values):
                gene_ID=index_val[posi]
    
                max_val=float(norm_min_max_dic[gene_ID]['max'])
                min_val=float(norm_min_max_dic[gene_ID]['min'])
                if max_val!=0 and max_val!=min_val:
                    norm_data=(data-min_val)/(max_val-min_val)
                else:
                    norm_data=data
                norm_tpm_list.append(norm_data)
    
            norm_tpm_df=pd.DataFrame(norm_tpm_list, index=index_val, columns=column_val)
    
    
        if option_dict['--task']=="1":
            # Get validation results.
            ## WSI results
            if option_dict['--wsi_folder']!="":
                wsi_accuracy, wsi_auc, wsi_f1=wsi_validation(wsi_class_info_list, WSI_h5_feature_path, WSI_chkpoint_file)
            else:
                wsi_accuracy, wsi_auc, wsi_f1=0,0,0
            ## Transcriptome results
            if option_dict['--TPM_file']!="":
                transcriptome_accuracy, transcriptome_auc, transcriptome_f1=transcriptome_validation(norm_tpm_df, case_classification_dic, transcriptome_model, tdim)
            else:
                transcriptome_accuracy, transcriptome_auc, transcriptome_f1=0,0,0
            ## Multimodal results
            if option_dict['--wsi_folder']!="" and option_dict['--TPM_file']!="":
                wsi_feature, transcriptome_feature, targets=data_loader_for_multimodal(norm_tpm_df, wsi_class_info_list_for_multimodal, WSI_h5_feature_path, WSI_chkpoint_file, transcriptome_chkpoint_file, option_dict['--cuda_vis_dev'])
                multimodal_accuracy, multimodal_auc, multimodal_f1=multimodal_validation(wsi_feature, transcriptome_feature, targets, multimodal_model, option_dict['--multimodal_type'])
            else:
                multimodal_accuracy, multimodal_auc, multimodal_f1=0,0,0
                

            print("\n===============================================================================")
            print(f"Cross Validation (CV) Splits \t\t| Acc. \t\t| F1 \t\t| AUC")
            print("_______________________________________________________________________________")
            print(f"WSI_model_CV_splits_{split} \t\t\t| {wsi_accuracy:.3f} \t| {wsi_f1:.3f} \t| {wsi_auc:.3f}")
            print(f"Transcriptome_model_CV_splits_{split} \t| {transcriptome_accuracy:.3f} \t| {transcriptome_f1:.3f} \t| {transcriptome_auc:.3f}")
            print(f"Multimodal_model_CV_splits_{split} \t\t| {multimodal_accuracy:.3f} \t| {multimodal_f1:.3f} \t| {multimodal_auc:.3f}")
            print("===============================================================================")

            
            out_csv.writerow([f"CV_splits_{split}", f"{wsi_accuracy:.3f}",f"{wsi_f1:.3f}", f"{wsi_auc:.3f}",f"{transcriptome_accuracy:.3f}",f"{transcriptome_f1:.3f}",f"{transcriptome_auc:.3f}",f"{multimodal_accuracy:.3f}",f"{multimodal_f1:.3f}", f"{multimodal_auc:.3f}"])
    
        elif option_dict['--task']=="2":
            # Get prediction results.
            ## WSI results
            print(f"\n============= Cross Validation Splits {split} =============")
            if option_dict['--wsi_folder']!="":
                wsi_prediction_list=wsi_prediction(wsi_class_info_list, WSI_h5_feature_path, WSI_chkpoint_file)

                out_csv.writerow(["==========","Cross", "Validation", f"Splits-{split}", "=========="])
                for wsi_prediction_cont in wsi_prediction_list:
                    id=wsi_prediction_cont[0]
                    prediction_index=np.argmax(wsi_prediction_cont[1][0])
                    classification=label_dic[prediction_index]
                    prediction_value=wsi_prediction_cont[1][0][prediction_index]
                    print("WSI","\t","-","\t",id, "\t",classification,"\t", f"{prediction_value:.3f}")
                    out_csv.writerow(["WSI","-", id, classification, f"{prediction_value:.3f}"])
                    
            
            ## Transcriptome results
            print(f"\n============= Cross Validation Splits {split} =============")
            if option_dict['--TPM_file']!="":
                transcriptome_prediction_list=transcriptome_prediction(norm_tpm_df, case_classification_dic, transcriptome_model, tdim)
                
                out_csv.writerow(["==========","Cross", "Validation", f"Splits-{split}", "=========="])
                for transcriptome_prediction_cont in transcriptome_prediction_list:
                    id=transcriptome_prediction_cont[0]
                    prediction_index=np.argmax(transcriptome_prediction_cont[1].tolist())
                    classification=label_dic[prediction_index]
                    prediction_value=transcriptome_prediction_cont[1].tolist()[prediction_index]
                    print("Transcriptome", "\t",id,"\t", "-","\t",classification,"\t", f"{prediction_value:.3f}")
                    out_csv.writerow(["Transcriptome", id, "-", classification, f"{prediction_value:.3f}"])

            ## Multimodal results
            print(f"\n============= Cross Validation Splits {split} =============")
            if option_dict['--wsi_folder']!="" and option_dict['--TPM_file']!="":
                wsi_feature, transcriptome_feature, id_list=data_loader_for_multimodal_prediction(norm_tpm_df, wsi_class_info_list_for_multimodal, WSI_h5_feature_path, WSI_chkpoint_file, transcriptome_chkpoint_file, option_dict['--cuda_vis_dev'])
                multimodal_Y_prob=multimodal_prediction(wsi_feature, transcriptome_feature, multimodal_model, option_dict['--multimodal_type'])
                
                out_csv.writerow(["==========","Cross", "Validation", f"Splits-{split}", "=========="])
                for ind in range(len(id_list)):
                    id=id_list[ind]
                    prediction_index=np.argmax(multimodal_Y_prob[ind].tolist())
                    classification=label_dic[prediction_index]
                    prediction_value=multimodal_Y_prob[ind].tolist()[prediction_index]
                    print("Multimodal", "\t",id[1], "\t",id[0], "\t",classification,"\t", f"{prediction_value:.3f}")
                    out_csv.writerow(["Multimodal", id[1], id[0], classification, f"{prediction_value:.3f}"])
                

if __name__ == "__main__":
    main()