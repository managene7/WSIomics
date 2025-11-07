import pandas as pd
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
# import GPUtil
# from sklearn.model_selection import cross_val_score 
import sys
import os
# from sklearn.preprocessing import StandardScaler
import h5py
import torch.nn.functional as F
from multimodal_models.Transcriptome_model import TranscriptomeModel
from models.model_clam import CLAM_SB
import math
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

device="cuda" if torch.cuda.is_available() else "CPU"



# min-max regularization of TPM data

def log2conversion(tpm_data_df):
    columns=tpm_data_df.columns.values
    index=tpm_data_df.index.values
    tpm_data_trans = tpm_data_df.T
    tpm_data_trans_log2 = np.log2(tpm_data_trans.values+1).T
    out_df=pd.DataFrame(tpm_data_trans_log2, columns=columns, index=index)
    return out_df



def min_max_norm_TPM(TPM_df, PFI_file="", min_max_dic={}, num_markers=0, log2="1", scaling="1", filtering="1", mean_threshold="1"):
    # print("\n[ log2 transformation and Min-Max scaling of input data is in progress.. ]")
    if type(TPM_df)==str:
        TPM_df=pd.read_csv(TPM_df, index_col=0)
        TPM_df=TPM_df.dropna(axis=0)
    
    if int(num_markers)!=0:
        if filtering=="1":
            TPM_df_filtered=TPM_df[TPM_df.mean(axis=1)>=int(mean_threshold)]
        else:
            TPM_df_filtered=TPM_df
        if log2=="1":
            TPM_df_for_marker_search=log2conversion(TPM_df_filtered)
        else:
            TPM_df_for_marker_search=TPM_df_filtered
    else:
        TPM_df_for_marker_search=TPM_df
    
    # Don't do scaling if scaling option is "1" and finish.
    if scaling!="1":
        return TPM_df_for_marker_search, {}
        
    # TPM_df=log2conversion(TPM_df)
    
    if PFI_file !="":
        pfi_info=open(PFI_file,'r')
        pfi_value_ID=[[float(line.strip().split(',')[1]), line.strip().split(',')[0]] for line in pfi_info.readlines()[1:]]
        pfi_value_ID.sort()
        pfi_sorted_ID=[value[1] for value in pfi_value_ID]

        TPM_df_max1_sorted=TPM_df.filter(pfi_sorted_ID, axis=1)

        TPM_df_max1_sorted_columns=TPM_df_max1_sorted.columns
        TPM_df_max1_sorted_columns_filtered=[]
        for id in TPM_df_max1_sorted_columns:
            if "." not in id:
                TPM_df_max1_sorted_columns_filtered.append(id)
        
        TPM_df_max1_sorted=TPM_df_max1_sorted.filter(TPM_df_max1_sorted_columns_filtered,axis=1)

        pfi_info=open(PFI_file,'r')

        pfi_id_list=[id for id in pfi_sorted_ID if id in TPM_df_max1_sorted_columns_filtered]
        
        TPM_df_max1_sorted_filtered=TPM_df_max1_sorted.filter(pfi_id_list, axis=1)
    else:
        TPM_df_max1_sorted_filtered=TPM_df


    index_val=TPM_df_for_marker_search.index
    column_val=TPM_df_for_marker_search.columns

    TPM_np_max1_sorted=TPM_df_for_marker_search.values


    scaled_tpm_list=[]
    if min_max_dic=={}:
        min_max_dic={}
        for posi, data in enumerate(TPM_np_max1_sorted):
            gene_ID=index_val[posi]

            # max_val=max(data)
            # min_val=min(data)

            mean_val=np.mean(data)
            std_val=np.std(data)

            
            scaled_data=(data-mean_val)/std_val

            scaled_tpm_list.append(scaled_data)

            min_max_dic[gene_ID]={'mean':float(mean_val), 'std':float(std_val)}

        norm_tpm_df=pd.DataFrame(scaled_tpm_list, index=index_val, columns=column_val)
        return norm_tpm_df, min_max_dic
    else:
        for posi, data in enumerate(TPM_np_max1_sorted):
            gene_ID=index_val[posi]

            mean_val=float(min_max_dic[gene_ID]['mean'])
            std_val=float(min_max_dic[gene_ID]['std'])
            
            scaled_data=(data-mean_val)/std_val
            scaled_tpm_list.append(scaled_data)

        norm_tpm_df=pd.DataFrame(scaled_tpm_list, index=index_val, columns=column_val)

        return norm_tpm_df, min_max_dic



def find_TPM_marker_gene(TPM_file, PFI_file="",num_markers=300, exclude=[], tag=0, log2="1", scaling="1", filtering="1", mean_threshold="1", marker_folder=""):

    # return normalized TPM without idetifying marker gene
    if PFI_file=="":
        norm_TPM_df, min_max_dic=min_max_norm_TPM(TPM_file, PFI_file, num_markers=int(num_markers), log2=log2, scaling=scaling, filtering=filtering, mean_threshold=mean_threshold)
        return norm_TPM_df, min_max_dic
    # find marker genes and normalize TPMs
    else:
        norm_TPM_df, min_max_dic=min_max_norm_TPM(TPM_file, PFI_file, num_markers=int(num_markers), log2=log2, scaling=scaling, filtering=filtering, mean_threshold=mean_threshold)

        print(f"[ Validation and Testing data are excluded from the DataFrame for marker search, and imputed with flanking data.. ]")
        
        pfi_info=pd.read_csv(PFI_file, index_col=0)
        col_key=pfi_info.columns.item()
        
        pfi_dic=pfi_info.to_dict()[col_key]

        norm_TPM_df_for_imputation=norm_TPM_df.copy()
        
        norm_TPM_df_for_imputation=norm_TPM_df_for_imputation.filter(pfi_dic.keys(), axis=1)
        
        num_row=norm_TPM_df_for_imputation.shape[0]
        
        for id in exclude: # <== remove validation and testing data
            norm_TPM_df_for_imputation[id]=["drop"]*num_row
        norm_tpm_value=norm_TPM_df_for_imputation.values
        
        norm_tpm_filtered_id=norm_TPM_df_for_imputation.columns.tolist() 
        norm_tpm_gene_id=norm_TPM_df.index
        norm_TPM_df_column=norm_TPM_df.columns.tolist()

        pfi_np=np.array([pfi_dic[id] for id in norm_tpm_filtered_id if id in pfi_dic])

        #___________________________

        

        z_list_pre=[]
        z_list_min_max=[]
        std_list_min_max=[]
        corr_list=[]
        flanking=5
        
        for ind in range(len(norm_tpm_value)):
            x=range(len(norm_tpm_value[0]))
            id=norm_tpm_gene_id[ind]

            y_pre=norm_tpm_value[ind].tolist()
            y_pre_average=sum([num for num in y_pre if num!="drop"])/len([num for num in y_pre if num!="drop"])

            y=[]
            for k in range(len(y_pre)):
                if k<= flanking:
                    left=k
                else:
                    left=k-flanking
                if k+flanking >=len(y_pre):
                    right=len(y)
                else:
                    right=k+flanking
                window=y_pre[left:right]
                
                if "drop" not in window:
                    y.append(sum(window)/len(window))
                else:
                    filtered_window=[cont for cont in window if cont!="drop"]
                    if len(filtered_window)!=0:
                        y.append(sum(filtered_window)/len(filtered_window))
                    else:
                        y.append(y_pre_average)
                    

            z=np.polyfit(x,y,1)
            p=np.poly1d(z)
            
            if abs(z[0]) >= 0.00001:
                z_list_min_max.append([abs(z[0])])

                corr=np.corrcoef(pfi_np, y)[0,1]
                corr_list.append(abs(round(corr,3)))
                
                z_list_pre.append([abs(z[0]), id])
        
        # Get 2000th corr value
        corr_list_copy=corr_list[:]
        corr_list_copy.sort()
        corr_list_copy.reverse()
        if len(corr_list_copy)>=2000:
            corr_rank=corr_list_copy[1999]
        else:
            corr_rank=0
        #_______________________
        

        z_list_final=[]

        for ind in range(len(z_list_min_max)):
            
            if corr_list[ind] >= corr_rank:
                z_list_final.append(z_list_pre[ind])
        
        z_list_final.sort()
        z_list_final.reverse()
        
        sorted_id_list=[cont[1] for cont in z_list_final]

        if int(num_markers) >= len(sorted_id_list):
            num_markers=len(sorted_id_list)
            print(f"[ <Alert!>: Number of markers is adjusted to {num_markers} due to not enough markers over thresholds. ]\n")
        else:
            print(f"[ High-ranked {num_markers} markers are selected among a total of {len(sorted_id_list)} markers over thresholds. ]\n")

        num_markers=int(num_markers)

        

        all_marker_list=[]
        for k in range(num_markers):
            all_marker_list.append(sorted_id_list[k])
        
        tpm_df=pd.read_csv(TPM_file, index_col=0)
        tpm_marker_df=tpm_df.filter(all_marker_list, axis=0)

        marker_file_path=os.path.join(marker_folder, TPM_file[:-4]+f"__marker_TPM_{tag}_{num_markers}.csv")
        tpm_marker_df.to_csv(marker_file_path)

        marker_min_max_dic={}

        return tpm_marker_df, marker_min_max_dic



def marker_search_data_processing_XGBoost(training_dataset, classification_file_name, classification_format_file):
    classification_format=open(classification_format_file, 'r')
    classification_format_dic={line.strip().split()[0]: int(line.strip().split()[1])for line in classification_format.readlines() if line.strip() !=""}

    classification_data=pd.read_csv(classification_file_name, index_col=0)
    classification_data=classification_data.drop('slide_id', axis=1)

    classification_types=list(set(classification_data['label'].values.tolist()))
    for class_type in classification_types:
        classification_data['label']=classification_data['label'].replace(class_type, classification_format_dic[class_type])

    classification_dic=classification_data.to_dict()['label']

    datasets_for_marker_search=training_dataset.filter(classification_dic.keys(), axis=1)

    data_T=datasets_for_marker_search.T
    y_list=[classification_dic[id] for id in data_T.index.tolist()]
    
    X_train, X_test, y_train, y_test = train_test_split(data_T, y_list, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def XGBoost_marker_search(TPM_path, classification_file_name, classification_format_file, include=[], CV_tag="0", num_markers=150):
    
    print("\n[ Marker search by XGBoost with training data started.. ]")
    TPM_df=pd.read_csv(TPM_path, index_col=0)
    TPM_df=TPM_df.dropna(axis=0)
    TPM_df_filtered=TPM_df.filter(include, axis=1)
    
    X_train, X_test, y_train, y_test=marker_search_data_processing_XGBoost(TPM_df_filtered, classification_file_name, classification_format_file)
    
    scaled_X_train, min_max_dic=min_max_norm_TPM(X_train.T)
    scaled_X_test, _=min_max_norm_TPM(X_test.T,min_max_dic=min_max_dic)
    
    gene_id_list=scaled_X_train.index.tolist()
    
    scaled_X_train=scaled_X_train.T
    scaled_X_test=scaled_X_test.T
    
    xgb_model = XGBClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
    xgb_model.fit(scaled_X_train, y_train)
    
    y_pred = xgb_model.predict(scaled_X_test)
    y_pred_proba = xgb_model.predict_proba(scaled_X_test)[:, 1] 
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    

    # Explain model predictions using SHAP
    explainer = shap.Explainer(xgb_model, scaled_X_train)
    shap_values = explainer(scaled_X_test)

    # Plot summary of SHAP values
    plt.figure()
    shap.summary_plot(shap_values, scaled_X_test, show=False)
    plt.savefig(f"{TPM_path.split('/')[-1][:-4]}_shap_XGBoost_summary_plot_{len(marker_list)}_{CV_tag}.png", dpi=300, bbox_inches="tight")
    plt.close

    importance = xgb_model.feature_importances_
    marker_list_pre=[]
    for idx, importance in enumerate(importance):
        if importance>0:
            marker_list_pre.append([importance,gene_id_list[idx]])
    marker_list_pre.sort()
    marker_list_pre.reverse()
    marker_list=[cont[1] for cont in marker_list_pre[:int(num_markers)]]
    TPM_df=pd.read_csv(TPM_path, index_col=0)
    marker_df=TPM_df.loc[marker_list]

    print(f"[ XGBoost training Accuracy: {accuracy:.3f} and AUROC: {roc_auc:.3f} | Number of identified marker genes: {len(marker_list)} ]\n")
    marker_df.to_csv(f"{TPM_path.split('/')[-1][:-4]}_markers_by_XGBoost_{len(marker_list)}.csv")
    return marker_df, min_max_dic

def marker_search_data_processing_LassoReg(training_dataset, PFI_file_name):
    pfi_df=pd.read_csv(PFI_file_name, index_col=0)
    pfi_dic=pfi_df.to_dict()['PFI']

    pfi_ids=pfi_dic.keys()
    
    datasets_for_marker_search=training_dataset.filter(pfi_ids, axis=1)#[training_dataset.mean(axis=1)>=1]
    
    data_T=datasets_for_marker_search.T

    y_list=[pfi_dic[id] for id in data_T.index.tolist()]
    
    X_train, X_test, y_train, y_test = train_test_split(data_T, y_list, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
    
def LassoReg_marker_search(TPM_path, PFI_file_name, exclude=[], CV_tag="0", num_markers=150):
    from sklearn.linear_model import Lasso
    # from sklearn.pipeline import make_pipeline
    from sklearn.metrics import mean_squared_error, r2_score

    print("\n[ Marker search by Lasso Regression with training data started.. ]")
    TPM_df=pd.read_csv(TPM_path, index_col=0)
    TPM_df=TPM_df.dropna(axis=0)
    TPM_df_filtered=TPM_df.drop(columns=exclude, errors='ignore')
    
    X_train, X_test, y_train, y_test=marker_search_data_processing_LassoReg(TPM_df_filtered, PFI_file_name)
    
    scaled_X_train, min_max_dic=min_max_norm_TPM(X_train.T)
    scaled_X_test, _=min_max_norm_TPM(X_test.T,min_max_dic=min_max_dic)
    
    gene_id_list=scaled_X_train.index.tolist()
    
    scaled_X_train=scaled_X_train.T
    scaled_X_test=scaled_X_test.T
    
    
    # Standardize the data (Lasso is sensitive to feature scaling)
    lasso = Lasso(alpha=0.1)  # Alpha controls regularization strength
    lasso.fit(scaled_X_train, y_train)
    
    # Predict on the test set
    y_pred = lasso.predict(scaled_X_test)
    
    # Evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    
    
    # lasso_model = lasso.named_steps['lasso']  # Extract the Lasso model from the pipeline
    
    marker_list = [gene_id_list[idx] for idx, coef in enumerate(lasso.coef_) if coef != 0]
    
    TPM_df=pd.read_csv(TPM_path, index_col=0)
    marker_df=TPM_df.loc[marker_list]
    
    print(f"[ Lasso Regression Mean Squared Error: {mse:.4f} | R-squared: {r2:.4f} | Number of identified marker genes: {len(marker_list)} ]\n")
    marker_df.to_csv(f"{TPM_path.split('/')[-1][:-4]}_markers_by_LassoReg_{len(marker_list)}.csv")
    
    
    explainer = shap.Explainer(lasso, scaled_X_train)
    shap_values = explainer(scaled_X_test)
    # Plot summary of SHAP values

    plt.figure()
    shap.summary_plot(shap_values, scaled_X_test, show=False)
    plt.savefig(f"{TPM_path.split('/')[-1][:-4]}_shap_LassoReg_summary_plot_{len(marker_list)}_{CV_tag}.png", dpi=300, bbox_inches="tight")
    plt.close

    
    min_max_dic={}
    return marker_df, min_max_dic



def parse_sample_classification(class_info_file, classification_format_file): #return {wsi_id:[patient_id, label]}
    class_info=pd.read_csv(class_info_file, index_col=0)
    nonredundant_ids=list(set(class_info.index.values))
    
    classification_file=open(classification_format_file,'r')
    classification_file_lines=classification_file.readlines()
    label_conversion_dict={}
    for line in classification_file_lines:
        line=line.strip().split()
        try:
            label_conversion_dict[line[0]]=int(line[1])
        except:
            print("""
            Error: classification format should contain label and number.
                  
            For example:
            Resistant   0
            Sensitive   1

            The provided classification format raised error in the number part.
            Check the format and run again.
            """)
            sys.exit()
    
    case_classification_dict={}
    for id in nonredundant_ids:
        case=class_info.loc[id].values
        if case.shape==(2,):
            classification=label_conversion_dict[case[1]]
            case_classification_dict[case[0]]=[id,classification] # id = wsi_id, classification = numerical classification
        else:
            classification_list=[]
            for each_case in case:
                classification=label_conversion_dict[each_case[1]]
                case_classification_dict[each_case[0]]=[id,classification] # id = wsi_id, classification = numerical classification

                classification_list.append(each_case[1])
            if len(set(classification_list))>1:
                sys.exit(f"{id} has two different classification. Please check classification information and try the pipline again.")
    return case_classification_dict



#______for multimodal___________
def parse_data_split_from_clam_for_multimodal(clam_training_dir, WSI_label_csv, label_format): # return: {splits_#.csv:{'train':list, 'validation': list, 'test':list},...}
    files=os.listdir(clam_training_dir)
    split_files=[file for file in files if file.startswith('splits_')]
    
    class_dict=parse_sample_classification(WSI_label_csv, label_format) # <--------- change file path
    split_dict={}
    for split_ in split_files:
        file_path=os.path.join(clam_training_dir,split_)
        split_info=pd.read_csv(file_path)

        #_train list__
        train=split_info.loc[:,'train'].dropna().values
        train_list=[]
        for wsi in train:
            case_class=class_dict[wsi]
            train_list.append([wsi]+case_class) # [wsi_id, case_id, label]
        #_validation list__
        validation=split_info.loc[:,'val'].dropna().values
        validation_list=[]
        for wsi in validation:
            case_class=class_dict[wsi]
            validation_list.append([wsi]+case_class) # [wsi_id, case_id, label]

        #_test list__
        test=split_info.loc[:,'test'].dropna().values
        test_list=[]
        for wsi in test:
            case_class=class_dict[wsi]
            test_list.append([wsi]+case_class) # [wsi_id, case_id, label]

        split_dict[split_[:-4]]={'train':train_list, 'validation':validation_list, 'test':test_list} # save data as a dictionary format.
    return split_dict
#___________________________________

def parse_data_split_from_clam_for_omics(clam_training_dir, WSI_label_csv, label_format): # return: {splits_#.csv:{'train':list, 'validation': list, 'test':list},...}
    files=os.listdir(clam_training_dir)
    split_files=[file for file in files if file.startswith('splits_')]
    
    class_dict=parse_sample_classification(WSI_label_csv, label_format) # <--------- change file path
    split_dict={}
    for split_ in split_files:
        file_path=os.path.join(clam_training_dir,split_)
        split_info=pd.read_csv(file_path)

        #_train list__
        train=split_info.loc[:,'train'].dropna().values
        filter=[]
        train_list=[]
        for wsi in train:
            case_class=class_dict[wsi]
            if case_class[0] not in filter:
                train_list.append(case_class) # [case_id, label]
                filter.append(case_class[0])
        #_validation list__
        validation=split_info.loc[:,'val'].dropna().values
        filter=[]
        validation_list=[]
        for wsi in validation:
            case_class=class_dict[wsi]
            if case_class[0] not in filter:
                validation_list.append(case_class) # [case_id, label]
                filter.append(case_class[0])
        #_test list__
        test=split_info.loc[:,'test'].dropna().values
        filter=[]
        test_list=[]
        for wsi in test:
            case_class=class_dict[wsi]
            if case_class[0] not in filter:
                test_list.append(case_class) # [case_id, label]
                filter.append(case_class[0])

        split_dict[split_[:-4]]={'train':train_list, 'validation':validation_list, 'test':test_list} # save data as a dictionary format.
    return split_dict


def data_loader(tpm_data, case_target_list, min_max_dic={}):
    scaled_tpm_data, min_max_dic=min_max_norm_TPM(tpm_data, min_max_dic=min_max_dic)
    tpm_data=scaled_tpm_data.T
    sample_ids=[]
    label_dic={}
    for data in case_target_list:
        label_dic[data[0]]=data[1]
        sample_ids.append(data[0])
    sample_tpms=tpm_data.filter(sample_ids, axis=0)
    sample_tpms=sample_tpms.reindex(index=sample_ids).dropna()
    
    targets=[] 

    for sample_id in sample_tpms.index.values:
        targets.append(label_dic[sample_id])

    targets=np.array(targets)

    
    return sample_tpms.values, targets, min_max_dic # export data and label as numpy format.

def feature_from_h5(h5_path): # Read features of a WSI

    with h5py.File(h5_path,'r') as hdf5_file:
        features = hdf5_file['features'][:]
        # coords = hdf5_file['coords'][:]

    features = torch.from_numpy(features)
    return features



def get_feature_from_wsi_model(wsi_parameter_path, wsi_feature, cuda_vis_dev):

    device=f"cuda:{cuda_vis_dev}" if torch.cuda.is_available() else "CPU"
    
    model_dict = {"dropout": 0.25, 'n_classes': 2, "embed_dim": 1024}
    wsi_model=CLAM_SB(**model_dict).to(device)
    wsi_model.eval()

    ckpt = torch.load(wsi_parameter_path, weights_only=True, map_location="cuda")
    
    ckpt_clean = {}
    for key in ckpt.keys():
        if 'instance_loss_fn' in key:
            continue
        ckpt_clean.update({key.replace('.module', ''):ckpt[key]})
    wsi_model.load_state_dict(ckpt_clean, strict=True)        

    with torch.no_grad():
        logits, Y_prob, Y_hat, A_raw, results_dict = wsi_model(wsi_feature.to(device), instance_eval=False, return_features=True)
    
    return results_dict['features'].squeeze().cpu().numpy()


def get_feature_from_omics_model(omics_parameter_path, sample_tpm, tdim, cuda_vis_dev):

    device=f"cuda:{cuda_vis_dev}" if torch.cuda.is_available() else "CPU"

    transcriptome_model=TranscriptomeModel(tdim).to(device)
    transcriptome_model.eval()

    
    state_dict = torch.load(omics_parameter_path, weights_only=True, map_location="cuda")

    model_state_dict = {k: v for k, v in state_dict.items() if k in transcriptome_model.state_dict()} 
    norm_min_max_dic = {k: v for k, v in state_dict.items() if k not in transcriptome_model.state_dict()}
    
    remove_prefix = 'module.'
    state_dict_filtered = {k[len(remove_prefix):] if k.startswith(remove_prefix) else k: v for k, v in model_state_dict.items()}

    transcriptome_model.load_state_dict(state_dict_filtered)
    
    with torch.no_grad():
        logits, Y_prob, Y_hat, feature=transcriptome_model(sample_tpm.to(device))
    
    return feature.cpu().numpy()


def data_loader_for_multimodal(tpm_data, multimodal_target_list, feature_h5_folder, wsi_parameter_path, omics_parameter_path, cuda_vis_dev, min_max_dic={}): # return (wsi_features, tpm, label)              
    scaled_tpm_data, min_max_dic=min_max_norm_TPM(tpm_data, min_max_dic=min_max_dic)
    tpm_data=scaled_tpm_data#.T
    tdim=tpm_data.shape[0]
    
    tpm_data_dic=tpm_data.to_dict('list')
    wsi_features=[]
    sample_tpms=[]
    targets=[] 
    count=0

    for data in multimodal_target_list: # multimodal_target_list = [wsi_id, case_id, label(number)]

        h5_file_path=os.path.join(feature_h5_folder,f'{data[0]}.h5')
        if data[1] in tpm_data_dic:
            sample_tpm=torch.FloatTensor(np.array(tpm_data_dic[data[1]]))

            tpm_feature=get_feature_from_omics_model(omics_parameter_path, sample_tpm, tdim, cuda_vis_dev)
            sample_tpms.append(torch.tensor(tpm_feature))

            wsi_raw_feature=feature_from_h5(h5_file_path)
            wsi_feature=get_feature_from_wsi_model(wsi_parameter_path, wsi_raw_feature, cuda_vis_dev)
            wsi_features.append(torch.tensor(wsi_feature))
            
            targets.append(data[2])
        else:
            pass
        
    targets=np.array(targets)
    

    return np.array(wsi_features), np.array(sample_tpms), targets, min_max_dic # export data and label as numpy format.




class EarlyStopping(object):
    def __init__(self, patience=2, save_path="model.pth", norm_min_max_dic="", cuda_vis_dev="0"):
        self._min_loss = np.inf
        self._patience = patience
        self._path = save_path
        self.__counter = 0
        self._norm_min_max_dic=norm_min_max_dic
        self._cuda_vis_dev=cuda_vis_dev
 
    def should_stop(self, model, loss):
        if loss < self._min_loss:
            self._min_loss = loss
            self.__counter = 0

            state_dict=model.state_dict()
            state_dict.update(self._norm_min_max_dic)
            
            torch.save(state_dict, self._path)
        elif loss > self._min_loss:
            self.__counter += 1
            if self.__counter >= self._patience:
                return True
        return False

    def elbow_stop(self, model):
        state_dict=model.state_dict()
        state_dict.update(self._norm_min_max_dic)
        torch.save(state_dict, self._path)
   
    def load(self, model):

        device=torch.device(f"cuda:{self._cuda_vis_dev}") if torch.cuda.is_available() else "CPU"

        state_dict=torch.load(self._path, weights_only=True, map_location="cuda")

        model_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict()} 
        
        norm_min_max_dic = {k: v for k, v in state_dict.items() if k not in model.state_dict()}

        model.load_state_dict(model_state_dict)
        
        return model.to(device), norm_min_max_dic
    
    @property
    def counter(self):
        return self.__counter
    
def Omics_Inference(model,data,target, cuda_vis_dev):

    device=f"cuda:{cuda_vis_dev}" if torch.cuda.is_available() else "CPU"
    
    with torch.no_grad():  
        model.eval()

        x,y=torch.from_numpy(data).float().to(device), torch.from_numpy(target).float().to(device)
        
        logits, Y_prob, Y_hat, feature = model(x)   #<====== for multiple classification
        loss=nn.CrossEntropyLoss()(logits, y.long()).item()  #<====== for multiple classification
        return Y_prob, loss

def Multimodal_Inference(model,wsi_feature, omics_feature,target, cuda_vis_dev):

    device=f"cuda:{cuda_vis_dev}" if torch.cuda.is_available() else "CPU"
    
    with torch.no_grad():  
        model.eval()

        x1, x2, y=torch.from_numpy(wsi_feature).float().to(device), torch.from_numpy(omics_feature).float().to(device),torch.from_numpy(target).float().to(device)
        logits, Y_prob, Y_hat, feature = model(x1, x2)
        loss=nn.CrossEntropyLoss()(logits, y.long()).item()
        return Y_prob, loss

def loss_graph(train_loss_list, val_loss_list, test_loss_list, loader, early_stop_epoch, cv_set_number,path):
    plt.figure()
    plt.plot(train_loss_list, label="Train Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.plot(test_loss_list, label="Test Loss")
    plt.axvline(early_stop_epoch, color="grey", linestyle="--", label="Early Stop Epoch")
    
    plt.xlabel("Epoch")
    plt.legend()
    plt.show()   
    if path[-1]=="/":
        path=path[:-1]
    if loader=="transcriptome":
        plt.savefig(f"{path}/Transcriptome_loss_fig_CV-{cv_set_number}")
    elif loader=="multimodal":
        plt.savefig(f"{path}/Multimodal_loss_fig_CV-{cv_set_number}")


