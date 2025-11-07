import pandas as pd
import torch 
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import GPUtil
from sklearn.model_selection import cross_val_score 
import sys
import os
import gc

import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from tqdm import tqdm
# from kneed import KneeLocator
import copy
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from Multimodal_utils import EarlyStopping, parse_data_split_from_clam_for_omics, data_loader, data_loader_for_multimodal, parse_data_split_from_clam_for_multimodal, min_max_norm_TPM, XGBoost_marker_search, LassoReg_marker_search ,find_TPM_marker_gene
from Multimodal_utils import Omics_Inference, Multimodal_Inference, loss_graph

import warnings
import csv



warnings.filterwarnings("ignore")


def Accuracy(output, target):
    
    softmax = nn.Softmax(dim=1)

    prediction_argmax=softmax(output.cpu()).argmax(dim=1)
    target=target.cpu()
    

    accuracy=accuracy_score(target, prediction_argmax)
    f1=f1_score(target.numpy(), prediction_argmax.numpy())

    acc_each_class={}
    prediction_argmax_list=prediction_argmax.tolist()
    target_list=target.tolist()

    # To calculate accuracy statistics in each class
    for i in range(len(target_list)):
        prediction_value=prediction_argmax_list[i]
        target_value=target_list[i]
        if prediction_value==target_value:
            match=1
        else:
            match=0
            
        if target_value not in acc_each_class:
            acc_each_class[target_value]=[match,1]
        else:
            acc_each_class[target_value][0] += match
            acc_each_class[target_value][1] += 1
    
    acc_each_class_out={}    
    for key, value in acc_each_class.items():
        acc=value[0]/value[1]
        acc_each_class_out[key]=[acc, value[1]]
    #_____________________________________________
        
    return accuracy, acc_each_class_out, float(f1) # acc_each_class={1: [accuracy, sample_number], 2: [accuracy, sample_number],...}

def find_elbow(x, y):
    # Convert x and y to numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Get the line connecting the first and last points
    line_start = np.array([x[0], y[0]])
    line_end = np.array([x[-1], y[-1]])
    
    # Calculate distances from each point to the line
    # First, derive the line vector and its unit vector
    line_vec = line_end - line_start
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    # Compute the vector from the first point to each point
    vec_from_start = np.vstack([x - line_start[0], y - line_start[1]]).T
    
    # Project each point onto the line
    proj_lengths = np.dot(vec_from_start, line_vec_norm)
    proj_points = np.outer(proj_lengths, line_vec_norm) + line_start
    
    # Compute distances from each point to its projection on the line (perpendicular distance)
    distances = np.linalg.norm(vec_from_start - (proj_points - line_start), axis=1)
    
    # The elbow point is where the distance is maximized
    elbow_index = np.argmax(distances)
    return x[elbow_index]
    

def training(data_split_info, WSI_label_csv, label_format, TPM_data_path, loader, learning_rate, batch_size=32, max_epoch=30000, rand_seed=3000, tag="results", loss_path="", WSI_h5_feature_path="", WSI_chkpoint_path="", multimodal_type="MLP", multimodal_embed_dim=256, cuda_vis_dev="0", clinical_data="", num_markers="500", marker_search="2", log2="1", scaling="1", filtering="1", mean_threshold="1", early_stopping="0", elbow_prop=0.5):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(rand_seed) 
    torch.manual_seed(rand_seed)


    device=f"cuda:{cuda_vis_dev}" if torch.cuda.is_available() else "cpu"

    if device==f'cuda:{cuda_vis_dev}':
        gc.collect()
    
        torch.cuda.empty_cache()
        GPUtil.showUtilization()
    print(f'\n{device} is used.\n\n')

    
    if early_stopping !="0":
        max_epoch=int(early_stopping)
        
    # Read split information ____________________________
    if loader=="transcriptome":    
        from multimodal_models.Transcriptome_model import TranscriptomeModel

        split_dic=parse_data_split_from_clam_for_omics(data_split_info, WSI_label_csv, label_format)

        split_dic_for_marker=split_dic

    elif loader=="multimodal":
        from multimodal_models.Multimodal_model import MultimodalModel_MLP

        split_dic=parse_data_split_from_clam_for_multimodal(data_split_info, WSI_label_csv, label_format)

        split_dic_for_marker=parse_data_split_from_clam_for_omics(data_split_info, WSI_label_csv, label_format)


    else:
        print("\nError: choose loader between 'transcriptome' and 'multimodal' and try again!\n")
        sys.exit()

    #___________________
    
    cross_validation_set=[*split_dic.keys()]
    cross_validation_set.sort()

    acc_report_list=[]

    acc_raw_each_class_dict={}
    val_acc_raw_each_class_dict={}

    init=0
    for cv_set in cross_validation_set:
        train_data_info=split_dic[cv_set]['train']
        validation_data_info=split_dic[cv_set]['validation']

        if early_stopping !="0":
            train_data_info=train_data_info+validation_data_info
            
        test_data_info=split_dic[cv_set]['test']

        
        
        if early_stopping !="0":
            exclude_list=[id[0] for id in split_dic_for_marker[cv_set]['test']]
        else:
            exclude_list=[id[0] for id in split_dic_for_marker[cv_set]['validation']+split_dic_for_marker[cv_set]['test']]
        
        include_list=[id[0] for id in split_dic_for_marker[cv_set]['train']]
        cv_set_number=cv_set.split('_')[-1]


        model_parameter_folder=TPM_data_path[:-4]+f"_{tag}"
        
        if not os.path.isdir(model_parameter_folder):
            os.makedirs(model_parameter_folder)
        
        if marker_search=="2":
            
            tpm_data, norm_min_max_dic=find_TPM_marker_gene(TPM_data_path, clinical_data, num_markers, exclude_list, cv_set, log2=log2, scaling=scaling, filtering=filtering, mean_threshold=mean_threshold, marker_folder=model_parameter_folder) # <== search marker genes 
        # elif marker_search=="3":
    
        #     tpm_data, norm_min_max_dic=XGBoost_marker_search(TPM_data_path, WSI_label_csv, label_format, include=include_list, CV_tag=cv_set, num_markers=num_markers)
        # elif marker_search=="4":
        #     tpm_data, norm_min_max_dic=LassoReg_marker_search(TPM_data_path, clinical_data, exclude=exclude_list, CV_tag=cv_set, num_markers=num_markers)
        else:
            tpm_data, norm_min_max_dic=min_max_norm_TPM(TPM_data_path, num_markers=int(num_markers), log2=log2, scaling=scaling, filtering=filtering, mean_threshold=mean_threshold)
    
        if loader=="transcriptome":
                
            # early_stopper=EarlyStopping(patience=2000, save_path=f"./z_multimodal_chk_{tag}/transcriptome_model_{cv_set}.pth", norm_min_max_dic=norm_min_max_dic, cuda_vis_dev=cuda_vis_dev)
            model_save_path=f"./{model_parameter_folder}/transcriptome_model_{cv_set}.pth"
            early_stopper=EarlyStopping(patience=1000, save_path=model_save_path, norm_min_max_dic=norm_min_max_dic, cuda_vis_dev=cuda_vis_dev)

            train_data, train_target, min_max_dic=data_loader(tpm_data,train_data_info)
            validation_data, validation_target, _=data_loader(tpm_data,validation_data_info, min_max_dic)
            test_data, test_target, _=data_loader(tpm_data,test_data_info, min_max_dic)

            
            tdim=train_data.shape[-1]
            
            model=TranscriptomeModel(tdim).to(device)
            # model=nn.DataParallel(model) #<=========================
            optimizer=torch.optim.NAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))#, eps=1e-8, weight_decay=4e-5)
            # optimizer=torch.optim.SGD(model.parameters(), lr=learning_rate)

            if init==0:
                init=1
                print("\n\n*********************************")
                print("*** Train transcriptome model ***")
                print("*********************************\n\n")
                print("Random Seed = "+str(rand_seed)+"\n")
                print(TranscriptomeModel(tdim),"\n\n")
                # with open(f"./{model_parameter_folder}/z_multimodal_training_reports_{tag}.txt",'a') as train_report:

                train_report= open(f"./{model_parameter_folder}/z_WSIomics_training_reports_{tag}.txt",'a')
                train_report.write(str(rand_seed)+"\n")
                train_report.write(str(TranscriptomeModel(tdim)))

        
        elif loader=="multimodal":
            # model_parameter_folder=TPM_data_path[:-4]+f"_{tag}"
            model_save_path=f"./{model_parameter_folder}/multimodal_model_{cv_set}.pth"

            early_stopper=EarlyStopping(patience=1000, save_path=model_save_path, norm_min_max_dic=norm_min_max_dic, cuda_vis_dev=cuda_vis_dev)
            
            # early_stopper=EarlyStopping(patience=2000, save_path=f"./z_multimodal_chk_{tag}/multimodal_model_{cv_set}.pth", norm_min_max_dic=norm_min_max_dic, cuda_vis_dev=cuda_vis_dev)

            train_wsi_feature, train_tpm_feature, train_target, min_max_dic=data_loader_for_multimodal(
                tpm_data,
                train_data_info,
                WSI_h5_feature_path, 
                f"{WSI_chkpoint_path}/s_{cv_set_number}_checkpoint.pt", 
                f"./{model_parameter_folder}/transcriptome_model_splits_{cv_set_number}.pth", 
                cuda_vis_dev
                )

            validation_wsi_feature, validation_tpm_feature, validation_target, _=data_loader_for_multimodal(
                tpm_data,
                validation_data_info,
                WSI_h5_feature_path, 
                f"{WSI_chkpoint_path}/s_{cv_set_number}_checkpoint.pt", 
                f"./{model_parameter_folder}/transcriptome_model_splits_{cv_set_number}.pth",
                cuda_vis_dev,
                min_max_dic=min_max_dic
                )
            
            test_wsi_feature, test_tpm_feature, test_target, _=data_loader_for_multimodal(
                tpm_data,
                test_data_info,
                WSI_h5_feature_path, 
                f"{WSI_chkpoint_path}/s_{cv_set_number}_checkpoint.pt", 
                f"./{model_parameter_folder}/transcriptome_model_splits_{cv_set_number}.pth", 
                cuda_vis_dev,
                min_max_dic=min_max_dic
                )

            tdim=train_tpm_feature.shape[-1]
            WSI_dim=train_wsi_feature.shape[1]
            transcriptome_dim=train_tpm_feature.shape[1]

            if multimodal_type=="Attn":
                model=MultimodalModel_Attn(WSI_dim=512, transcriptome_dim=tdim, n_classes=2, embed_dim=multimodal_embed_dim).to(device)
                optimizer=torch.optim.NAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))#, eps=1e-8, weight_decay=4e-5)
                
                
                if init==0:
                    init=1
                    
                    print("\n\n******************************")
                    print("*** Train multimodal model ***")
                    print("******************************\n\n")        
                    print(MultimodalModel_Attn(WSI_dim=512, transcriptome_dim=tdim),"\n\n")
                    # with open(f"./{model_parameter_folder}/z_WSIomics_training_reports_{tag}.txt",'a') as train_report:

                    train_report=open(f"./{model_parameter_folder}/z_WSIomics_training_reports_{tag}.txt",'a')
                    train_report.write(str(MultimodalModel_Attn(WSI_dim=512, transcriptome_dim=tdim, n_classes=2)))

            elif multimodal_type=="MLP":
                model=MultimodalModel_MLP(WSI_dim=512, transcriptome_dim=tdim, n_classes=2).to(device)

                optimizer=torch.optim.NAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))#, eps=1e-8, weight_decay=4e-5)
                    
                
                if init==0:
                    init=1
                    
                    print("\n\n******************************")
                    print("*** Train multimodal model ***")
                    print("******************************\n\n")        
                    print(MultimodalModel_MLP(),"\n\n")
                    # with open(f"./{model_parameter_folder}/z_WSIomics_training_reports_{tag}.txt",'a') as train_report:
                    
                    train_report=open(f"./{model_parameter_folder}/z_WSIomics_training_reports_{tag}.txt",'a')
                    train_report.write(str(MultimodalModel_MLP(WSI_dim=512, transcriptome_dim=tdim, n_classes=2)))

        else:
            print("\nError: choose loader between 'transcriptome' and 'multimodal' and try again!\n")
            sys.exit()
        
        epoc_list=[]
        loss_list=[]
        val_loss_list=[]
        elbow_epoch_list=[[0,0]]

        epoc_to_elbow_dic={}
        elbow_to_state_dic={}
        
        # min_valid_loss=np.inf

        elbow_stop=False
        early_stop=False

        val_acc=[]
        val_f1=[]
        val_auc=[]
        val_loss=[]
        train_loss_list=[]
        test_loss_list=[]

        val_acc_final=0
        val_f1_final=0
        val_auc_final=0
        val_loss_final=0
        

        valid_loss=0.0
        train_loss_epoch=0.0
        for epoch in tqdm(range(max_epoch), desc=f"Training for {cv_set}", ncols=150):
            
            train_loss=0.0 
            model.train()

            if batch_size >= len(train_target):
                batch_size=len(train_target)
            iter=round(len(train_target)/batch_size)
            for k in range(iter): #<----------
                start=k*batch_size #<----------
                end=(k+1)*batch_size #<----------
                
                optimizer.zero_grad()
                if loader=="transcriptome":
                    x, y=torch.from_numpy(train_data[start:end]).float().to(device), torch.from_numpy(train_target[start:end]).float().to(device)
                    
                    logits, Y_prob, Y_hat, feature=model(x)
                    # prediction,_=model(x)

                elif loader=="multimodal":
                    x1, x2, y=torch.from_numpy(train_wsi_feature[start:end]).float().to(device), torch.from_numpy(train_tpm_feature[start:end]).float().to(device), torch.from_numpy(train_target[start:end]).float().to(device)
                    logits, Y_prob, Y_hat, feature = model(x1, x2)

                else:
                    print("\nError: choose loader between 'transcriptome' and 'multimodal' and try again!\n")
                    sys.exit()
                loss=nn.CrossEntropyLoss()(logits, y.long())
                
                # loss=F.binary_cross_entropy(Y_prob, y)
                
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            train_loss=train_loss/iter #<----------
            
            
            batch_accuracy=0

            with torch.no_grad():  # calculate Loss value of validation loop
                model.eval()
                
                if loader=="transcriptome":
                    x,y=torch.from_numpy(validation_data).float().to(device), torch.from_numpy(validation_target).float().to(device)
                    logits, Y_prob, Y_hat, feature=model(x)
                    
                    _, test_loss=Omics_Inference(model, test_data, test_target, cuda_vis_dev)
                    test_loss_list.append(test_loss)

                elif loader=="multimodal":
                    x1, x2, y=torch.from_numpy(validation_wsi_feature).float().to(device), torch.from_numpy(validation_tpm_feature).float().to(device),torch.from_numpy(validation_target).float().to(device)
                    logits, Y_prob, Y_hat, feature = model(x1, x2)

                    _, test_loss=Multimodal_Inference(model, test_wsi_feature, test_tpm_feature, test_target, cuda_vis_dev)
                    test_loss_list.append(test_loss)

                else:
                    print("\nError: choose loader between 'transcriptome' and 'multimodal' and try again!\n")
                    sys.exit()
                v_loss=nn.CrossEntropyLoss()(logits, y.long()).item()
                accuracy, validation_acc_each_class, f1=Accuracy(Y_prob,y)
                
                # v_loss=F.binary_cross_entropy(Y_prob, y)
                # accuracy=(Y_prob>0.5).float().eq(y).float().mean().item()

                try:
                    # r_a_score=roc_auc_score(y.cpu(), Y_hat.cpu())
                    r_a_score=roc_auc_score(y.cpu(), Y_prob.cpu()[:,1]) # <=========== modified
                except:
                    r_a_score=0

                val_acc.append(accuracy)
                val_f1.append(f1)
                val_auc.append(r_a_score)
                val_loss.append(v_loss)
                train_loss_list.append(train_loss)
                

                
                valid_loss += v_loss
                train_loss_epoch += train_loss

                if early_stopping =="0":
                    if epoch > 100 and early_stopper.should_stop(model, v_loss):
                        early_stop=True
                        early_stop_epoch=epoch -early_stopper.counter
    
                        val_acc_final=val_acc[-early_stopper.counter-1]
                        val_f1_final=val_f1[-early_stopper.counter-1]
                        val_auc_final=val_auc[-early_stopper.counter-1]
                        val_loss_final=val_loss[-early_stopper.counter-1]
                        train_loss_final=train_loss_list[-early_stopper.counter-1]
                        
                        break
                    else:
                        val_acc_final=accuracy
                        val_f1_final=f1
                        val_auc_final=r_a_score
                        val_loss_final=v_loss
                        train_loss_final=train_loss
                else:
                    if epoch ==int(early_stopping)-1:

                        max_elbow_epoch=max(elbow_epoch_list)[0]
                        state_epoch=int(max_elbow_epoch* elbow_prop)
                        
                        for k in range(1000):
                            state_epoch_search=state_epoch+k
                            if state_epoch_search in elbow_to_state_dic:
                                
                                elbow_state_dict=elbow_to_state_dic[state_epoch_search]
                                break
                            
                        
                        # state_elbow=epoc_to_elbow_dic[state_epoc]
                        # state_dict=elbow_to_state_dic[state_elbow]
                        
                        torch.save(elbow_state_dict, model_save_path)

                        early_stop_epoch=state_epoch_search
    
                        val_acc_final=val_acc[state_epoch_search-1]
                        val_f1_final=val_f1[state_epoch_search-1]
                        val_auc_final=val_auc[state_epoch_search-1]
                        val_loss_final=val_loss[state_epoch_search-1]
                        train_loss_final=train_loss_list[state_epoch_search-1]                              
                        break
                        # else:
                            
                        #     break
                    # else:
                    #     val_acc_final=accuracy
                    #     val_f1_final=f1
                    #     val_auc_final=r_a_score
                    #     val_loss_final=v_loss
                    #     train_loss_final=train_loss
                        
            
            if epoch !=0:
                epoc_list.append(epoch)
                loss_list.append(train_loss)
                elbow_epoch=find_elbow(np.array(epoc_list), np.array(loss_list)).item()
                # kn=KneeLocator(np.array(epoc_list), np.array(loss_list))
                # if kn.knee.item() >1:
                

                if elbow_epoch>max(elbow_epoch_list)[0]:
                    elbow_stop=True
                    early_stop=True


                    state_dict_to_save=model.state_dict()
                    state_dict_to_save.update(norm_min_max_dic)
                    
                    if elbow_epoch not in elbow_to_state_dic:
                        elbow_to_state_dic[elbow_epoch]=copy.deepcopy(state_dict_to_save)

                    
                    # torch.save(state_dict, self._path)

                    
                    
                    # early_stop_epoch=elbow_epoc

                    # val_acc_final=val_acc[-1]
                    # val_f1_final=val_f1[-1]
                    # val_auc_final=val_auc[-1]
                    # val_loss_final=val_loss[-1]
                    # train_loss_final=train_loss_list[-1]     

                    # early_stopper.elbow_stop(model)

                epoc_to_elbow_dic[epoch]=elbow_epoch
                
                elbow_epoch_list.append([elbow_epoch, epoch])

                
                val_loss_list.append(v_loss)
            # early_stop

            if early_stop==False:
                early_stop_epoch=epoch
                
                state_dict=model.state_dict()
                state_dict.update(norm_min_max_dic)
                
                if loader=="transcriptome":
                    torch.save(state_dict, f"./{model_parameter_folder}/transcriptome_model_{cv_set}.pth")
                elif loader=="multimodal":
                    torch.save(state_dict, f"./{model_parameter_folder}/multimodal_model_{cv_set}.pth")
        # End of the EPOC loop ___________________________________________________________________________________

        

        
        # test evaluation__________________________________
        with torch.no_grad():  # calculate Loss value of validation loop
            model_test, norm_min_max_dic=early_stopper.load(model)
            model_test.eval()

            if loader=="transcriptome":
                x,y=torch.from_numpy(test_data).float().to(device), torch.from_numpy(test_target).float().to(device)
                logits, Y_prob, Y_hat, feature =model_test(x)
            elif loader=="multimodal":
                x1, x2, y=torch.from_numpy(test_wsi_feature).float().to(device), torch.from_numpy(test_tpm_feature).float().to(device),torch.from_numpy(test_target).float().to(device)
                logits, Y_prob, Y_hat, feature = model_test(x1, x2)
            else:
                print("\nError: Choose loader 'transcriptome' or 'multimodal' and try again!\n")
                sys.exit()

            v_loss=nn.CrossEntropyLoss()(logits, y.long())
            test_accuracy, test_acc_each_class, test_f1=Accuracy(Y_prob, y)  #(prediction>0.5).float().eq(y).float().mean().item()
            
            try:
                # test_auc=roc_auc_score(y.cpu(), Y_hat.cpu())
                print
                test_auc=roc_auc_score(y.cpu(), Y_prob.cpu()[:,1]) # <========== modified
            except:
                test_auc=0
        if early_stopping =="0":
            acc_report_list.append((cv_set, val_acc_final, val_f1_final, val_auc_final, test_accuracy, test_f1, test_auc)) #test, train,val ACC, val auc, test  accuracy, test auc
        else:
            acc_report_list.append((cv_set, test_accuracy, test_f1, test_auc)) #test, train,val ACC, val auc, test  accuracy, test auc


        train_report.write(f"\n\n>>>>>>>>>>  {cv_set}  <<<<<<<<<<\n")
        try:
            if early_stopping =="0":
                train_report.write(f"\n| No. training samples: {len(train_target)} | No. validation samples: {len(validation_target)} | No. test samples: {len(test_target)} |\n")
                print(f"\n| No. training samples: {len(train_target)} | No. validation samples: {len(validation_target)} | No. test samples: {len(test_target)} |\n")
            else:
                train_report.write(f"\n| No. training samples: {len(train_target)} | No. test samples: {len(test_target)} |\n")
                print(f"\n| No. training samples: {len(train_target)} | No. test samples: {len(test_target)} |\n")
        except:
            pass
        if early_stopping =="0":
            train_report.write(f"\nEarlyStopping: [Epoch: {epoch -early_stopper.counter}]")
            train_report.write(f"|  Validation Accuracy: {val_acc_final:.3f}   Test accuracy: {test_accuracy:.3f}  |  Validation F1 score: {val_f1_final:.3f}   Test F1 score: {test_f1:.3f}  |  Validation AUC score: {val_auc_final:.3f}   Test AUC score: {test_auc:.3f} |\n")
            
            print(f"\nEarlyStopping: [Epoch: {epoch -early_stopper.counter}]")
            print(f"|  Validation Accuracy: {val_acc_final:.3f}   Test accuracy: {test_accuracy:.3f}  |  Validation F1 score: {val_f1_final:.3f}   Test F1 score: {test_f1:.3f}  |  Validation AUC score: {val_auc_final:.3f}   Test AUC score: {test_auc:.3f} |\n")
        else:
            train_report.write(f"Elbow stop epoc: {state_epoch_search}\n")
            train_report.write(f"|  Test accuracy: {test_accuracy:.3f}  |  Test F1 score: {test_f1:.3f}  |  Test AUC score: {test_auc:.3f} |\n")
            print(f"Elbow stop epoc: {state_epoch_search}\n")
            print(f"|  Test accuracy: {test_accuracy:.3f}  |  Test F1 score: {test_f1:.3f}  |  Test AUC score: {test_auc:.3f} |\n")

        # generate dic data to calculate statistics of accuracy for each classe
        
        for key in range(len(test_acc_each_class)):
            try:
                if early_stopping =="0":
                    train_report.write(f"Validation Accuracy for Class_{key}: {validation_acc_each_class[key][0]:.3f}  Count for Class_{key}: {validation_acc_each_class[key][1]}  |  Test Accuracy for Class_{key}: {test_acc_each_class[key][0]:.3f}  Count for Class_{key}: {test_acc_each_class[key][1]}\n")
                    print(f"Validation Accuracy for Class_{key}: {validation_acc_each_class[key][0]:.3f}  Count for Class_{key}: {validation_acc_each_class[key][1]}  |  Test Accuracy for Class_{key}: {test_acc_each_class[key][0]:.3f}  Count for Class_{key}: {test_acc_each_class[key][1]}")
                else:
                    train_report.write(f"Test Accuracy for Class_{key}: {test_acc_each_class[key][0]:.3f}  Count for Class_{key}: {test_acc_each_class[key][1]}\n")
                    print(f"Test Accuracy for Class_{key}: {test_acc_each_class[key][0]:.3f}  Count for Class_{key}: {test_acc_each_class[key][1]}")              
            except:
                print("-Warning-: Error occurred in accuracy or AUC calculation.")
            
            if key not in acc_raw_each_class_dict:
                try:
                    acc_raw_each_class_dict[key]=[test_acc_each_class[key][0]]
                except:
                    acc_raw_each_class_dict[key]=[0]
            else:
                try:
                    acc_raw_each_class_dict[key].append(test_acc_each_class[key][0])
                except:
                    acc_raw_each_class_dict[key].append(0)
            
            if key not in val_acc_raw_each_class_dict:
                try:
                    val_acc_raw_each_class_dict[key]=[validation_acc_each_class[key][0]]
                except:
                    val_acc_raw_each_class_dict[key]=[0]
            else:
                try:
                    val_acc_raw_each_class_dict[key].append(validation_acc_each_class[key][0])                
                except:
                    val_acc_raw_each_class_dict[key].append(0)                
        print("\n")
        #________________________
        



        # Draw Loss Graph..
        if early_stopping !="0":
            loss_graph(train_loss_list, [], test_loss_list, loader, early_stop_epoch, cv_set_number, loss_path)
        else:
            loss_graph(train_loss_list, val_loss_list, test_loss_list, loader, early_stop_epoch, cv_set_number, loss_path)

    if early_stopping =="0":
        acc_report_columns=["Split","Validation Accuracy", "Validation F1 Score", "Validation AUC", "Test Accuracy", "Test F1 Score", "Test AUC"]
    else:
        acc_report_columns=["Split", "Test Accuracy", "Test F1 Score", "Test AUC"]
        
    report_pd=pd.DataFrame(acc_report_list, columns=acc_report_columns)

    
    if early_stopping =="0":
        val_acc_mean=report_pd["Validation Accuracy"].mean()
        val_acc_std=report_pd["Validation Accuracy"].std()
    
        val_f1_mean=report_pd["Validation F1 Score"].mean()
        val_f1_std=report_pd["Validation F1 Score"].std()    
    
        val_auc_mean=report_pd["Validation AUC"].mean()
        val_auc_std=report_pd["Validation AUC"].std()
        

        test_acc_mean=report_pd["Test Accuracy"].mean()
        test_acc_std=report_pd["Test Accuracy"].std()
    
        test_f1_mean=report_pd["Test F1 Score"].mean()
        test_f1_std=report_pd["Test F1 Score"].std()    
        
        test_auc_mean=report_pd["Test AUC"].mean()
        test_auc_std=report_pd["Test AUC"].std()
    
        acc_report_list.append(("Average", val_acc_mean, val_f1_mean, val_auc_mean, test_acc_mean, test_f1_mean, test_auc_mean)) #test, train,val ACC, val auc, test  accuracy, test auc
        acc_report_list.append(("Stdev", val_acc_std, val_f1_std, val_auc_std, test_acc_std, test_f1_std, test_auc_std)) #test, train,val ACC, val auc, test  accuracy, test auc
        
    else:

        test_acc_mean=report_pd["Test Accuracy"].mean()
        test_acc_std=report_pd["Test Accuracy"].std()
    
        test_f1_mean=report_pd["Test F1 Score"].mean()
        test_f1_std=report_pd["Test F1 Score"].std()    
        
        test_auc_mean=report_pd["Test AUC"].mean()
        test_auc_std=report_pd["Test AUC"].std()

        acc_report_list.append(("Average", test_acc_mean, test_f1_mean, test_auc_mean)) #test, train,val ACC, val auc, test  accuracy, test auc
        acc_report_list.append(("Stdev", test_acc_std, test_f1_std, test_auc_std)) #test, train,val ACC, val auc, test  accuracy, test auc

    
        acc_report_list.append((cv_set, test_accuracy, test_f1, test_auc)) #test, train,val ACC, val auc, test  accuracy, test auc



    
    if loader=="transcriptome":
        report_pd.to_csv(f"./{model_parameter_folder}/zz_Transcriptome_model_report_{tag}.csv", index=False)
    elif loader=="multimodal":
        report_pd.to_csv(f"./{model_parameter_folder}/zzz_Multimodal_model_report_{tag}.csv", index=False)


    
    if early_stopping =="0":
        print("*** Cross Validation Statistics ***")
        print(f"\nMean Val  Accuracy: {val_acc_mean:.3f}  std: {val_acc_std:.3f}     Mean Val  F1: {val_f1_mean:.3f}  std: {val_f1_std:.3f}     Mean Val  AUC: {val_auc_mean:.3f}  std: {val_auc_std:.3f}")
        print(f"Mean Test Accuracy: {test_acc_mean:.3f}  std: {test_acc_std:.3f}     Mean Test F1: {test_f1_mean:.3f}  std: {test_f1_std:.3f}     Mean Test AUC: {test_auc_mean:.3f}  std: {test_auc_std:.3f}\n\n")
    else:
        print("*** Cross Validation Statistics ***")
        print(f"\nMean Test Accuracy: {test_acc_mean:.3f}  std: {test_acc_std:.3f}     Mean Test F1: {test_f1_mean:.3f}  std: {test_f1_std:.3f}     Mean Test AUC: {test_auc_mean:.3f}  std: {test_auc_std:.3f}\n\n")
        
    # print test accuracy in each classes
    each_acc=[]
    
    for key in range(len(acc_raw_each_class_dict)):
        test_raw_accuracy=np.array(acc_raw_each_class_dict[key])
        val_raw_accuracy=np.array(val_acc_raw_each_class_dict[key])
        
        if early_stopping =="0":
            each_acc_cont=f"Validation Accuracy for Class {key}: {np.mean(val_raw_accuracy):.3f}  std: {np.std(val_raw_accuracy):.3f}  |  Test Accuracy for Class {key}: {np.mean(test_raw_accuracy):.3f}  std: {np.std(test_raw_accuracy):.3f}"
        else:
            each_acc_cont=f"Test Accuracy for Class {key}: {np.mean(test_raw_accuracy):.3f}  std: {np.std(test_raw_accuracy):.3f}"
        
        each_acc.append(each_acc_cont)
        print(each_acc_cont)
    print("\n")
    print("***********************************\n\n")
    #________________________________
    
    # with open(f"./{model_parameter_folder}/z_multimodal_training_reports_{tag}.txt",'a') as train_report:

    if early_stopping =="0":
        train_report.write(f"\nMean Val  Accuracy: {val_acc_mean:.3f}  std: {val_acc_std:.3f}     Mean Val  F1: {val_f1_mean:.3f}  std: {val_f1_std:.3f}     Mean Val  AUC: {val_auc_mean:.3f}  std: {val_auc_std:.3f}+\n")
        train_report.write(f"Mean Test Accuracy: {test_acc_mean:.3f}  std: {test_acc_std:.3f}     Mean Test F1: {test_f1_mean:.3f}  std: {test_f1_std:.3f}     Mean Test AUC: {test_auc_mean:.3f}  std: {test_auc_std:.3f}\n\n")
        train_report.write("\n".join(each_acc)+"\n")
    else:
        train_report.write(f"\nMean Test Accuracy: {test_acc_mean:.3f}  std: {test_acc_std:.3f}     Mean Test F1: {test_f1_mean:.3f}  std: {test_f1_std:.3f}     Mean Test AUC: {test_auc_mean:.3f}  std: {test_auc_std:.3f}\n\n")


    if device==f"cuda:{cuda_vis_dev}":
        torch.cuda.empty_cache()


