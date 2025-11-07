#!/usr/bin/env python

import os

"""
Created on Mon Jan 18 17:41:10 2021

@author: minkj
"""
#________________ option parse _______________________________
import sys 

args = sys.argv[1:]

option_dict={
    '--cuda_vis_dev':'0','--patch_level':'1', '--wsi_ext':'svs', '--encoder':"1", '--task':'1', '--num_split':'10', '--val_frac':'0.1', '--test_frac':'0.1', '--embed_dim':'1', '--drop_out':'0.25', '--model_type':'1', '--num_classes':'2','--create_heatmap':'1', '--restart':'1', '--patch_size':'256', '--wsi_seed':'1', '--exec':'123', '--lr_transcriptome':"0.000001", '--lr_multimodal':"0.0000001",'--batch_size':"32", '--rnd_seed':'3000', '--iteration':"1", '--tag':'Results', '--multimodal_type':'MLP','--multimodal_embed_dim':64, '--raw_clinical_data':"",'--num_markers':"500", "--heatmapt_chkpt":"0", "--marker_search":"2", "--purpose":"1", "--log2": "1", "--scaling":"1", "--filtering": "1", "--mean_threshold":"1", "--early_stopping":"0", "--elbow_prop":0.5}
help="""
===================================================================================================================================
To run this pipeline, you must prepare the following two files first:

1. WSI classification file in CSV format.
example (important: slide ID must be without extension): 

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

singularity exec --nv --bind /host/DATA/folder:/DATA WSIomics.sif WSIomics.py --help


Example to run in Docker:

docker run -it -v /host/DATA/folder:/DATA --gpus=all --shm-size 64g managene7/WSIomics:v.1.2 bash

WSIomics.py --help


* '/host/DATA/folder' => host folder path that has the folder for input WSI files.
** Use options instead of '-help'

If you find any bugs or have a recommendation, send an email to: minkju3003@gmail.com
___________________________________________________________________________________________________________________________________

###   Select modules to run   ###
--exec                  (option)        123 : Execute WSI, transcriptome, and multimodal training (default)
                                        23  : Execute transcriptome and multimodal training
                                        1   : Execute WSI training only
                                        2   : Execute transcriptome training only
                                        3   : Execute multimodal training only

###   CLAM options   ###
--wsi_folder            (required)      path to a folder containing raw whole-slide image files.
--classification_csv    (required)      classification csv file name [format (header): case_id, slide_id (without extension), label]
--label_format          (required)      file name for label format information

--cuda_vis_dev          (option)        cuda_visible_device (default is '0')
--val_frac              (option)        fraction for the validation (default: 0.1)
--test_frac             (option)        fraction for the test (default: 0.1)
--num_split             (option)        number of splits for validation (default: 10)
--wsi_seed              (option)        Random seed for CLAM. (default is 1)
--patch_level           (option)        Patch level of WSIs. 0, 1 (default), or 2           <== required for heatmap
--wsi_ext               (option)        Extension of image. svs or tiff (default is svs)    <== required for heatmap
--patch_size            (option)        Patch size for creating patches (default is 256)
--drop_out              (option)        0< float value <1 (default: 0.25)

--create_heatmap        (option)        1: no (default).                          
                                        2: create heatmaps only.                            <== '--exec' option must contain 1
                                        3: create heatmaps with the pipeline execution      <== '--exec' option must contain 1
--heatmapt_chkpt        (option)        check point number of WSI model for heatmap (default is 0)
                                        
--restart               (option)        Restart CLAM pipeline                               <== '--exec' option must contain 1
                                        1: from create patches (default)
                                        2: from feature extraction
                                        3: from training splits
                                        4: from training
                                        5: from evaluation
                                        6: create heatmap 

###   Multimodal options   ###
--TPM_file              (required)      csv file name containing TPM data for transcriptome or multimodal model
--tag                   (option)        Tag to be added to output folder and file names (default is 'Results')
--iteration             (option)        Number of iterations with different random seeds (default is 1)   
--rnd_seed              (option)        random seed for transcriptome and multimodal models (default is 3000)
--lr_transcriptome      (option)        learning rate for transcriptome model training (default is 0.000001)
--lr_multimodal         (option)        learning rate for multimodal model training (default is 0.0000001)
--batch_size            (option)        batch size for both transcriptome and multimodal model (default is 32)
--log2                  (option)        convert to log2 value. 1: yes, 2: no (default ia 1)
--scaling               (option)        perform standard scaling. 1: yes, 2: no (default is 1)
--filtering             (option)        filter out low-expression genes. 1: yes, 2: no (default is 1)
--mean_threshold        (option)        average omics value to be used for filtering low values (default is 1)
--early_stopping        (option)        0 to use the early stopping function 
                                        or specific epoch number (e.g. 2000) to be stopped by elblow stopping (default is 0)
--elbow_prop            (option)        proportion to determine epoch from the epoch of the elbow point. (default is 0.5)
                                        Valid when --early_stopping > 0. This value should be determined empirically.

###   Automated TPM marker search   ###
--marker_search         (option)        1: no,  2: by trendline (default)
--raw_clinical_data     (option)        raw clinical data file (e.g., Progress-Free Interval) (default is '')
                                        => If not entered, the pipeline will omit the marker search process.
--num_markers           (option)        number of markers to be identified by slope value of trendline (default is 500)

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
            if args[0]=="-help" or args[0]=="-h":
                print (help)
                sys.exit()

#__option check_____

curr_dir=os.getcwd()

##__add DATA_tag to input files
option_dict['--wsi_folder']=option_dict['--wsi_folder']
option_dict['--classification_csv']=option_dict['--classification_csv']
option_dict['--label_format']=option_dict['--label_format']

##__output folder names__
created_patch_folder=option_dict['--wsi_folder']+"_1_CREATED_PATCHES"
feature_output=option_dict['--wsi_folder']+"_2_FEATURE_OUTPUTS"
split_folder=option_dict['--wsi_folder']+"_3_SPLIT_OUTPUTS"
train_out_dir=option_dict['--wsi_folder']+"_4_TRAINING"
eval_out_dir=option_dict['--wsi_folder']+"_5_EVALUATION"
heatmap_out_dir=option_dict['--wsi_folder']+"_6_HEATMAPS"

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
elif option_dict['--encoder']=="2":
    model_name='conch_v1'
    embed_dim="512"
else:
    print ("ERROR!!: you chose the wrong '--encoder' option. Choose the right option and try it again.")
    sys.exit()


if option_dict['--model_type']=="1":
    model_type="clam_sb"
elif option_dict['--model_type']=="2":
    model_type="clam_mb"
elif option_dict['--model_type']=="3":
    model_type="mil"
else:
    print ("ERROR!!: you chose the wrong '--model_type' option. Choose the right option and try it again.")
    sys.exit()

if option_dict['--restart']=="6":
    option_dict['--create_heatmap']="2"
#__________________


if '1' in option_dict['--exec']:
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
    
        # 3. training splits
    
        else:
            if 3 >= int(option_dict['--restart']):
                command_3 = f"python {curr_dir}/create_splits_seq.py --task task_1_tumor_vs_normal --seed 1 --k {option_dict['--num_split']} --label_file {option_dict['--label_format']} --split_folder {split_folder} --classification_csv {option_dict['--classification_csv']} --task {task} --val_frac {option_dict['--val_frac']} --test_frac {option_dict['--test_frac']}"
                out_log.write(command_3+"\n")
                print ("\n\n3/6___Training splits in progress..\n", command_3, "\n\n")
                training_splits=os.system(command_3)
                
            else:
                training_splits=0
    
        if training_splits!=0:
            print ("ERROR!!: Training splits failed. Please check input files or options and try it again.\n\n")
            sys.exit()
    
        # 4. training
        else:
            if 4 >= int(option_dict['--restart']):
                if option_dict['--task']=='1':
                    command_4=f"CUDA_VISIBLE_DEVICES={option_dict['--cuda_vis_dev']} python {curr_dir}/main.py --seed {option_dict['--wsi_seed']} --split_dir {split_folder} --feature_dir {feature_output} --classification_csv {option_dict['--classification_csv']} --results_dir {train_out_dir} --drop_out {option_dict['--drop_out']} --early_stopping --lr 2e-4 --k {option_dict['--num_split']} --exp_code {option_dict['--wsi_folder'].split('/')[-1]} --weighted_sample --bag_loss ce --inst_loss svm --task task_1_tumor_vs_normal --model_type {model_type} --log_data --data_root_dir {option_dict['--wsi_folder']} --embed_dim {embed_dim} --label_file {option_dict['--label_format']}"
                    out_log.write(command_4+"\n")
                    print ("4/6___Training started..\n", command_4, "\n\n")
                    training=os.system(command_4)
    
                elif option_dict['--task']=='2':
                    command_4=f"CUDA_VISIBLE_DEVICES={option_dict['--cuda_vis_dev']} python {curr_dir}/main.py --seed {option_dict['--wsi_seed']} --split_dir {split_folder} --feature_dir {feature_output} --classification_csv {option_dict['--classification_csv']} --results_dir {train_out_dir} --drop_out {option_dict['--drop_out']} --early_stopping --lr 2e-4 --k {option_dict['--num_split']} --exp_code {option_dict['--wsi_folder'].split('/')[-1]} --weighted_sample --bag_loss ce --inst_loss svm --task task_2_tumor_subtyping --model_type {model_type} --log_data --subtyping --data_root_dir {option_dict['--wsi_folder']} --embed_dim {embed_dim} --label_file {option_dict['--label_format']}"
                    out_log.write(command_4+"\n")
                    print ("4/6___Training started..\n", command_4, "\n\n")
                    training=os.system(command_4)
    
                else:
                    print ("ERROR!!: You chose the wrong '--task' option. Choose the right option and try it again.")
                    sys.exit()
            else:
                training=0
    
        if training!=0:
            print ("ERROR!!: Training failed. Please check input files or options and try it again.\n\n")
            sys.exit()
    
        # 5. evaluation
        else:
            if 5 >= int(option_dict['--restart']):
                
                if option_dict['--task']=='1':
                    command_5=f"CUDA_VISIBLE_DEVICES={option_dict['--cuda_vis_dev']} python {curr_dir}/eval.py --training_dir {train_out_dir} --split_dir {split_folder} --dataset_csv {option_dict['--classification_csv']} --k {option_dict['--num_split']} --models_exp_code {option_dict['--wsi_folder'].split('/')[-1]}_s{option_dict['--wsi_seed']} --save_exp_code {option_dict['--wsi_folder'].split('/')[-1]}_s{option_dict['--wsi_seed']}_cv --task task_1_tumor_vs_normal --model_type {model_type} --results_dir {eval_out_dir} --data_root_dir {feature_output} --label_file {option_dict['--label_format']}  --embed_dim {embed_dim}"
                    out_log.write(command_5+"\n")
                    print ("\n\n5/6___Evaluation started..\n", command_5, "\n\n")
                    evaluation=os.system(command_5)
    
                elif option_dict['--task']=='2':
                    command_5=f"CUDA_VISIBLE_DEVICES={option_dict['--cuda_vis_dev']} python {curr_dir}/eval.py --training_dir {train_out_dir} --split_dir {split_folder} --dataset_csv {option_dict['--classification_csv']} --k {option_dict['--num_split']} --models_exp_code {option_dict['--wsi_folder'].split('/')[-1]}_s{option_dict['--wsi_seed']} --save_exp_code {option_dict['--wsi_folder'].split('/')[-1]}_s{option_dict['--wsi_seed']}_cv --task task_2_tumor_subtyping --model_type {model_type} --results_dir {eval_out_dir} --data_root_dir {feature_output}  --label_file {option_dict['--label_format']}  --embed_dim {embed_dim}"
                    out_log.write(command_5+"\n")
                    print ("\n\n5/6___Evaluation started..\n", command_5, "\n\n")
                    evaluation=os.system(command_5)
            else:
                evaluation=0
    
        if evaluation!=0:
            print ("ERROR!!: Evaluation failed. Please check input files or options and try it again.\n\n")
            sys.exit()
    
    # 6. create heatmap.
    
    if option_dict['--create_heatmap'] in ['2','3']:
        
        label_file=open(option_dict['--label_format'], 'r')
        label_file_read=label_file.readlines()
        
        with open("temp_label.txt",'w') as tmp_label:
            label_cont=""
    
            num_classes=0
            for line in label_file_read:
                line=line.strip().split()
                try:
                    new_line="    "+line[0]+":    "+line[1]+"\n"
                    label_cont+=new_line
                    num_classes+=1
                except:
                    pass
            tmp_label.write(label_cont)
    
        tmp_label=open("temp_label.txt",'r')
        label_cont_whole=tmp_label.read()
        label_cont_whole=f"""{label_cont_whole}"""
        os.system("rm temp_label.txt")
    
        train_out_dir=option_dict['--wsi_folder']+"_4_TRAINING"
    
        config_template=f"""
# CUDA_VISIBLE_DEVICES={option_dict['--cuda_vis_dev']} python {curr_dir}/create_heatmaps.py --config config_template.yaml\n
--- 
exp_arguments:
  # number of classes
  n_classes: {str(num_classes)}
  # name tag for saving generated figures and assets
  save_exp_code: HEATMAP_OUTPUT 
  # where to save raw asset files
  raw_save_dir: {heatmap_out_dir}/heatmap_raw_results
  # where to save final heatmaps
  production_save_dir: {heatmap_out_dir}/heatmap_production_results
  batch_size: 256
data_arguments: 
  # where is data stored; can be a single str path or a dictionary of key, data_dir mapping
  data_dir: {option_dict['--wsi_folder']}/ 
  # column name for key in data_dir (if a dict mapping is used)
  data_dir_key: source
  # csv list containing slide_ids (can additionally have seg/patch paramters, class labels, etc.)
  process_list: {option_dict['--classification_csv']}
  # preset file for segmentation/patching
  preset: presets/bwh_biopsy.csv
  # file extention for slides
  slide_ext: .{option_dict['--wsi_ext']}
  # label dictionary for str: interger mapping (optional)
  label_dict:
{label_cont_whole}
patching_arguments:
  # arguments for patching
  patch_size: 256
  overlap: 0.5
  patch_level: {option_dict['--patch_level']}
  custom_downsample: 1
encoder_arguments:
  # arguments for the pretrained encoder model
  model_name: {model_name} 
  target_img_size: 224 # resize images to this size before feeding to encoder
model_arguments: 
  # arguments for initializing model from checkpoint
  ckpt_path: {train_out_dir}/s_{option_dict["--heatmapt_chkpt"]}_checkpoint.pt  
  model_type: clam_sb # see utils/eval_utils
  initiate_fn: initiate_model # see utils/eval_utils/
  model_size: small
  drop_out: 0.
  embed_dim: {embed_dim}
heatmap_arguments:
  # downsample at which to visualize heatmap (-1 refers to downsample closest to 32x downsample)
  vis_level: 1
  # transparency for overlaying heatmap on background (0: background only, 1: foreground only)
  alpha: 0.4
  # whether to use a blank canvas instead of original slide
  blank_canvas: false
  # whether to also save the original H&E image
  save_orig: true
  # file extension for saving heatmap/original image
  save_ext: png
  # whether to calculate percentile scores in reference to the set of non-overlapping patches
  use_ref_scores: true
  # whether to use gaussian blur for further smoothing
  blur: true
  # whether to shift the 4 default corner points for checking if a patch is inside a foreground contour
  use_center_shift: true
  # whether to only compute heatmap for ROI specified by x1, x2, y1, y2
  use_roi: false 
  # whether to calculate heatmap with specified overlap (by default, coarse heatmap without overlap is always calculated)
  calc_heatmap: true
  # whether to binarize attention scores
  binarize: false
  # binarization threshold: (0, 1)
  binary_thresh: -1
  # factor for downscaling the heatmap before final dispaly
  custom_downsample: 1
  cmap: jet
sample_arguments:
  samples:
    - name: "topk_high_attention"
      sample: true
      seed: 1
      k: 15 # save top-k patches
      mode: topk
          """
        with open("config_for_heatmap.yaml",'w') as config_file:
            config_file.write(config_template)
    
        if 6 >= int(option_dict['--restart']):
            command_6=f"CUDA_VISIBLE_DEVICES={option_dict['--cuda_vis_dev']} python create_heatmaps.py --config config_for_heatmap.yaml --heatmap_out_folder {heatmap_out_dir} --save_exp_code {option_dict['--wsi_folder']}"
            out_log.write(command_6+"\n")
            print ("\n\n6/6___Create heatmap started..\n",command_6, "\n\n")
            create_heatmap=os.system(command_6) 
    


if '2' in option_dict['--exec'] or '3' in option_dict['--exec']:
    import pandas as pd
    from Multimodal_training_utils import training
    from Multimodal_utils import min_max_norm_TPM

    os.environ["CUDA_VISIBLE_DEVICES"] = option_dict['--cuda_vis_dev']
    
    raw_wsi_folder=option_dict['--wsi_folder']
    TPM_data_path=option_dict['--TPM_file']
    

    
    WSI_h5_feature_path=f"./{raw_wsi_folder}_2_FEATURE_OUTPUTS/h5_files"
    WSI_chkpoint_path=f"./{raw_wsi_folder}_4_TRAINING/"
    data_split_path=f"./{raw_wsi_folder}_4_TRAINING/" 
    
    
    WSI_label_csv_path=option_dict['--classification_csv']
    label_format_path=option_dict['--label_format']   

    #____________________________

    
    for k in range(int(option_dict['--iteration'])):
        
        rand_num = int(option_dict['--rnd_seed'])+k*10

        tag=option_dict['--tag']+"_rnd"+str(rand_num)


        all_result_folder=TPM_data_path[:-4]+f"_{tag}"
        
        if os.path.isfile(f"./{all_result_folder}/z_multimodal_training_reports_{tag}.txt"):
            os.system(f"rm ./{all_result_folder}/z_multimodal_training_reports_{tag}.txt")
    
        loss_path=all_result_folder


        # run transcriptome training module        
        if '2' in option_dict['--exec']:
            training(data_split_info=data_split_path, WSI_label_csv=WSI_label_csv_path, label_format=label_format_path, TPM_data_path=TPM_data_path, loader="transcriptome", learning_rate=float(option_dict['--lr_transcriptome']), batch_size=int(option_dict['--batch_size']), rand_seed=rand_num, tag=tag, loss_path=loss_path, multimodal_type=option_dict['--multimodal_type'], cuda_vis_dev=option_dict['--cuda_vis_dev'], clinical_data=option_dict['--raw_clinical_data'], num_markers=option_dict['--num_markers'], marker_search=option_dict['--marker_search'], log2=option_dict["--log2"], scaling=option_dict['--scaling'], filtering=option_dict['--filtering'], mean_threshold=option_dict['--mean_threshold'], early_stopping=option_dict["--early_stopping"], elbow_prop=float(option_dict["--elbow_prop"]))


        # run multimodal training module
        if '3' in option_dict['--exec']:

            training(data_split_info=data_split_path, WSI_label_csv=WSI_label_csv_path, label_format=label_format_path, TPM_data_path=TPM_data_path, loader="multimodal", learning_rate=float(option_dict['--lr_multimodal']), batch_size=int(option_dict['--batch_size']), rand_seed=rand_num, tag=tag, loss_path=loss_path, WSI_h5_feature_path=WSI_h5_feature_path, WSI_chkpoint_path=WSI_chkpoint_path, multimodal_type=option_dict['--multimodal_type'], multimodal_embed_dim=int(option_dict['--multimodal_embed_dim']), cuda_vis_dev=option_dict['--cuda_vis_dev'], clinical_data=option_dict['--raw_clinical_data'], num_markers=option_dict['--num_markers'], marker_search=option_dict['--marker_search'], log2=option_dict["--log2"], scaling=option_dict['--scaling'], filtering=option_dict['--filtering'], mean_threshold=option_dict['--mean_threshold'], early_stopping=option_dict["--early_stopping"], elbow_prop=float(option_dict["--elbow_prop"]))

        


