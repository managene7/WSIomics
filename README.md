
<h2>WSIomics: A Multimodal Deep Learning Pipeline for Binary Classification Using Whole-Slide Images and Transcriptomic Profiles in Cancer</h2>
WSIomics is an end-to-end pipeline to train multimodal AI models using whole-slide images (WSIs) and transcriptomic profile data. It is designed for binary classification and can be applied to various cancer types. The pipeline includes an automated marker gene search to reduce the dimension of transcriptome data. The pipeline also includes a new training stopping method, elbow stopping, designed to find the optimal training epoch for sparse data. 
The pipeline adapts intermediate multimodal fusion. Therefore, it trains three models. One is the WSI model, another is the transcriptome model, and the other is the multimodal model. WSI model training was performed by the CLAM package (https://github.com/mahmoodlab/CLAM) with the UNI pretrained encoder. For the transcriptomic model, normalized expression values, such as transcripts per million, were used as input. 
<p></p>

![Picture4](https://github.com/user-attachments/assets/670feddd-e180-4980-9e30-055ba11d76a8)

<h3>Installation</h3>

1. Download the WSIomics pipeline<br>
   ```
   git clone https://github.com/managene7/WSIomics.git
   ```
3. Create conda environment and install Python libraries<br>
   ```
   conda create --name <env> --file <this file> python=3.11.7<br>
   pip install -r requirements.txt
   ```
4. Install prerequisite software<br>
   ```
   apt-get install -y openslide-tools libgl1-mesa-glx zlib1g libpng-dev libjpeg-dev libtiff-dev libopenjp2-7 libopenjp2-tools libgdk-pixbuf2.0-dev libxml2-dev libsqlite3-dev libglib2.0-dev libcairo2-dev pkg-config valgrind emacs-nox libltdl-dev libtiff-tools exiftool git git-lfs cmake liblcms2-dev libtiff-dev libpng-dev libheif1 libheif-dev libz-dev unzip libzstd-dev libwebp-dev build-essential hwinfo
   ```
5. Install openJPG<br>
   ```
   wget https://github.com/uclouvain/openjpeg/archive/master.zip<br>
   unzip master.zip<br>
   cd openjpeg-master/<br>
   mkdir build<br>
   cd build<br>
   cmake -DCMAKE_BUILD_TYPE=Release ..<br>
   make<br>
   make install<br>
   make clean<br>
   ```
6. Install UNI encoder<br>
   Follow the instruction in this link: https://github.com/mahmoodlab/UNI/blob/main/README_old.md

<h3>Docker Image</h3>
We recommend to use the docker container for the pipeline.<br>

Download the image:<br>
```
docker pull managene7/wsiomics:v.2.1
```

Run docker image (change '/path/to/data' to the path where data exists): <br>
```
docker run -it --gpus=all --shm-size 64g -v /path/to/data:/DATA managene7/wsiomics:v.2.1 bash
```

<h3>Run the pipeline</h3>
<h4>Download example datasets</h4>
*** Example datasets exist in the Example_datasets folder of the WSIomics pipeline in GitHub. ***

To download WSIs from TCGA database use the UCEC_Phamaceutical_adj_gdc_manifest.txt file in the Example_datasets folder.

If gdc-client is not installed, install the gdc-client:

```
conda install -c bioconda gdc-client
```

Download WSIs using gdc-client (change '/path/to/' to the path where the manifest file exists):

```
mkdir WSIs
cd WSIs
gdc-client download -m /path/to/UCEC_Phamaceutical_adj_gdc_manifest.txt
```
WSI file folder must contain WSI file only. After downloading using gdc-client, move the WSI files using 'move_WSIs.py'.

```
python move_WSIs.py
# enter WSI folder name
```

<h4>Display help</h4>

```
WSIomics.py --help
```

Then it will displsy the following content:
```
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

```


<h4>command example</h4>
The pipeline can run WSI, transcriptome, and multimodal model training all at once or individually, but WSI model training must be run first because transcriptome and multimodal model training use data generated by WSI model training.
The following is one example to run all the three model training with a single command line.

```
WSIomics.py --exec 123 --wsi_folder WSIs --classification_csv UCEC_PFI_548__pfi6-pfi36_labeled.csv --label_format Classification_format.txt --TPM_file UCEC_CDS.csv --tag test_training --early_stopping 2000 --elbow_prop 0.2 --marker_search 2 --raw_clinical_data UCEC_PFI_548.csv --num_markers 1000
```

For gene expression, TPM is used here. However, any kinds of normalized expression data, such as FPKM and RPKM, can be used.

<h5>Example of classification_csv:</h5>

```
case_id	slide_id	label
TCGA-D1-A0ZS	TCGA-D1-A0ZS-01A-01-MS1.28d10901-23db-4a0e-884b-ff838e408be2	Resistant
TCGA-D1-A0ZS	TCGA-D1-A0ZS-01Z-00-DX1.8021A060-3CA2-418E-AE16-C48E911F5C25	Resistant
TCGA-EY-A72D	TCGA-EY-A72D-01A-01-TS1.6290A65C-CECA-4C42-879F-A11519E88993	Resistant
TCGA-EY-A72D	TCGA-EY-A72D-01Z-00-DX1.4303F43E-9702-4089-A854-4072C87A1FAC	Resistant
TCGA-D1-A16E	TCGA-D1-A16E-01Z-00-DX1.C5BE7CE0-B192-4E7C-AC9D-51854C0FCDB5	Sensitive
TCGA-D1-A16E	TCGA-D1-A16E-01A-02-TSB.0d3cd7be-8add-4d9b-b147-7a9e3c7fe00f	Sensitive
TCGA-AJ-A3NC	TCGA-AJ-A3NC-01Z-00-DX1.A31856AC-EDD5-4955-BFED-08C162A1C143	Sensitive
TCGA-AJ-A3NC	TCGA-AJ-A3NC-01A-01-TSA.CDB70B96-0425-4593-8052-1C5A87A855EC	Sensitive
......
```

<h5>Example of label_format:</h5>

```
Sensitive       0
Resistant       1
```

<h5>Example of raw_clinical_data:</h5>

```
case_id	PFI
TCGA-AX-A3FV	0
TCGA-SL-A6J9	2
TCGA-AJ-A8CW	4
TCGA-AX-A3FT	6
TCGA-AX-A3FZ	6
TCGA-AX-A3FS	7
TCGA-SL-A6JA	7
TCGA-AX-A3G6	8
TCGA-AX-A3G4	9
TCGA-AX-A3G7	9
TCGA-AX-A3GI	12
TCGA-4E-A92E	13
...
```

<h5>'--early_stopping'</h5>

The default option of determining optimal training epochs is early stopping. However, if the '--early_stopping' option value is set as 2000 or other number, the pipeline will use the elbow stopping function. The elbow stopping function determines the epoch value at elbow point of training loss curve first. The '--elbow_prop' option will determine the training stopping epoch by proportion of the epoch at elbow point. The elbow_prop value should be determined empirically from several training tests.

<h5>'--marker_search'</h5>

The marker search function will automatically determine the marker genes using clinical data (for example, progression-free interval values) and transcriptome expression values. For transcriptome expression values, any kinds of normalized expression values, such as TPM, FRKM, and RPKM, can be used. 


<h5>This pipeline is developed by Center for Biomedical Informatics and Information Technology, National Cancer Institute, National Institutes of Health</h5>
=== If you have any questions or suggestions, please contact to Dr. Minkyu Park. email: minkju3003@gmail.com ===
