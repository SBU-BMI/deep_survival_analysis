# Deep Learning for Survival Analysis in Breast Cancer with Whole Slide Image Data
This repo provides the code for "Deep Learning for Survival Analysis in Breast Cancer with Whole Slide Image Data", Bioinfomatics, 2022.

# Dependencies
Python 3.6 
PyTorch 1.0 
OpenSlide 1.1.1
Lifeline 0.26.3

# Data
## Data Preprocess
As the tumor prediction data and the TIL prediction data are tuples, we need to transform the tuples into images and interpolate tumor prediction data and the TIL prediction data into tiles. We also need to divide the Whole Slides Image (WSI) data into tiles. Configure the "root_path" and the "output_path" for "WSI", "Nuclei_segs", "Tumor_preds" and "TIL_preds" in the ./configs/CONFIG.json. Run
```
python interpolation.py --config ./configs/CONFIG.json 
```
to transform the tumor and TIL prediction data into images/tiles. Run 
```
python wsi2patches.py --config ./configs/CONFIG.json 
```
to divide the WSI data into tiles. 

We also need to compute the foreground mask that filter out the background tissue of the WSIs. Run
```
python micnn_train4_ml.py --config ./configs/CONFIG.json --scale SCALE --fg
```
SCALE: 1, 4, 8 or 16. The foreground masks are scale specific, meaning for each scale, we need to compute the foreground masks for it. 

## RGB Whole Slide Image (WSI) data
RGB WSI data are stored in tiles. Let WSI-ROOT be the root of WSI data. The folder is organized as follows:

WSI-ROOT/

--------TCGA-3C-AALJ-01Z-00-DX1/
------------------1_1_4000_4000_0.25_1_PATCH.png
------------------1_4001_4000_4000_0.25_1_PATCH.png
...

--------TCGA-3C-AALK-01Z-00-DX1/
------------------1_1_4000_4000_0.25_1_PATCH.png
------------------1_4001_4000_4000_0.25_1_PATCH.png
...

...

WSI-ROOT contains a list of folders. The name of each folder is the WSI id. For example, TCGA-3C-AALJ-01Z-00-DX1 in the above structure is the WSI id. Each folder named as WSI id contains a list of png files, which are tiles of the WSI. Each tile is of size 4000 by 4000. The format of each png file is: x_y_tx_ty_mpp_1_PATCH.png, where
x: the x coordinate of the tile, from left to right starting from 1
y: the y coordinate of the tile, from top to bottom starting from 1
tx: the width of the tile 
ty: the height of the tile
mpp: the MPP value, ~0.25

## Nuclear segmentation data, Tumor prediction data, TIL prediction data
Denote the 3 channel image that is concatenated using the nuclear segmentation map, the tumor prediction map and the TIL prediction map by NTL.
### Nuclear segmentation data
Let Nu-ROOT denote the root of the nuclear segmentation map. The data is organized as follows:

Nu-ROOT/

--------TCGA-3C-AALJ-01Z-00-DX1.777C0957-255A-42F0-9EEB-A3606BCF0C96.svs/
------------------1_1_4000_4000_0.25_1_SEG.png
------------------1_4001_4000_4000_0.25_1_SEG.png
...

--------TCGA-3C-AALK-01Z-00-DX1..4E6EB156-BB19-410F-878F-FC0EA7BD0B53.svs/
------------------1_1_4000_4000_0.25_1_SEG.png
------------------1_4001_4000_4000_0.25_1_SEG.png
...

...

The format of the file names are similar to those of the RGB WSI data.

### Tumor prediction data
Let the Tu-ROOT be the root of the tumor prediction data. The data is organized as follows:

Tu-ROOT/

--------TCGA-3C-AALJ-01Z-00-DX1/
------------------1_1_4000_4000_0.25_1_INTP.png
------------------1_4001_4000_4000_0.25_1_INTP.png
...

--------TCGA-3C-AALK-01Z-00-DX1/
------------------1_1_4000_4000_0.25_1_INTP.png
------------------1_4001_4000_4000_0.25_1_INTP.png
...

...

The format of the file names are similar to those of the RGB WSI data.

### TIL prediction data
Let the TIL-ROOT be the root of the tumor prediction data. The data is organized as follows:

TIL-ROOT/

--------TCGA-3C-AALJ-01Z-00-DX1/
------------------1_1_4000_4000_0.25_1_INTP.png
------------------1_4001_4000_4000_0.25_1_INTP.png
...

--------TCGA-3C-AALK-01Z-00-DX1/
------------------1_1_4000_4000_0.25_1_INTP.png
------------------1_4001_4000_4000_0.25_1_INTP.png
...

...

The format of the file names are similar to those of the RGB WSI data.

# Train and test 
## Set the configure file
Before training the model, we need to set the parameters in the configure file. All experimental settings are stored in ./configs/CONFIG.json. I list the parameters you need to set below. You can ignore other parameters in the CONFIG.json.
["tile_precess"]["WSIs"]["output_path"]: the root path of WSI images, i.e., WSI-ROOT. 
["tile_precess"]["Nuclei_segs]["output_path"]: the root path of nuclear segmentation maps, i.e., Nu-ROOT. 
["tile_precess"]["Tumor_preds"]["output_path"]: the root path of tumor prediction data, i.e., Tu-ROOT. 
["tile_precess"]["TIL_preds"]["output_path"]: the root path of TIL prediction data, i.e., TIL-ROOT. 
["tile_precess"]["label_file"]: the label file, check out the get_wsi_id_labels function in data_preprocess/data_preprocess.py to see how I get the labels for the WSIs according to WSI ids. 
["dataset"]["input_nc"]: 6 or 3. Default: 6. 
["dataset"]["data_part"]: 1 or 2 or 3. Default: 3. 1 means train only one model using RGB WSI data. 2 means train only one model using NTL data. 3 means train two models using RGB and NTL data, respectively. If ["dataset"]["data_part"] = 1 or 2, then ["dataset"]["input_nc"] should be 3. If If ["dataset"]["data_part"] = 3, then ["dataset"]["input_nc"] should be 6. 
["dataset"]["data_file_path"]: Path to the data split. In general, it should be the absolute path of ./datasets which contains: train.txt, valid.txt and test.txt.
["dataset"]["n_patches_per_wsi"]: Default 16. Number of patches per WSI during each training epoch. 
["dataset"]["n_patches_per_wsi_eval"]: Default 1024. Number of patches per WSI for validation and testing. 
["dataset"]["interval"]: Default 750. Number of days to divide the survival time.  
["dataset"]["n_intervals"]: Default 5. Number of intervals. 
["dataset"]["batch_size"]: Default 16. Batch size on number of WSIs.
["dataset"]["num_workers"]: Default 64. 
["dataset"]["patch_size"]: Default [224, 224]. Patch size for training the deep learning model. 
["dataset"]["max_num_patches"]: Default 2000. Maximum of number of patches per WSI. 
["dataset"]["mask_root"]: mask root. The mask is for filter the non-tissue background of the WSI. 
["train"]["n_epochs"]: Default 2000. Number of epochs. 
["train"]["learning_rate"]: Default 1e-4. Learning rate.
["train"]["output_dir"]: output directory. 
["train"]["log_freq"]: Default 1. How many epochs to record the losses. 
["train"]["save_freq"]: Default 500. How many epochs to save the checkpoints.

## Train
To train the deep learning model for survival analysis, run the following command:
```
python micnn_train4_ml.py --config ./configs/CONFIG.json --scale SCALE --gpu_ids GPU_ID
```
We tried SCALE of 1, 4, 8 and 16. 

## Extract the deep learning features for each WSI
After the model is trained, we extract features for each WSI
Run
```
python aggregate_features_4.py --config ./configs/CONFIG.json --scale SCALE --mode MODE --ch CH --epoch EPOCH --gpu_ids GPU_ID
```
MODE: train or valid or test. Extract features for training data, validation data or testing data.
CH: rgb or pred. Extract features for RGB WSI data or NTL data. 
EPOCH: Checkpoint at which epoch, 500, 1000, 1500, 2000
Let OUTPUT-DIR be the output directory. The feature of each WSI will be saved under OUTPUT-DIR/feat_dir/epoch_$EPOCH/MODE/WSI-ID. The feature file name of the RGB WSI is feat_level_out_rgb.npy, and the feature file name of the NTL data is feat_level_out_pred.npy. 

## Extract clinical features for each WSI
clinical_3feat.py extracts age, gender and stage features for each WSI. In clinical_3feat.py:
feat_root: the directory that contains train, valid and test, these three subdirectories. Each subdirectory contains the folders that are named by WSI IDs. feat_root is just used to read the WSI_IDs from train/valid/test these three modes.
brca_info_path = './dataset/brca_info.csv' (default).
csv_file_path = './dataset/dataset_for_survival.csv' (default).
clinic_feat_dir: the directory you want to save the clinical features.

## Train and test the Cox survival analysis model
Edit the data_info['train'], the data_info['valid'] and the data_info['test'] variables in cox_prediction_multi_feat_tune.py
For example, data_info['train'] = [[FEAT-DIR-1, FN-1], [FEAT-DIR-2, FN-2], ... ], where FEAT-DIR-1 is the directory of the features for training data and FN-1 is the file name. This file will concatenate the features of FEAT-DIR-1, FEAT-DIR-2, and so on. FEAT-DIR-* could be feature directories of different scales or the directory of clinical feature. data_info['valid'] and data_info['test'] could be set similarly. Set the PCA ratios data_info['pca_ratio'] = []. The length of data_info['pca_ratio'] should be equal to the length of data_info['train']. Each value in data_info['pca_ratio'] corresponds to each feature directory. Then, run 
```
python cox_prediction_multi_feat_tune.py 
```
and check the outputted c-index. 

Edit the data_info['train'], the data_info['test'] and the data_info['pca_ratio'] variables in cindex_bootstrap_multi_feat.py and run 
```
python cindex_bootstrap_multi_feat.py 
```
to get the 95\% confidence intervals of c-indices and hazard ratios of bootstrap sampling. 






