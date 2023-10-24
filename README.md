# MAE-ViTClassifier
Masked-Autoencoder Vision Transformer, fine tuned for classification of white blood cells. 

> To download the trained models, weights and training data, go here: https://drive.google.com/drive/folders/1o4GPH2EEgmVH2pQ8pcBgf9YnH2i5BHIg?usp=sharing

# MAE Training

> WARNING: this is a long and slow process. For 5x A6000 cards, the process took 8hours. 

**To generate the training data**

1. run util>crop_image.py on the pRCC dataset
2. consolidate all images from CAM16 and the cropped pRCC dataset into a new folder (e.g. `pre_training` ) 

**To train the MAE from scratch** 

1. use `main_mae_viz.py` to train a MAE model from scratch. Investigate the file for model archtiecture and point it to the right folders for pre_training 
2. If unable to run, then use `run_mae.py` using commandline arguments. 
3. Below is an example commandline execution
	- ensure dateset name points to the full path of your dataset root
	- all others are relative filepaths 

```javascript
  python3 run_mae.py \
  --dataset_name ~/NUS-NeuralNetworks \
  --train_dir "data/mae_train/*" \
  --validation_dir "data/mae_test/*" \
  --output_dir ~/NUS-NeuralNetworks/models/MAE_full100_4 \
  --remove_unused_columns False \
  --label_names pixel_values \
  --mask_ratio 0.25 \
  --base_learning_rate 8e-5 \
  --weight_decay 0.01 \
  --num_train_epochs 100 \
  --warmup_ratio 0.05 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 4 \
  --per_device_eval_batch_size 12 \
  --logging_strategy epoch \
  --logging_steps 200 \
  --save_strategy epoch \
  --load_best_model_at_end True \
  --save_total_limit 3 \
  --seed 12 \
  --do_train \
  --do_eval \
  --overwrite_output_dir \
  --evaluation_strategy epoch \
```


# ViT Training

**Training the ViT from scratch**

- use `main_vitclassifier_fromscratch.py` 
- a `Folder` class defines the location of the different training sets, so configure that to suit your data. 1 class instance = 1 training run
- for hyperparameters, review the arguments in `run_model` function. 
- For ablation studies, ensure the architecture matches the models used in the MAE run

**Training the ViT from MAE transfer learning**

- use `main_vitclassifier_transferlearning.py` ** **
- similar to the VIT from scratch:
	- a `Folder` class defines the location of the different training sets, so configure that to suit your data. 1 class instance = 1 training run
	- for hyperparameters, review the arguments in `run_model` function. 

# Evaluation for MaE 

- for training plots, use `eval_mae_trainingPlots.py` 
- for reconstruction, use `eval_mae_viz.py` 

# Evaluation for ViT 

- for training plots, use `eval_vit_trainingPlots.py`
- for evaluating performance on testset, use `eval_performanceMetrics.py` . It runs the model on the test set folder and builds a dataframe with predicted, actual and pvalue. Each one is exported as a csv to the model folder 
- for visualizing performance, use `eval_vit_performancePlots.py` 

---

# Dataset Statistics

## Statistics of WBC
|||   WBC_100   || WBC_50 |      | WBC_10 |      | WBC_1 |      |
|:-----------:|:-----:|:----:|:----------:|:------:|:----:|:------:|:----:|:-----:|:----:|
|     | Train |      | Validation |  Train |      |  Train |      | Train |      |
|     Class    |  data | mask |    data    |  data  | mask |  data  | mask |  data | mask |
|   Basophil  |  176  |  17  |     36     |   88   |   8  |   17   |   1  |   1   |   0  |
| Eosinophils |  618  |  61  |     126    |   309  |  30  |   61   |   6  |   6   |   0  |
|  Lymphocyte |  2015 |  201 |     412    |  1007  |  100 |   201  |  20  |   20  |   2  |
|   Monocyte  |  466  |  46  |     95     |   233  |  23  |   46   |   4  |   4   |   0  |
|  Neutrophil |  5172 |  517 |    1059    |  2586  |  258 |   517  |  51  |   51  |   5  |
|    Total#   |  8447 |  842 |    1728    |  4223  |  419 |   842  |  82  |   82  |   7  |



## Statistics of CAM16 and pRCC
|  ||CAM16                             ||| pRCC  |
|:------:|:-----:|:----:|:----:|:----:|:-------:|
|        | Train       || Validation  | Test | Train |
|        | data  | mask | data | data | data  |
| Normal | 379   | 37   | 54   | 108  | -  |
| Tumor  | 378   | 37   | 54   | 108  | -     |
| Total# | 757   | 74   | 108  | 216  | 1419  |




We summarize the statistics of datasets WBC, CAM16 and pRCC in the Tables above.
In WBC_100, we provide a ratio of 5:1 data for training and validation set in each cell type. And there are three segregations (i.e., WBC_50, WBC_10, WBC_10) for WBC_100, which contain 50%, 10% and 1% data of the whole set, respectively. Both WBC and CAM16 have additional mask annotation, where 10% of the samples have masks, each name corresponding to the image name. Note, pRCC and CAM16 are offered as the pre-training set, and pRCC comes without any label.