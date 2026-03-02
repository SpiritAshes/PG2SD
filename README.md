# PG<sup>2</sup>SD 

This repository contains the code corresponding to the paper "Progressive gradient-guided self-distillation keypoint detection" published in "TMM". The code is currently a work - in - progress. We are constantly updating and optimizing it to ensure better performance and functionality.


# Requirements

    python==3.6
    pytorch==1.7.1+cu110
    tensorboard
    tqdm
    numpy
    pyyaml
    opencv-python
    pillow
    progress
    path
    
## Dataset Source

1. download the [Aachen dataset](https://drive.google.com/drive/folders/1fvb5gwqHCV4cr4QPVIEMTWkIhCpwei7n) to `datasets/aachen`.
2. ./download_training_data.sh

## Training Configuration

You have the flexibility to tailor the training process to your specific needs by modifying the configuration parameters in the `config/train.yaml` file. This allows you to adjust settings such as learning rates, batch sizes, and other hyperparameters to optimize training for your dataset and hardware setup.

## Running the Training Script

Make sure your data is organized correctly, as this will impact the training process.

```
python train.py
```

## Viewing training results

   Run the following command to launch TensorBoard and visualize the training metrics:

```
tensorboard --logdir=[path_to_your_logs]
```
 
   Replace `path_to_your_logs` with the actual path to the directory containing your TensorBoard logs.

  
## Evaluation on HPatches

1. Download the Hpatches dataset

```
cd eval_hpatches/hpatches_sequences/
bash download.sh
cd ..
```

2.Configure the `../config/extract.yaml` file before the extraction point.

```
python extract_points.py
```

3. evaluation

```
python eval.py
```

## Evaluation on Aachen Day-Night and Aachen Day-Night v1.1

1. You need to download and configure the dataset according to [local_feature_evaluation](https://github.com/tsattler/visuallocalizationbenchmark/tree/master/local_feature_evaluation).

2.Configure the `../config/extract_aachen.yaml` file before the extraction point.

```
cd eval_Aachen
python extract_aachen.py
```
This is an extraction script we use ourselves for reference only. You may encounter some difficulties during the usage process, but its main structure is very clear. You can customize and adjust it to suit your habits.

3. The evaluation shall also be conducted in accordance with the guidelines provided by [local_feature_evaluation](https://github.com/tsattler/visuallocalizationbenchmark/tree/master/local_feature_evaluation).

## Evaluation on ETH

1. Download the [data](https://github.com/ahojnnes/local-feature-evaluation/blob/master/INSTRUCTIONS.md)

2. Configure the `../config/extract_ETH.yaml` file before the extraction point.

```
cd eval_ETH
python extract_ETH.py
```

3. evaluation

```
python reconstruction_pipeline.py
```

# Citation

```
@ARTICLE{PG2SD,
  author={Z. Li, J. Cao, Q. Hao, H. Yao and Y. Wang},
  journal={IEEE Transactions on Multimedia}, 
  title={Progressive Gradient-Guided Self-Distillation Keypoint Detection}, 
  year={2025},
  doi={10.1109/TMM.2025.3607731}}
```
For detailed citations, please refer to the IEEE Xplore.

# Acknowledgment
This project has to some extent been informed by the efforts of several open-source projects, with [D2-Net](https://github.com/mihaidusmanu/d2-net/tree/master), [R2D2](https://github.com/naver/r2d2/tree/master?tab=readme-ov-file) and [ASLFeat](https://github.com/lzx551402/ASLFeat?tab=readme-ov-file) being part of the references. We are grateful for their contributions to the open-source community and encourage users of our project to also consider citing these projects when utilizing our code. Please be aware that while this project is governed by the [MIT](LICENSE), the use of the referenced projects' code must comply with their respective licenses. Users are advised to review these licenses to ensure proper compliance and understand that they are solely responsible for any legal implications arising from the use of the code. We appreciate your respect for the intellectual property rights of all contributors and recommend seeking legal counsel if you have any questions regarding licensing.
