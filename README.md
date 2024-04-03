# Deep learning phase recovery (DLPR)

Kaiqiang Wang and Edmund Y. Lam, "Harnessing Data and Physics for Deep Learning Phase Recovery", [\[Preprint\]](https://arxiv.org/abs/2404.01360).

## Environment requirements
- python == 3.8.18
- pytorch == 2.0.0
- matplotlib == 3.7.2
- numpy == 1.24.3
- opencv_python == 4.8.1.78
- scipy == 1.12.0
- scikit-image == 0.21.0
- tqdm == 4.65.0

The following software is recommended:

- CUDA == 11.7
-  cuDNN == 8.5.0

## File structure
```
DLPR
├── dataset_generation.py
├── train.py
├── one_infer.py
├── multiple_infer.py
├── functions
│   ├── data_read.py
│   ├── network.py
│   ├── propagation.py
│   └── train_func.py
├── datasets
│   ├── raw_images
│   │   ├── ...
│   ├── train_in
│   │   ├── ...
│   ├── train_gt
│   │   ├── ...
│   ├── test_in
│   │   ├── ...
│   └── test_gt
│       ├── ...
├── models_and_results
│   ├── model_weights
│   │   ├── ...
│   ├── results_DD_tPD_CD
│   │   ├── ...
│   └── results_uPD_tPDr
└──     ├── ...

```
| Folders or files | Use |  
|  :----  | :----  |
| `dataset_generation.py` | generate paired hologram-phase datasets |  
| `train.py` | train the neural network in the strategies of DD, tPD, and CD | 
| `one_infer.py` | infer the neural network in the strategies of DD, tPD, or CD | 
| `multiple_infer.py` | infer the neural network in the strategies of uPD or tPDr | 
| `/functions/` | contains functions to be called | 
| `/datasets/raw_images/` | contains raw images used for dataset generation | 
| `/datasets/train_in/`| contains the generated holograms for network training  | 
| `/datasets/train_gt/`| contains the generated phases for network training | 
| `/datasets/test_in/`| contains the generated holograms for network inference  | 
| `/datasets/test_gt/`| contains the generated phases for network inference  | 
| `/models_and_results/model_weights/`|  contains the trained network weights and training process | 
| `/models_and_results/results_DD_tPD_CD/`|  contains the inference results of DD, tPD, and CD.  | 
| `/models_and_results/results_uPD_tPDr/`|  contains the inference results of uPD and tPDr| 

## Step 1: dataset generation
- Download image datasets such as [ImageNet](https://www.image-net.org/), [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html), [COCO](https://cocodataset.org/), [LFW](https://vis-www.cs.umass.edu/lfw/), and [MNIST](https://yann.lecun.com/exdb/mnist/). Alternatively,  [randimage](https://pypi.org/project/randimage/) can be used to generate random unnatural images.
- Put the downloaded or generated images in the folder `/datasets/raw_images/`.
- Run `dataset_generation.py` to generate paired hologram-phase datasets.
```sh
python dataset_generation.py
```
- Then, hologram-phase datasets are generated in the folders `/datasets/train_in/`, `/datasets/train_gt/`, `/datasets/test_in/`, and `/datasets/test_gt/`.

## Step 2: network training
- Run `train.py` to train the neural network in the strategy of dataset-driven(DD), trained physics-driven (tPD), or co-driven (CD).
```sh
python train.py -s 'DD'
```
```sh
python train.py -s 'tPD'
```
```sh
python train.py -s 'CD'
```
- Then, `.pth`, `.csv`, and `.png` are saved in the folder `/models_and_results/model_weights/`.

## Step 3: network inference
### Step 3.1: inference for DD, tPD, and CD strategies.
- Run `one_infer.py`.
```sh
python one_infer.py -s 'DD'
```
```sh
python one_infer.py -s 'tPD'
```
```sh
python one_infer.py -s 'CD'
```
- Then, inference results are saved in the folder `/models_and_results/results_DD_tPD_CD/`.

### Step 3.2: inference for the strategies of untrained physics-driven (uPD) and trained physics-driven with refinement (tPDr).
- Run `multiple_infer.py`.
```sh
python multiple_infer.py -s 'uPD'
```
Note that _python multiple_infer.py -s 'uPD'_ can be run before Step 2.
```sh
python multiple_infer.py -s 'tPDr'
```
- Then, inference results are saved in the folder `/models_and_results/results_uPD_tPDr/`.
