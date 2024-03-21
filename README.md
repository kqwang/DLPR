# Deep learning phase recovery (DLPR)

For the paper "Deep learning phase recovery: dataset-driven, physics-driven, or co-driven?" (under review)

## Environment requirements

- pytorch == 2.0.0
- matplotlib == 3.7.2
- numpy == 1.24.3
- opencv_python == 4.8.1.78
- scipy == 1.12.0
- scikit-image == 0.21.0
- tqdm == 4.65.0

The following software is recommended:
- python == 3.8.0
- CUDA == 10.1

## Step 1: dataset generation
- Download image datasets such as [ImageNet](https://www.image-net.org/), [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html), [COCO](https://cocodataset.org/), [LFW](https://vis-www.cs.umass.edu/lfw/), and [MNIST](https://yann.lecun.com/exdb/mnist/).
- Put the downloaded  images in the folder `/datasets/raw_images/`.
- Run `main_dataset_generation.py` to generate paired hologram-phase datasets.
```sh
python main_dataset_generation.py
```
- Then, hologram-phase datasets are generated in the folder `/datasets/train_in/`, `/datasets/train_gt/`, `/datasets/test_in/`, and `/datasets/test_gt/`.

## Step 2: network training
- Run `main_train.py` to train the neural network in the strategy of dataset-driven(DD), trained physics-driven (tPD), or co-driven (CD).
```sh
python main_train.py -s 'DD'
```
```sh
python main_train.py -s 'tPD'
```
```sh
python main_train.py -s 'CD'
```
- Then, `.pth`, `.csv`, and `.png` are saved in the folder `/models_and_results/model_weights/`.

## Step 3: network inference
### inference in DD, tPD, and CD.
- Run `main_single_inference.py`.
```sh
python main_single_inference.py -s 'DD'
```
```sh
python main_single_inference.py -s 'tPD'
```
```sh
python main_single_inference.py -s 'CD'
```
- Then, inference results are saved in the folder `/models_and_results/results_DD_tPD_CD/`.

### inference in untrained physics-driven (uPD), trained physics-driven with refinement (tPDr).
- Run `main_multiple_inferences`.
```sh
python main_multiple_inferences -s 'uPD'
```
Note that _python main_multiple_inferences -s 'uPD'_ can be run before Step 2.
```sh
python main_multiple_inferences -s 'tPDr'
```
- Then, inference results are saved in the folder `/models_and_results/results_uPD_tPDr/`.
