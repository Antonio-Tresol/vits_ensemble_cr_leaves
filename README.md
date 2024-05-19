## Transformers Unidos: Eficacia De los Modelos Ensemble-ViT en Clasificación Automática de Flora Costarricense 🌿

This repository contains the code used to run the experiments for the paper "Transformers Unidos: Eficacia De los Modelos Ensemble-ViT en Clasificación Automática de Flora Costarricense 🌿". The paper explores the effectiveness of Vision Transformers ensemble models in comparison to an ensemble model with convolutional networks for the recognition of Costa Rican flora.

## Contents

* **checkpoints:** Directory to store model checkpoints.
* **conv:** Code for convolutional models.
* **experiments:** Scripts to run the experiments.
* **metrics:** csv with the data from the experiments as well as jupyter notebooks analysing the rusults.
* **vit:** Code for Vision Transformer models.
* **configuration.py:** Configuration constants for the experiments.
* **data_modules.py and datasets.py:** Code for data loading and preprocessing.
* **helper_functions.py:** Helper functions for the experiments.
* **README.md:** This file.

## Usage

1. Clone the repository:
```
git clone https://github.com/Antonio-Tresol/vits_ensemble_cr_leaves.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Get dataset from [source](https://www.clei.org/cleiej/index.php/cleiej/article/view/413 )
4. Update the `configuration.py` file with the dataset path and other desired settings.
5. Run the experiments using the scripts in the `experiments` directory.

## Results

The results of the experiments are presented in the paper. The code in this repository can be used to reproduce the experiment.

## Contact

If you have any questions, please contact Antonio Badilla-Olivas at anthonny.badilla@ucr.ac.cr.

