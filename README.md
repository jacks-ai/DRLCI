# DRLCI: Dual-view Representation Learning Framework for Predicting Drug-Gene Interactions via Causal Intervention
<img width="1713" height="1057" alt="image" src="https://github.com/user-attachments/assets/3e4dc5b9-7368-4df1-8250-f643cfa0db07" />

## 1. Requirements

The main experiment (`Code/Main.py`) requires the following dependencies:

```
Python                        3.10.4
PyTorch                       1.11.0
torch-sparse                  0.6.13
NumPy                         1.22.3
pandas                        1.5.2
SciPy                         1.7.3
Matplotlib                    3.5.3
scikit-learn                  1.1.3
```

`torch-sparse` contains compiled extensions, so install a wheel that matches
the installed PyTorch and CUDA versions. Weights & Biases (`wandb`) is optional;
the current main experiment continues normally when it is not installed.

## 2. Datasets

We utilize three datasets to evaluate DGCL: DrugBank, DGIdb, and LINCS L1000. The DrugBank and DGIdb datasets are divided into training and testing sets with an 8:2 ratio. LINCS L1000 is used as an external testing set.

## 3. Training and Evaluation

Please use the following command:

* DrugBank
```bash
python Main.py --data DrugBank
```
* DGIdb
```bash
python Main.py --data DGIdb
```
* LINCS L1000
```bash
python Main.py --data LINCS 
