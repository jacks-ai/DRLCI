# DRLCI: Dual-view Representation Learning Framework for Predicting Drug-Gene Interactions via Causal Intervention
<img width="1713" height="1057" alt="image" src="https://github.com/user-attachments/assets/3e4dc5b9-7368-4df1-8250-f643cfa0db07" />

## 1. Project Structure

The code and data are organized as follows:

```
DRLCI-main/
├── README.md                            # Project documentation: requirements, datasets, 
├                                          reproduction commands, baselines, results, web tool
├── Code/                                # Source code
│   ├── Main.py                          # Main entry: training, evaluation, cold-start, and metric export
│   ├── Model_sparse.py                  # DRLCI model: graph encoder, gated structure-text fusion, 
│   │                                      contrastive loss
│   ├── DataHandler.py                   # Data loading and interaction graph construction (train/test sets)
│   ├── Params.py                        # Hyperparameters and command-line arguments
│   ├── BioLinkBERT-large.py             # Downloads the pretrained BioLinkBERT-large model (text encoder)
│   ├── train_biolinkbert_classifier.py  # Fine-tunes BioLinkBERT with an MLP head 
│                                           for interaction-type classification
│   └── Utils/
│       ├── Utils.py                     # Shared utilities: contrastive loss, cross-entropy, regularization
│       └── TimeLogger.py                # Console logging helpers
└── Data/
    ├── DGIdb/                           # DGIdb dataset
    ├── DrugBank/                        # DrugBank dataset
    └── LINCS/                           # LINCS L1000 dataset (external validation)
```

## 2. Requirements

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

## 3. Datasets

We utilize three datasets to evaluate DRLCI: DrugBank, DGIdb, and LINCS L1000. The statistics of the three datasets are summarized below.

| Statistics | DrugBank | DGIdb | LINCS L1000 |
| :--- | :---: | :---: | :---: |
| # of drugs | 425 | 1185 | 187 |
| # of genes | 11284 | 1664 | 3769 |
| # of interactions | 80924 | 11366 | 20610 |
| # of types | 2 | 14 | 2 |

## 4. Training and Evaluation

Run the following commands from the `Code/` directory (where `Main.py` is located):

### Main Experiment
* DrugBank
```bash
python Main.py --data DrugBank
```
* DGIdb
```bash
python Main.py --data DGIdb
```
The ACC for each relationship type in the DGIdb dataset, together with the normalized confusion matrices for the DGIdb and DrugBank datasets, are exported automatically once the main experiment finishes.

### Generalization Assessment
* LINCS L1000
```bash
python Main.py --data LINCS
```
* Drug Cold-Start
```bash
python Main.py --data DGIdb --cold true --cold_entity drug
```
* Gene Cold-Start
```bash
python Main.py --data DGIdb --cold true --cold_entity gene
```

## 5. Baselines

We compare DRLCI with five state-of-the-art drug–gene interaction prediction methods. Each baseline is run with its official implementation on the DrugBank and DGIdb datasets.

### CoSMIG

*Communicative subgraph representation learning for multi-relational inductive drug–gene interaction prediction (IJCAI 2022).*

```bash
git clone https://github.com/biomed-AI/CoSMIG.git
cd CoSMIG

# DrugBank (transductive)
CUDA_VISIBLE_DEVICES=0 python main.py --data-name DrugBank --testing --dynamic-train --dynamic-test --dynamic-val --save-results --max-nodes-per-hop 200
# DGIdb (inductive)
CUDA_VISIBLE_DEVICES=0 python main.py --data-name DGIdb --testing --mode inductive --dynamic-train --dynamic-test --dynamic-val --save-results --max-nodes-per-hop 200
```

### DGCL

*Prediction of multi-relational drug–gene interaction via dynamic hypergraph contrastive learning (Briefings in Bioinformatics, 2023).*

```bash
git clone https://github.com/wentao228/DGCL.git
cd DGCL

# DrugBank
python Main.py --data DrugBank --iteration 5
# DGIdb
python Main.py --data DGIdb --iteration 5
```

### AGDNGDA

*AGDNGDA: unraveling drug-associated genes with adaptive graph diffusion network (Journal of Chemical Information and Modeling, 2025).*

```bash
git clone https://github.com/altriavin/AGDNGDA.git
cd AGDNGDA

# Runs with the repository's default configuration
python main.py
```

### MNCL

*Predicting drug–gene interaction via multi-negative contrastive learning (Tsinghua Science and Technology, 2025).*

```bash
git clone https://github.com/hle341/MNCL.git
cd MNCL

# DrugBank
python Main.py --data DrugBank
# DGIdb
python Main.py --data DGIdb
```

### BiGvCL

*BiGvCL: bipartite graph-based cross-domain contrastive learning model for predicting drug–gene interactions.*

```bash
git clone https://github.com/heshida01/BiGvCL.git
cd BiGvCL

# DrugBank
python src/main.py --data drugbank --epoch 500 --gpu 0
# DGIdb
python src/main.py --data dgidb --epoch 500 --gpu 0
```

## 6. Experimental Results

We compare DRLCI with the baselines described above on the DGIdb and DrugBank datasets. All metrics are reported as the mean over five folds. The best result in each column is shown in **bold**, and the second best is <u>underlined</u>.

### DGIdb

| Model | ACC |  Precision   |    Recall    |      F1      | MCC | AUPRC | AUROC | *P*-Value |
| :--- | :---: |:------------:|:------------:|:------------:| :---: | :---: | :---: | :---: |
| DGCL | 0.929 |    0.931     |    0.930     |    0.930     | 0.902 | 0.871 | 0.992 | 6.15e-06 |
| AGDNGDA | 0.938 |    0.933     |    0.940     |    0.936     | 0.911 | 0.867 | 0.990 | 9.33e-04 |
| MNCL | <u>0.944</u> | <u>0.943</u> | <u>0.944</u> | <u>0.943</u> | <u>0.916</u> | <u>0.895</u> | <u>0.993</u> | 4.84e-04 |
| BiGvCL | 0.934 |    0.932     |    0.929     |    0.930     | 0.909 | 0.865 | 0.989 | 8.48e-06 |
| **DRLCI (Ours)** | **0.955** |  **0.955**   |  **0.954**   |  **0.954**   | **0.937** | **0.910** | **0.993** | — |

*Note: P-values are computed by comparing the ACC scores of DRLCI and each baseline across five folds using a paired two-tailed t-test.*

### DrugBank

| Model | ACC | Precision |    Recall    |      F1      | MCC | AUPRC | AUROC | *P*-Value |
| :--- | :---: | :---: |:------------:|:------------:| :---: | :---: | :---: |:---------:|
| DGCL | 0.699 | 0.698 |    0.701     |    0.699     | 0.392 | 0.718 | 0.732 | 6.48e-06  |
| AGDNGDA | 0.705 | 0.694 |    0.710     |    0.702     | 0.409 | 0.716 | 0.728 | 5.13e-05  |
| MNCL | 0.712 | 0.698 |  **0.761**   | <u>0.728</u> | <u>0.427</u> | 0.719 | 0.734 | 1.15e-04  |
| BiGvCL | 0.708 | <u>0.712</u> |    0.702     |    0.707     | 0.416 | <u>0.723</u> | **0.745** | 7.96e-04  |
| **DRLCI (Ours)** | **0.723** | **0.732** | <u>0.750</u> |  **0.741**   | **0.443** | **0.725** | <u>0.739</u> |     —     |

*Note: P-values are computed by comparing the ACC scores of DRLCI and each baseline across five folds using a paired two-tailed t-test.*

### Generalization

To assess generalization, we further evaluate DRLCI and the baselines under three settings: the external **LINCS L1000** dataset, **Drug Cold-Start** (interactions involving drugs unseen during training), and **Gene Cold-Start** (interactions involving genes unseen during training). ACC is reported, and DRLCI achieves the best performance under all three settings.

| Model | LINCS L1000 | Drug Cold-Start | Gene Cold-Start |
| :--- | :---: | :---: | :---: |
| DGCL | 0.616 | 0.649 | 0.813 |
| AGDNGDA | 0.617 | 0.717 | 0.811 |
| MNCL | <u>0.623</u> | 0.724 | <u>0.828</u> |
| BiGvCL | 0.622 | <u>0.729</u> | 0.807 |
| **DRLCI (Ours)** | **0.635** | **0.758** | **0.853** |




## 7. Web Tool

The DRLCI web tool is available at **http://8.148.158.242:8501/**.

**Single Prediction**: Upon opening the web link, users land on the Single Prediction home page. Here, users can enter a Drug ID and a Gene ID to obtain the interaction predicted by the DRLCI framework. The page displays the predicted probability distribution across interaction categories, enabling convenient inspection of individual drug–gene pairs.

**Batch Prediction**: Users can select an evaluation dataset and perform DRLCI predictions for all drug-gene pairs in the selected dataset. The page summarizes the prediction results, including the numbers of drug-gene pairs, unique drugs, unique genes, and predicted interaction types. The complete prediction results can also be downloaded as a CSV file. In addition, the page automatically constructs an interactive drug-gene interaction knowledge graph based on the top-ranked interaction predicted by DRLCI for each pair. The graph supports zooming, node dragging, and hover-based inspection of entity information, predicted interaction types, and prediction probabilities.

The detailed operational steps for using the web platform are outlined below:

| Step | Feature Name | Description |
| :---: | :--- | :--- |
| **1** | **Search Interaction** | After entering a Drug ID and a Gene ID, the DRLCI framework predicts the interaction type and displays the corresponding prediction probability for the selected drug–gene pair. |
| **2** | **Batch Prediction →** | Navigates from the Single Prediction page to the Batch Prediction page, where users can select a predefined evaluation dataset for large-scale prediction. |
| **3** | **Run Batch Prediction** | Applies the DRLCI framework to all drug–gene pairs in the selected dataset and presents a summary of the results, including the numbers of pairs, drugs, genes, and predicted interaction types. |
| **4** | **Download Prediction Results** | Exports the complete batch prediction results as a CSV file, including the Drug ID, Gene ID, top-ranked interaction type, and corresponding prediction probability. |
| **5** | **Knowledge Graph**<br>*(automatically displayed)* | Visualizes the batch prediction results as an interactive drug–gene interaction graph. Users can zoom, drag nodes, and hover over nodes or edges to inspect entity information, predicted interaction types, and prediction probabilities. |
