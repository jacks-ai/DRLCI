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
```

## 4. web tool

http://8.148.158.242:8501/

Single Prediction : Upon opening the web link, users land on the Single Prediction home page. Here, users can enter a Drug ID and a Gene ID to obtain the interaction predicted by the DRLCI framework. The page displays the predicted probability distribution across interaction categories, enabling convenient inspection of individual drug–gene pairs.

Batch Prediction: Users can select an evaluation dataset and perform DRLCI predictions for all drug-gene pairs in the selected dataset. The page summarizes the prediction results, including the numbers of drug-gene pairs, unique drugs, unique genes, and predicted interaction types. The complete prediction results can also be downloaded as a CSV file. In addition, the page automatically constructs an interactive drug-gene interaction knowledge graph based on the top-ranked interaction predicted by DRLCI for each pair. The graph supports zooming, node dragging, and hover-based inspection of entity information, predicted interaction types, and prediction probabilities.

The detailed operational steps for using the web platform are outlined below:
| Step | Feature Name | Description |
| :---: | :--- | :--- |
| **1** | **Search Interaction** | After entering a Drug ID and a Gene ID, the DRLCI framework predicts the interaction type and displays the corresponding prediction probability for the selected drug–gene pair. |
| **2** | **Batch Prediction →** | Navigates from the Single Prediction page to the Batch Prediction page, where users can select a predefined evaluation dataset for large-scale prediction. |
| **3** | **Run Batch Prediction** | Applies the DRLCI framework to all drug–gene pairs in the selected dataset and presents a summary of the results, including the numbers of pairs, drugs, genes, and predicted interaction types. |
| **4** | **Download Prediction Results** | Exports the complete batch prediction results as a CSV file, including the Drug ID, Gene ID, top-ranked interaction type, and corresponding prediction probability. |
| **5** | **Knowledge Graph**<br>*(automatically displayed)* | Visualizes the batch prediction results as an interactive drug–gene interaction graph. Users can zoom, drag nodes, and hover over nodes or edges to inspect entity information, predicted interaction types, and prediction probabilities. |





