# Continuous-Time and Multi-Level Graph Representation Learning for Origin-Destination Demand Prediction
This is an implementation of CMOD: [Continuous-Time and Multi-Level Graph Representation Learning for Origin-Destination Demand Prediction, KDD2022].
## Environment
- Python 3.6.12
- PyTorch 1.6.0
- NumPy 1.19.1
- tqdm 4.51.0
## Dataset
Step 1： Download the processed dataset from [Baidu Yun](https://pan.baidu.com/s/1guYb1Mxtrweucsdd2ZucnQ) (Access Code:luck).

Step 2: Put them into ./data directories.
## Train command
    # Train with NYTaxi
    python train_OD.py --data=NYTaxi
