## Pytorch examples for basic ASR

* `compute-posterior.py` gives outputs(posteriors) of the pytorch model which feed the command `latgen-faster-mapped` in [kaldi](https://github.com/kaldi-asr/kaldi).
* `decode.sh` combines kaldi command and pytorch model, generating the recognized lattice and scoring the WER/CER.
* `model.py` implements some basic acoustic models(CNN/DNN/RNN/TDNN/ResNet……)
* `data/dataset.py` is a simple wrapper of training corpus for DNN/CNN.
* `prepare_*.py` are some scripts for data pre-processing.
* updating……

### TIMIT(test set)

|                  MODEL                  |  PER  |
| :-------------------------------------: | :---: |
|            DNN(3X1024) + BN             | 25.3% |
|             DNN(4X512) + BN             | 24.1% |
|        DNN(4X512) + BN + Dropout        | 23.8% |
|       DNN(4X1024) + BN + Dropout        | 23.7% |
|        CNN(K10,P6) + DNN(2X512)         | 23.1% |
| CNN(K10,P6) + DNN(2X512) + BN + Dropout | 22.7% |
|        GRU(3X256) + BN + Dropout        | 23.3% |
|        GRU(4X256) + BN + Dropout        | 22.9% |
|        GRU(3X512) + BN + Dropout        | 22.7% |

### THCHS30(test set)
|                       MODEL                  |   WER   |
| :------------------------------------------: |  :---:  |
| CNN(K10,P6,C128) + DNN(2X512) + BN + Dropout |  25.44% |
| CNN(K8,P6,C256) + DNN(2X512) + BN + Dropout  |  24.82% |
|       GRU(3X512,T=20) + BN + Dropout         |  26.64% |
|      GRU(3X512,T=100) + BN + Dropout         |  24.66% |
|      GRU(10X256,T=100) + BN + Dropout        |  24.39% |
|     Res-GRU(10X256,T=100) + BN + Dropout     |  24.24% |
|        DNN(3X1024) + BN + Dropout            |  23.54% |
