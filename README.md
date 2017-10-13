## Notes

### experience(TIMIT test set)
|                  MODEL                  |  PER  |
| :-------------------------------------: | :---: |
|            DNN(3X1024) + BN             | 25.3% |
|             DNN(4X512) + BN             | 24.1% |
|        DNN(4X512) + BN + Dropout        | 23.8% |
|       DNN(4X1024) + BN + Dropout        | 23.7% |
|        CNN(K10,P6) + DNN(2X512)         | 23.1% |
| CNN(K10,P6) + DNN(2X512) + BN + Dropout | 22.7% |
|        RNN(3X256) + BN + Dropout        | 23.3% |
|        RNN(4X256) + BN + Dropout        | 22.9% |
|        RNN(3X512) + BN + Dropout        | 22.7% |

### experience(THCHS30 test set)
|                  MODEL                  |   WER   |
| :-------------------------------------: |  :---:  |
| CNN(K10,P6) + DNN(2X512) + BN + Dropout |  25.44% |
|       DNN(3X1024) + BN + Dropout        |  23.54% |
