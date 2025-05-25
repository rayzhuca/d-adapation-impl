# D-Adaptation Implementation

This repository contains an implementation for D-Adaptation and also
as well as a standard implementation of a CNN model for solving MNIST,
which utilizes D-Adaptation.

The two relevant files are:

-   `d_adam`: Implementation of the Adam variant of D-Adaptation
-   `train_mnist`: CNN Model for MNIST

The code closely follows the official implementation.

## Results

Output for standard Adam

```
--- Start Training ---
Epoch 1: Accuracy = 0.9699, Loss = 0.1021
Epoch 2: Accuracy = 0.9896, Loss = 0.0315
Epoch 3: Accuracy = 0.9937, Loss = 0.0201
Epoch 4: Accuracy = 0.9944, Loss = 0.0182
Epoch 5: Accuracy = 0.9956, Loss = 0.0131
Epoch 6: Accuracy = 0.9965, Loss = 0.0111
Epoch 7: Accuracy = 0.9968, Loss = 0.0103
Epoch 8: Accuracy = 0.9974, Loss = 0.0082
Epoch 9: Accuracy = 0.9980, Loss = 0.0064
Epoch 10: Accuracy = 0.9972, Loss = 0.0083

--- Start Testing ---

Training Set Metrics:
Macro Accuracy:  0.9971
Macro Precision: 0.9971
Macro Recall:    0.9971
Macro F1-score:  0.9971

Test Set Metrics:
Macro Accuracy:  0.9913
Macro Precision: 0.9913
Macro Recall:    0.9911
Macro F1-score:  0.9912
```

Output for D-Adapted Adam

```
--- Start Training ---
Epoch 1: Accuracy = 0.9617, Loss = 0.1163
Epoch 2: Accuracy = 0.9921, Loss = 0.0263
Epoch 3: Accuracy = 0.9950, Loss = 0.0167
Epoch 4: Accuracy = 0.9959, Loss = 0.0126
Epoch 5: Accuracy = 0.9967, Loss = 0.0107
Epoch 6: Accuracy = 0.9965, Loss = 0.0104
Epoch 7: Accuracy = 0.9972, Loss = 0.0086
Epoch 8: Accuracy = 0.9968, Loss = 0.0090
Epoch 9: Accuracy = 0.9980, Loss = 0.0059
Epoch 10: Accuracy = 0.9977, Loss = 0.0069

--- Start Testing ---

Training Set Metrics:
Macro Accuracy:  0.9985
Macro Precision: 0.9985
Macro Recall:    0.9985
Macro F1-score:  0.9985

Test Set Metrics:
Macro Accuracy:  0.9923
Macro Precision: 0.9922
Macro Recall:    0.9922
Macro F1-score:  0.9922
```
