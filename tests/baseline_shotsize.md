# Shot-Size Classification Baseline Report

## 1. Experiment

**Task:** classify movie frames by shot size  
**Evaluation script:** `embed_shotsize.py`  
**Model:** logistic regression over image embeddings  

The experiment evaluates seven shot-size categories:

- close up
- extreme close up
- extreme wide
- medium
- medium close up
- medium wide
- wide

## 2. Similarity and leakage diagnostic

The maximum similarity between each test image and the training set was:

| Statistic | Value |
|---|---:|
| Median similarity | 0.822 |
| 95th-percentile similarity | 0.892 |
| Test images with similarity > 0.95 | 0 of 485 |

## 3. Validation hyperparameter selection

The regularization parameter `C` was selected using validation macro-F1.

| `C` | Validation macro-F1 |
|---:|---:|
| 0.1 | 0.582 |
| 1 | 0.653 |
| 10 | 0.677 |
| 100 | **0.678** |

**Selected value:** `C = 100`

## 4. Test-set performance

The final model was evaluated on the complete test set and on the test set
after removing near-duplicate samples.

| Test-set condition | Accuracy | Macro-F1 | Number of samples |
|---|---:|---:|---:|
| Strict evaluation | 0.670 | 0.669 | 470 |
| Permissive evaluation | 0.678 | — | 470 |
| Strict evaluation, without near-duplicates | 0.670 | 0.669 | 470 |
| Permissive evaluation, without near-duplicates | 0.678 | — | 470 |

The strict and near-duplicate-filtered results are identical in the reported
evaluation.

## 5. Confusion matrix

Rows represent the **true** class and columns represent the **predicted**
class. Values are sample counts.

| True class \ Predicted class | Close up | Extreme close up | Extreme wide | Medium | Medium close up | Medium wide | Wide | Total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Close up | 47 | 9 | 0 | 0 | 8 | 0 | 0 | 64 |
| Extreme close up | 16 | 43 | 1 | 0 | 3 | 1 | 1 | 65 |
| Extreme wide | 0 | 1 | 54 | 0 | 0 | 0 | 15 | 70 |
| Medium | 1 | 2 | 0 | 39 | 14 | 11 | 2 | 69 |
| Medium close up | 11 | 4 | 0 | 12 | 38 | 1 | 0 | 66 |
| Medium wide | 0 | 0 | 1 | 9 | 0 | 49 | 9 | 68 |
| Wide | 0 | 0 | 12 | 0 | 0 | 11 | 45 | 68 |
| **Total** | **75** | **59** | **68** | **60** | **63** | **73** | **72** | **470** |

## 6. Summary

The best validation result was obtained with `C = 100`, reaching a macro-F1 of
0.678. On the test set, the model achieved an accuracy of 0.670 and a
macro-F1 of 0.669 under strict evaluation. The confusion matrix shows the
strongest diagonal performance for `extreme wide` and `medium wide`, while
confusions are more frequent between visually similar shot sizes such as
`close up` and `extreme close up`, and between `medium` and `medium close up`.
