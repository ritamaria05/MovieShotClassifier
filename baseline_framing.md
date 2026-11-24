# Framing classification: baseline and experiments

## 1. Objective and evaluation

The goal is to classify a movie frame's framing into seven categories: **2 shot, 3 shot, establishing shot, group shot, insert, over the shoulder** and **single**. The experiments began with pretrained SigLIP image features and compared classification heads, multi-task learning, a larger backbone, class balancing and fine-tuning.

The current split contains 6,043 training images, 955 validation images and 445 test images. The reported **strict accuracy** and **macro-F1** use the 391 single-label test images; the remaining 54 test images have multiple labels. Macro-F1 averages the F1 scores of the classes equally. A separate **lenient accuracy** counts a single prediction as correct when it matches any recorded label.

The training/validation split is grouped by film. The dataset preparation excludes multi-label training examples, while retaining multi-label test examples. The original task was treated as single-label, but framing categories can overlap; this choice is a limitation if the intended application needs to return all applicable labels.

## 2. Initial baseline: frozen SigLIP and logistic regression

The initial experiment used frozen `google/siglip-base-patch16-224` embeddings and a logistic regression head. The head's settings were selected using validation macro-F1. Its strict test accuracy was **76.21%** and macro-F1 was **75.10%**.

Rows in the following matrices are actual labels; columns are predicted labels.

| Actual \ Predicted | 2 shot | 3 shot | establishing shot | group shot | insert | over the shoulder | single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 shot | 47 | 2 | 0 | 0 | 0 | 9 | 0 |
| 3 shot | 3 | 35 | 5 | 8 | 1 | 10 | 1 |
| establishing shot | 2 | 1 | 41 | 4 | 2 | 1 | 0 |
| group shot | 0 | 4 | 2 | 32 | 1 | 4 | 1 |
| insert | 0 | 0 | 6 | 0 | 60 | 1 | 3 |
| over the shoulder | 1 | 4 | 2 | 1 | 0 | 28 | 2 |
| single | 1 | 0 | 5 | 0 | 3 | 3 | 55 |

We investigated whether the result could be improved through multi-task learning, a different head, a different backbone, class balancing or fine-tuning.

## 3. Multi-task learning

Multi-task learning (MTL) trains related tasks using a shared representation. Here, we compared framing and shot-size classification in a shared model against single-task models. MTL can improve generalization when useful information is shared between tasks; see this [overview](https://www.geeksforgeeks.org/deep-learning/multi-task-learningmtl-for-deep-learning/).

| Task | Single-task macro-F1 | MTL macro-F1 |
| --- | ---: | ---: |
| Framing | 0.770 | 0.760 |
| Shot size | 0.703 | 0.707 |

MTL did not substantially improve either task. The differences were approximately one standard deviation, so they may reflect run-to-run variation.

Possible explanations, not established causes:

- **Different cues:** shot size concerns subject scale, while framing often concerns the number and arrangement of people. The tasks can be related but may not share enough useful cues in this setup.
- **Limited overlap:** only about 230 images carried both labels; most images updated only one task head.
- **Frozen backbone:** SigLIP's representation could not adapt to both tasks; the shared trainable component was a small MLP layer.
- **Possible feature bottleneck:** distinctions such as 3 shot versus group shot or over the shoulder versus 2 shot can depend on spatial layout. A global 224-pixel embedding might not preserve all relevant detail.

## 4. Comparing classification heads

The initial choice of logistic regression had not been justified against alternatives. An evaluator compared a majority-class baseline, logistic regression, RBF SVM, k-nearest neighbors (kNN) and an MLP, using film-grouped cross-validation for selection.

| Framing head | Strict test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Majority | 0.171 | 0.042 |
| Logistic regression, C=3 (selected by cross-validation) | 0.765 | 0.754 |
| RBF SVM | 0.767 | 0.760 |
| kNN | 0.604 | 0.570 |
| MLP | **0.795** | **0.781** |

Although the MLP scored best on the test set in this table, logistic regression was selected by film-grouped cross-validation. Selection using validation rather than test results keeps the test set separate from model selection. The MLP in this evaluator and the separately trained MLP below used different training/selection runs, so their rounded metrics differ.

### Frozen base SigLIP + MLP baseline

An MLP, or multilayer perceptron, is a feedforward neural network with an input layer, one or more hidden layers and an output layer. Here, the MLP classifies SigLIP embeddings rather than processing raw pixels itself.

The separately trained MLP achieved **79.28% strict accuracy** and **77.87% macro-F1**, with 310 correct predictions out of 391:

| Actual \ Predicted | 2 shot | 3 shot | establishing shot | group shot | insert | over the shoulder | single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 shot | 51 | 2 | 0 | 0 | 0 | 4 | 1 |
| 3 shot | 5 | 42 | 2 | 9 | 1 | 3 | 1 |
| establishing shot | 2 | 1 | 36 | 7 | 3 | 0 | 2 |
| group shot | 0 | 6 | 0 | 35 | 1 | 1 | 1 |
| insert | 0 | 0 | 4 | 0 | 62 | 0 | 4 |
| over the shoulder | 3 | 5 | 2 | 2 | 0 | 24 | 2 |
| single | 1 | 0 | 2 | 0 | 3 | 1 | 60 |

Against the original logistic regression matrix, the MLP's largest gain was on **3 shot**, while it missed more **establishing shot** and **over the shoulder** examples. For over the shoulder, precision increased from **50.0% to 72.7%**, while recall decreased from **73.7% to 63.2%**.

The MLP's nonlinear decision boundaries may fit relationships in the frozen feature space better than a linear head. Differences in class weighting, training and sampling may also contribute, so the results do not prove which factor caused the gain.

## 5. Larger backbone: SO400M

We replaced base SigLIP with `google/siglip-so400m-patch14-384`, a larger backbone accepting 384-pixel inputs. Embeddings were recomputed into model-specific `.npy` files, leaving the original cache intact. The SO400M embedding run took about **16 hours**; the earlier base-model embedding run was estimated at roughly two hours.

The embedding script uses `im.draft('RGB', (448, 448))` for reduced JPEG decoding. Higher model input resolution cannot recover detail absent from the source image, so original dimensions and decoding behavior matter when interpreting the comparison.

### SO400M: correct predictions by class

| Actual class | Logistic regression | MLP |
| --- | ---: | ---: |
| 2 shot | 51 / 58 | 53 / 58 |
| 3 shot | 40 / 63 | 48 / 63 |
| establishing shot | 42 / 51 | 38 / 51 |
| group shot | 32 / 44 | 32 / 44 |
| insert | 57 / 70 | 60 / 70 |
| over the shoulder | 22 / 38 | 18 / 38 |
| single | 57 / 67 | 58 / 67 |

The SO400M MLP performed better overall than SO400M logistic regression, particularly on 3 shot; logistic regression did better on establishing shot and over the shoulder.

## 6. Class-balanced MLP

The existing MLP did not explicitly balance classes. We tested inverse-frequency sample weighting on the existing embeddings, using the same MLP grid, seed and validation-selection approach. Weighting gives less frequent classes more influence during training; it does not add images or new visual information.

| Model | Strict accuracy | Macro-F1 |
| --- | ---: | ---: |
| Base SigLIP + MLP | **79.3%** | **77.9%** |
| SO400M + MLP | 78.5% | 76.5% |
| Base SigLIP + balanced MLP | 78.0% | 76.9% |
| SO400M + balanced MLP | 77.5% | 76.3% |

Balancing did not improve overall metrics in these runs. With base SigLIP, **over-the-shoulder recall** rose from **63.2% to 73.7%**, while precision fell from **72.7% to 57.1%**. Thus, the model found more examples of that class but produced more false positives. A weighted PyTorch head or a balanced sampler were also identified as possible alternatives, but the recorded experiment used weighted scikit-learn training.

## 7. Fine-tuning base SigLIP

Fine-tuning continues training a pretrained model on the framing task, allowing its image representation to adapt. The implementation in [finetune_framing.py](../scripts/finetune_framing.py) follows two stages:

1. Train a new MLP head while SigLIP remains frozen; retain the head checkpoint with the best validation macro-F1.
2. Restore that checkpoint, unfreeze the last **two** vision encoder layers and train both the head and those layers. The head learning rate is **0.0001**, while the backbone learning rate is **0.00001**. Other backbone layers remain frozen.

After each epoch, validation macro-F1 is checked. Training stops after three consecutive epochs without improvement over the best score, then restores the best checkpoint. Fine-tuning requires processing images during training; cached embeddings cannot be reused after the backbone changes. A related earlier implementation for shot size is in [finetune_shotsize.py](../scripts/finetune_shotsize.py).

### Recorded run

| Fine-tuning epoch | Train macro-F1 | Validation macro-F1 |
| ---: | ---: | ---: |
| 1 | 0.6751 | 0.6862 |
| **2** | 0.7953 | **0.7364** |
| 3 | 0.8618 | 0.7337 |
| 4 | 0.9123 | 0.7358 |
| 5 | 0.9608 | 0.7245 |

The best checkpoint was epoch 2. Early stopping occurred after epoch 5 because epochs 3–5 did not beat the best validation score. The widening train/validation gap suggests overfitting. This run used seed 42; multiple-seed validation remains future work.

| Model | Strict test accuracy | Test macro-F1 |
| --- | ---: | ---: |
| Frozen base SigLIP + MLP | 79.28% | 77.87% |
| Fine-tuned base SigLIP + MLP | **80.05%** | **78.99%** |

The fine-tuned model correctly classified 313 of 391 single-label test images. Its test confusion matrix was:

| Actual \ Predicted | 2 shot | 3 shot | establishing shot | group shot | insert | over the shoulder | single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 shot | 51 | 1 | 1 | 0 | 1 | 4 | 0 |
| 3 shot | 5 | 39 | 3 | 9 | 0 | 4 | 3 |
| establishing shot | 3 | 0 | 39 | 4 | 3 | 0 | 2 |
| group shot | 2 | 5 | 1 | 35 | 0 | 0 | 1 |
| insert | 1 | 0 | 4 | 0 | 61 | 0 | 4 |
| over the shoulder | 3 | 1 | 2 | 2 | 0 | 27 | 3 |
| single | 1 | 0 | 2 | 0 | 3 | 0 | 61 |

The frozen MLP baseline refits its head on train plus validation, while the fine-tuned experiment evaluates its validation-selected checkpoint directly. The procedures are therefore not identical.

## 8. Consolidated results

All rows below use the 391 single-label test examples. The separate head-comparison experiment in section 4 has its own training and selection procedure.

| Model | Correct / 391 | Strict accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Fine-tuned base SigLIP + MLP | **313** | **80.05%** | **78.99%** |
| Frozen base SigLIP + MLP | 310 | 79.28% | 77.87% |
| Frozen SO400M + MLP | 307 | 78.52% | 76.54% |
| Frozen base SigLIP + balanced MLP | 305 | 78.01% | 76.88% |
| Frozen SO400M + balanced MLP | 303 | 77.49% | 76.3% |
| Frozen SO400M + logistic regression | 301 | 76.98% | 75.44% |
| Frozen base SigLIP + logistic regression | 298 | 76.21% | 75.10% |

Fine-tuning produced the highest recorded strict result, but the difference from the frozen baseline is small and based on one fine-tuning seed. These results do not establish an 80% upper limit or prove that model size is irrelevant.

## 9. Possible limitations

- **Visually close categories:** deciding whether background people count as subjects or whether composition matters more than count can make 2 shot, 3 shot, group shot and over the shoulder difficult to distinguish. The confusion matrix identifies disagreements, not their causes.
- **Single-label formulation:** the test set has 54 multi-label images, but the model returns one label and strict accuracy excludes those images. This demonstrates overlap in the task, not that every single-label error is ambiguous.
- **Uneven class coverage:** the training split includes 1,100 single examples, 903 3 shot examples and 458 over-the-shoulder examples. Fewer examples may reduce coverage of visual variation; balancing improved over-the-shoulder recall but lowered its precision and overall accuracy.
- **Representation:** the classifier uses a pooled image feature, with no explicit person count, bounding boxes or shoulder positions. SigLIP may encode such cues implicitly, but the experiment does not directly supervise them.
- **Fine-tuning setup:** only the last two vision layers, one default configuration and one seed have been evaluated. The observed train/validation gap suggests overfitting.

## 10. Possible next steps

1. **Review validation errors and labeling rules.** Inspect misclassified images and class coverage, especially among over the shoulder, 3 shot and group shot. Distinguish model errors from genuinely ambiguous examples; do not assume labels are wrong.
2. **Evaluate further fine-tuning runs.** Compare seeds and settings using film-grouped validation. Benchmark hardware first: full image-based fine-tuning is expensive on CPU.
3. **Test person and spatial information.** A detector could provide person counts, bounding-box sizes and positions, or relative positions and overlap. Combine these with SigLIP features and compare against the embedding-only baseline. Detectors can miss partially visible people or shoulders, and count alone will not resolve every framing category.
4. **Consider multi-label classification if it matches the goal.** An image could receive both 2 shot and over the shoulder. This changes the task and requires separate multi-label metrics and a clear evaluation protocol. First check whether annotations are complete: an unannotated class is not necessarily a confirmed negative.

Repeatedly comparing models on the same test set can bias selection. Use validation to choose future approaches and, if possible, reserve a fresh held-out test set for the final comparison.
