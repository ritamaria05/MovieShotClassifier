# Framing classification: experiment log

## Goal and baseline

We started with `embed_framing.py`, using frozen SigLIP embeddings and a logistic regression classifier to predict framing. The baseline was promising, but we wanted to improve it.

The original logistic regression confusion matrix was:

| Actual \ Predicted | 2 shot | 3 shot | establishing shot | group shot | insert | over the shoulder | single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 shot | 47 | 2 | 0 | 0 | 0 | 9 | 0 |
| 3 shot | 3 | 35 | 5 | 8 | 1 | 10 | 1 |
| establishing shot | 2 | 1 | 41 | 4 | 2 | 1 | 0 |
| group shot | 0 | 4 | 2 | 32 | 1 | 4 | 1 |
| insert | 0 | 0 | 6 | 0 | 60 | 1 | 3 |
| over the shoulder | 1 | 4 | 2 | 1 | 0 | 28 | 2 |
| single | 1 | 0 | 5 | 0 | 3 | 3 | 55 |

We explored three possible improvements:

1. Use **multi-task learning** to benefit from information shared with shot-size classification.
2. Compare **classification heads** instead of assuming logistic regression was the best choice.
3. Replace the **backbone** with a larger model.

## 1. Multi-task learning

Multi-task learning (MTL) trains multiple related tasks within one model using shared representations. It can improve generalization and reduce overfitting when the tasks share useful information. See this [overview of multi-task learning](https://www.geeksforgeeks.org/deep-learning/multi-task-learningmtl-for-deep-learning/).

In our experiment, MTL did not substantially improve either framing or shot-size classification:

| Task | Single-task macro-F1 | MTL macro-F1 |
| --- | ---: | ---: |
| Framing | 0.770 | 0.760 |
| Shot size | 0.703 | 0.707 |

The differences were approximately one standard deviation, so they may reflect run-to-run variation rather than a reliable improvement.

### Possible reasons

These are hypotheses, rather than established causes:

- **Different visual cues.** Shot size concerns subject scale, from close-up to wide. Framing often concerns the number and arrangement of people, such as a single, 2 shot, over-the-shoulder shot or group shot. The tasks can be related, but may not share enough useful cues for this model.
- **Limited label overlap.** Only about 230 images have both labels. Most images update just one task head, limiting the direct overlap between training examples.
- **Frozen features.** The SigLIP embeddings are frozen, so MTL cannot adapt the backbone's representation. The shared trainable component is a small MLP layer.
- **Possible feature bottleneck.** Errors such as over the shoulder versus 2 shot or 3 shot, and 3 shot versus group shot, depend on people and their spatial arrangement. The base model's 224-pixel global embedding might not preserve all the necessary detail. Person-count features, a stronger backbone or fine-tuning could be worth investigating.

## 2. Comparing classification heads

Logistic regression was our initial choice without a strong reason to prefer it over other heads. To test whether the head was a bottleneck, we built an evaluator for framing and shot size using a majority-class baseline, logistic regression, RBF SVM, k-nearest neighbors (kNN) and an MLP.

The recorded framing results were:

| Task | Head | Strict accuracy | Macro-F1 |
| --- | --- | ---: | ---: |
| Framing (n=391) | Majority | 0.171 | 0.042 |
| | Logistic regression, C=3 (selected) | 0.765 | 0.754 |
| | RBF SVM | 0.767 | 0.760 |
| | kNN | 0.604 | 0.570 |
| | MLP | **0.795** | **0.781** |

The evaluator selected logistic regression using film-grouped cross-validation, even though the MLP achieved the best *test* metrics in this table. Selection by validation rather than test results keeps the test set separate from model selection. These evaluator results and the separately trained MLP confusion matrix below come from different training/selection runs, so their rounded metrics are not identical.

### MLP framing experiment

An **MLP (multilayer perceptron)** is a feedforward neural network with an input layer, one or more hidden layers and an output layer. The layers are fully connected. Our MLP acts as a classifier on frozen SigLIP embeddings; it is not itself an image backbone.

The separately trained MLP produced this confusion matrix:

| Actual \ Predicted | 2 shot | 3 shot | establishing shot | group shot | insert | over the shoulder | single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 shot | 51 | 2 | 0 | 0 | 0 | 4 | 1 |
| 3 shot | 5 | 42 | 2 | 9 | 1 | 3 | 1 |
| establishing shot | 2 | 1 | 36 | 7 | 3 | 0 | 2 |
| group shot | 0 | 6 | 0 | 35 | 1 | 1 | 1 |
| insert | 0 | 0 | 4 | 0 | 62 | 0 | 4 |
| over the shoulder | 3 | 5 | 2 | 2 | 0 | 24 | 2 |
| single | 1 | 0 | 2 | 0 | 3 | 1 | 60 |

Compared with the original logistic regression matrix, the MLP's largest gain was on **3 shot**, while it missed more **establishing shot** and **over the shoulder** examples. For over the shoulder, however, it made fewer false predictions of that class: precision rose from **50.0% to 72.7%**, while recall fell from **73.7% to 63.2%**.

The MLP performed better overall on this test set. Its nonlinear decision boundaries may fit relationships in the frozen SigLIP feature space better than a linear classifier, although differences in class weighting, training and sampling may also contribute. Further evaluation across seeds and film-grouped splits is needed to establish how reliable the gain is.

## 3. Changing the backbone

We also tested replacing `google/siglip-base-patch16-224` with the larger `google/siglip-so400m-patch14-384`. The latter accepts 384-pixel inputs, potentially retaining more spatial detail.

Changing the backbone requires recomputing embeddings and caching them in new `.npy` files; the original `emb/` cache remains untouched. We expected this to take hours, possibly more than the roughly two hours for the original model. The SO400M run was recorded as taking **about 16 hours**.

### Resolution caveat

The embedding script uses `im.draft('RGB', (448, 448))` to reduce JPEG decoding size. If source images are small, a 384-pixel model cannot recover detail that is not present in the originals. Image sizes and decoding behavior should therefore be checked when comparing resolutions.

### SO400M: logistic regression versus MLP

| Actual class | Logistic regression correct | MLP correct |
| --- | ---: | ---: |
| 2 shot | 51 / 58 | 53 / 58 |
| 3 shot | 40 / 63 | 48 / 63 |
| establishing shot | 42 / 51 | 38 / 51 |
| group shot | 32 / 44 | 32 / 44 |
| insert | 57 / 70 | 60 / 70 |
| over the shoulder | 22 / 38 | 18 / 38 |
| single | 57 / 67 | 58 / 67 |

The MLP performed better overall with SO400M, particularly on 3 shot; logistic regression performed better on establishing shot and over the shoulder.

## Overall comparison

These metrics are calculated from the strict, single-label confusion matrices:

| Model | Correct / 391 | Accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| Base SigLIP + MLP | **310** | **79.3%** | **77.9%** |
| SO400M + MLP | 307 | 78.5% | 76.5% |
| SO400M + logistic regression | 301 | 77.0% | 75.4% |
| Base SigLIP + logistic regression | 298 | 76.2% | 75.1% |

**Conclusion:** Base SigLIP with an MLP was the best of these configurations on this test set. The larger SO400M backbone did not improve overall performance in these runs, despite its much longer embedding time. The differences are relatively small, so the ranking should be checked across additional seeds and film-grouped evaluations before treating it as definitive.

The ramaining errors cluster around related classes so the bottleneck could simply be the categorization as some of them overlap, for example, a single shot could also be considered a an over the shoulder with one person in it.

**Next Steps:** We could inspect misclassified images, check the label consistensy and class coverage, test for a targeted spacial, or consider multi-label classification (explain why we chose single label)

We probably will be moving forward onto another attribute instead of keep looking into this one because of time constraints.