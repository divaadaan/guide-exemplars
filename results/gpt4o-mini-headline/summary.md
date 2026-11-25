### Benchmark `gpt4o-mini-headline`

Backbone `openai/gpt-4o-mini`, seeds [0], test split, 95% percentile-bootstrap CIs (B=2000). Spend $21.17.

**aes-p7** (test n=311; human rater1 vs rater2: acc 0.624, QWK 0.599)

| Method | k | Acc | QWK | AdjErr | NonAdjErr |
|---|---:|---:|---:|---:|---:|
| **guide** | 10 | 0.621 [0.566, 0.675] | 0.380 [0.291, 0.465] | 0.363 [0.309, 0.418] | 0.016 [0.003, 0.032] |
| naive | 10 | 0.399 [0.347, 0.453] | 0.234 [0.161, 0.308] | 0.486 [0.431, 0.540] | 0.116 [0.080, 0.154] |
| random | 10 | 0.488 [0.442, 0.536] | 0.315 [0.252, 0.376] | 0.457 [0.413, 0.500] | 0.055 [0.035, 0.076] |
| knn | 10 | 0.514 [0.460, 0.569] | 0.358 [0.286, 0.431] | 0.460 [0.405, 0.514] | 0.026 [0.010, 0.045] |
| vote_k | 10 | 0.376 [0.328, 0.434] | 0.152 [0.093, 0.213] | 0.514 [0.460, 0.569] | 0.109 [0.074, 0.145] |

Reference: always predicting the most common score (level 2) gives accuracy 0.579 and QWK 0.

Predicted score counts on the test split (mean per draw for multi-draw methods):

| | level 0 | level 1 | level 2 | level 3 | failed |
|---|---:|---:|---:|---:|---:|
| *gold* | 2 | 65 | 180 | 64 | |
| guide | 1 | 73 | 234 | 3 | 0 |
| naive | 25 | 166 | 118 | 2 | 0 |
| random | 7.2 | 145 | 157.8 | 1 | 0 |
| knn | 3 | 127 | 180 | 1 | 0 |
| vote_k | 1 | 214 | 95 | 1 | 0 |

Paired GUIDE - method (accuracy): naive +0.222 [+0.161, +0.280], P(Δ>0)=1.00; random +0.132 [+0.087, +0.177], P(Δ>0)=1.00; knn +0.106 [+0.055, +0.161], P(Δ>0)=1.00; vote_k +0.244 [+0.180, +0.305], P(Δ>0)=1.00

