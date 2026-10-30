# Results

## Main results (human-verified test set)

| name                             |   macro_f1 |   ci_low |   ci_high |   accuracy |   n |   labels_used |
|:---------------------------------|-----------:|---------:|----------:|-----------:|----:|--------------:|
| baseline_majority                |     0.0272 |   0.0201 |    0.0344 |     0.1219 | 320 |           nan |
| teacher_gpt-4.1-mini             |     0.9184 |   0.8872 |    0.9487 |     0.9187 | 320 |           nan |
| tfidf_lr_all_pool_labels         |     0.9313 |   0.902  |    0.9556 |     0.9313 | 320 |          1920 |
| bert_all_pool_labels             |     0.9349 |   0.9082 |    0.9603 |     0.9344 | 320 |          1920 |
| bert_margin_640_labels           |     0.9415 |   0.9148 |    0.9658 |     0.9406 | 320 |           640 |
| bert_margin_640_labels_augmented |     0.9441 |   0.9182 |    0.9675 |     0.9437 | 320 |           640 |

## Teacher labels needed (MARBERTv2 student)

| target                   |   margin |   margin_diverse |   random | margin_saving_vs_random   | margin_diverse_saving_vs_random   |
|:-------------------------|---------:|-----------------:|---------:|:--------------------------|:----------------------------------|
| 90% of reference (0.841) |      320 |              320 |      320 | 0%                        | 0%                                |
| 95% of reference (0.888) |      400 |              320 |      480 | 17%                       | 33%                               |
| 98% of reference (0.916) |      480 |              480 |      560 | 14%                       | 14%                               |

## Robustness (macro-F1)

| set                           |   n |   student (active) |   student (all labels) |   tfidf (all labels) |   teacher |   student (active + augmentation) |
|:------------------------------|----:|-------------------:|-----------------------:|---------------------:|----------:|----------------------------------:|
| test (clean)                  | 320 |              0.942 |                  0.935 |                0.931 |     0.918 |                             0.944 |
| stress_arabizi                |  80 |              0.156 |                  0.19  |                0.087 |     0.802 |                             0.621 |
| stress_codeswitch             |  80 |              0.836 |                  0.742 |                0.818 |     0.938 |                             0.775 |
| stress_unseen_dialect         |  80 |              0.936 |                  0.937 |                0.88  |     0.938 |                             0.949 |
| perturbed: typos              | 320 |              0.933 |                  0.923 |                0.916 |   nan     |                             0.926 |
| perturbed: letter_variants    | 320 |              0.926 |                  0.926 |                0.931 |   nan     |                             0.932 |
| perturbed: arabizi_rule_based | 320 |              0.052 |                  0.064 |                0.067 |   nan     |                             0.576 |
| perturbed: truncated_half     | 320 |              0.843 |                  0.836 |                0.807 |   nan     |                             0.854 |
| perturbed: noise_emoji        | 320 |              0.926 |                  0.932 |                0.919 |   nan     |                             0.947 |

## Compute (1 CPU thread)

| model                      |   macro_f1 |   p50_ms |   p95_ms |   msgs_per_sec |   size_mb |   ram_mb |
|:---------------------------|-----------:|---------:|---------:|---------------:|----------:|---------:|
| student fp32               |      0.944 |    100.7 |    127   |            9.6 |     651.5 |     3016 |
| student int8               |      0.842 |     34.8 |     45.2 |           28.4 |     394.9 |     3400 |
| teacher gpt-4.1-mini (API) |      0.918 |    707   |   1362   |          nan   |     nan   |      nan |

## Hybrid mode: big-model usage vs quality

|   threshold |   sent_to_big_model |   macro_f1 |
|------------:|--------------------:|-----------:|
|        0    |               0     |     0.9441 |
|        0.3  |               0     |     0.9441 |
|        0.4  |               0.003 |     0.944  |
|        0.5  |               0.025 |     0.9533 |
|        0.6  |               0.037 |     0.9626 |
|        0.7  |               0.059 |     0.9626 |
|        0.8  |               0.097 |     0.975  |
|        0.85 |               0.122 |     0.9718 |
|        0.9  |               0.438 |     0.9561 |
|        0.95 |               1     |     0.9184 |
|        0.99 |               1     |     0.9184 |
|        1.01 |               1     |     0.9184 |

## Data

| set                   | dialect   |   n |
|:----------------------|:----------|----:|
| pool                  | eastern   | 480 |
| pool                  | hijazi    | 480 |
| pool                  | msa       | 480 |
| pool                  | najdi     | 480 |
| stress_arabizi        | arabizi   |  80 |
| stress_codeswitch     | mixed     |  80 |
| stress_unseen_dialect | egyptian  |  40 |
| stress_unseen_dialect | levantine |  40 |
| test                  | eastern   |  80 |
| test                  | hijazi    |  80 |
| test                  | msa       |  80 |
| test                  | najdi     |  80 |
