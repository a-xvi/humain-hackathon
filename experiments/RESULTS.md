# Results

## Main results (human-verified test set)

| name                             |   macro_f1 |   ci_low |   ci_high |   accuracy |   n |   labels_used |
|:---------------------------------|-----------:|---------:|----------:|-----------:|----:|--------------:|
| baseline_majority                |     0.0279 |   0.0208 |    0.035  |     0.1254 | 319 |           nan |
| teacher_gpt-4.1-mini             |     0.8841 |   0.8428 |    0.9186 |     0.8934 | 319 |           nan |
| tfidf_lr_all_pool_labels         |     0.8421 |   0.7932 |    0.8806 |     0.8652 | 319 |          1920 |
| bert_all_pool_labels             |     0.8927 |   0.8543 |    0.925  |     0.8997 | 319 |          1920 |
| bert_margin_640_labels           |     0.8824 |   0.8414 |    0.9168 |     0.8934 | 319 |           640 |
| bert_margin_640_labels_augmented |     0.8775 |   0.8377 |    0.9108 |     0.8871 | 319 |           640 |

## Teacher labels needed (MARBERTv2 student)

| target                   |   margin |   margin_diverse |   random |
|:-------------------------|---------:|-----------------:|---------:|
| 90% of reference (0.803) |      560 |              560 |      nan |
| 95% of reference (0.848) |      640 |              nan |      nan |
| 98% of reference (0.875) |      nan |              nan |      nan |

## Robustness (macro-F1)

| set                           |   n |   student (active) |   student (all labels) |   tfidf (all labels) |   teacher |   student (active + augmentation) |
|:------------------------------|----:|-------------------:|-----------------------:|---------------------:|----------:|----------------------------------:|
| test (clean)                  | 319 |              0.882 |                  0.893 |                0.842 |     0.884 |                             0.877 |
| stress_arabizi                |  79 |              0.085 |                  0.196 |                0.053 |     0.87  |                             0.243 |
| stress_codeswitch             |  79 |              0.756 |                  0.772 |                0.598 |     0.855 |                             0.767 |
| stress_unseen_dialect         |  80 |              0.93  |                  0.951 |                0.739 |     0.931 |                             0.951 |
| perturbed: typos              | 319 |              0.865 |                  0.877 |                0.836 |   nan     |                             0.869 |
| perturbed: letter_variants    | 319 |              0.83  |                  0.836 |                0.842 |   nan     |                             0.843 |
| perturbed: arabizi_rule_based | 319 |              0.074 |                  0.07  |                0.072 |   nan     |                             0.503 |
| perturbed: truncated_half     | 319 |              0.77  |                  0.792 |                0.745 |   nan     |                             0.772 |
| perturbed: noise_emoji        | 319 |              0.876 |                  0.898 |                0.841 |   nan     |                             0.869 |

## Compute (1 CPU thread)

| model                      |   macro_f1 |   p50_ms |   p95_ms |   msgs_per_sec |   size_mb |   ram_mb |
|:---------------------------|-----------:|---------:|---------:|---------------:|----------:|---------:|
| student fp32               |      0.877 |    112   |    134   |            8.8 |     651.5 |     3067 |
| student int8               |      0.888 |     46.2 |     60.7 |           21.3 |     394.9 |     3393 |
| teacher gpt-4.1-mini (API) |      0.884 |    678   |   1507   |          nan   |     nan   |      nan |

## Hybrid mode: big-model usage vs quality

|   threshold |   sent_to_big_model |   macro_f1 |
|------------:|--------------------:|-----------:|
|        0    |               0     |     0.8775 |
|        0.3  |               0     |     0.8775 |
|        0.4  |               0.016 |     0.8754 |
|        0.5  |               0.028 |     0.8891 |
|        0.6  |               0.053 |     0.8837 |
|        0.7  |               0.088 |     0.8865 |
|        0.8  |               0.141 |     0.905  |
|        0.85 |               0.223 |     0.9127 |
|        0.9  |               0.426 |     0.9126 |
|        0.95 |               1     |     0.8841 |
|        0.99 |               1     |     0.8841 |
|        1.01 |               1     |     0.8841 |

## Data

| set                   | dialect   |   n |
|:----------------------|:----------|----:|
| pool                  | eastern   | 480 |
| pool                  | hijazi    | 480 |
| pool                  | msa       | 480 |
| pool                  | najdi     | 480 |
| stress_arabizi        | arabizi   |  79 |
| stress_codeswitch     | mixed     |  79 |
| stress_unseen_dialect | egyptian  |  40 |
| stress_unseen_dialect | levantine |  40 |
| test                  | eastern   |  80 |
| test                  | hijazi    |  79 |
| test                  | msa       |  80 |
| test                  | najdi     |  80 |
