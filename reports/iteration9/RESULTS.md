# Iteration 9: wrong-rule sensitivity

The 225 iteration-7 synthetic tables were re-evaluated under five frozen rule registries (1,125 analyses; no fits or sampling). `correct` is the supplied registry, `omit_protein_identity` removes the exploratory A/G relation, and the three reverse variants intentionally swap biochemical inequality directions. Reversed variants are stress tests, not clinical models.

| arm | variant | inner_seed | any_violation_rate | any_coverage |
| --- | --- | --- | --- | --- |
| copula_repair | correct | 2027 | 0.0000 | 1.0000 |
| copula_repair | correct | 3027 | 0.0000 | 1.0000 |
| copula_repair | correct | 4027 | 0.0000 | 1.0000 |
| copula_repair | omit_protein_identity | 2027 | 0.0000 | 1.0000 |
| copula_repair | omit_protein_identity | 3027 | 0.0000 | 1.0000 |
| copula_repair | omit_protein_identity | 4027 | 0.0000 | 1.0000 |
| copula_repair | reverse_albumin_order | 2027 | 1.0000 | 1.0000 |
| copula_repair | reverse_albumin_order | 3027 | 1.0000 | 1.0000 |
| copula_repair | reverse_albumin_order | 4027 | 1.0000 | 1.0000 |
| copula_repair | reverse_bilirubin_order | 2027 | 0.9257 | 1.0000 |
| copula_repair | reverse_bilirubin_order | 3027 | 0.9160 | 1.0000 |
| copula_repair | reverse_bilirubin_order | 4027 | 0.9114 | 1.0000 |
| copula_repair | reverse_both | 2027 | 1.0000 | 1.0000 |
| copula_repair | reverse_both | 3027 | 1.0000 | 1.0000 |
| copula_repair | reverse_both | 4027 | 1.0000 | 1.0000 |
| ctgan_default300 | correct | 2027 | 0.9721 | 1.0000 |
| ctgan_default300 | correct | 3027 | 0.9598 | 1.0000 |
| ctgan_default300 | correct | 4027 | 0.9719 | 1.0000 |
| ctgan_default300 | omit_protein_identity | 2027 | 0.6530 | 1.0000 |
| ctgan_default300 | omit_protein_identity | 3027 | 0.6532 | 1.0000 |
| ctgan_default300 | omit_protein_identity | 4027 | 0.6906 | 1.0000 |
| ctgan_default300 | reverse_albumin_order | 2027 | 0.9994 | 1.0000 |
| ctgan_default300 | reverse_albumin_order | 3027 | 1.0000 | 1.0000 |
| ctgan_default300 | reverse_albumin_order | 4027 | 0.9997 | 1.0000 |
| ctgan_default300 | reverse_bilirubin_order | 2027 | 0.9908 | 1.0000 |
| ctgan_default300 | reverse_bilirubin_order | 3027 | 0.9842 | 1.0000 |
| ctgan_default300 | reverse_bilirubin_order | 4027 | 0.9888 | 1.0000 |
| ctgan_default300 | reverse_both | 2027 | 0.9998 | 1.0000 |
| ctgan_default300 | reverse_both | 3027 | 1.0000 | 1.0000 |
| ctgan_default300 | reverse_both | 4027 | 1.0000 | 1.0000 |
| full_prior | correct | 2027 | 0.0550 | 1.0000 |
| full_prior | correct | 3027 | 0.0497 | 1.0000 |
| full_prior | correct | 4027 | 0.0596 | 1.0000 |
| full_prior | omit_protein_identity | 2027 | 0.0125 | 1.0000 |
| full_prior | omit_protein_identity | 3027 | 0.0223 | 1.0000 |
| full_prior | omit_protein_identity | 4027 | 0.0199 | 1.0000 |
| full_prior | reverse_albumin_order | 2027 | 1.0000 | 1.0000 |
| full_prior | reverse_albumin_order | 3027 | 1.0000 | 1.0000 |
| full_prior | reverse_albumin_order | 4027 | 1.0000 | 1.0000 |
| full_prior | reverse_bilirubin_order | 2027 | 0.9754 | 1.0000 |
| full_prior | reverse_bilirubin_order | 3027 | 0.9636 | 1.0000 |
| full_prior | reverse_bilirubin_order | 4027 | 0.9768 | 1.0000 |
| full_prior | reverse_both | 2027 | 1.0000 | 1.0000 |
| full_prior | reverse_both | 3027 | 1.0000 | 1.0000 |
| full_prior | reverse_both | 4027 | 1.0000 | 1.0000 |
| tvae_compact100 | correct | 2027 | 0.8572 | 1.0000 |
| tvae_compact100 | correct | 3027 | 0.8540 | 1.0000 |
| tvae_compact100 | correct | 4027 | 0.8552 | 1.0000 |
| tvae_compact100 | omit_protein_identity | 2027 | 0.1908 | 1.0000 |
| tvae_compact100 | omit_protein_identity | 3027 | 0.2110 | 1.0000 |
| tvae_compact100 | omit_protein_identity | 4027 | 0.2187 | 1.0000 |
| tvae_compact100 | reverse_albumin_order | 2027 | 1.0000 | 1.0000 |
| tvae_compact100 | reverse_albumin_order | 3027 | 1.0000 | 1.0000 |
| tvae_compact100 | reverse_albumin_order | 4027 | 1.0000 | 1.0000 |
| tvae_compact100 | reverse_bilirubin_order | 2027 | 0.9808 | 1.0000 |
| tvae_compact100 | reverse_bilirubin_order | 3027 | 0.9819 | 1.0000 |
| tvae_compact100 | reverse_bilirubin_order | 4027 | 0.9818 | 1.0000 |
| tvae_compact100 | reverse_both | 2027 | 1.0000 | 1.0000 |
| tvae_compact100 | reverse_both | 3027 | 1.0000 | 1.0000 |
| tvae_compact100 | reverse_both | 4027 | 1.0000 | 1.0000 |

Validation-population rates and all per-rule lower/upper bounds are in the companion CSVs. These descriptive sensitivities do not establish clinical truth, significance, or a model winner.
