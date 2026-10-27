# Iteration 8: fit-only asinh-MAD representation

Nine new transformed fits and 27 new evaluations are compared with 27 full_prior and nine repaired-copula records. No outer test is opened. The transformed representation is retained as a negative development result because inverse decoding produces extreme numerical outliers and worsens rule compliance.

| arm | inner_seed | rf | logistic | numeric_w | association_rmse | violation | max_missing_error_fit | c2st_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asinh_mad | 2027 | 0.7624 | 0.7504 | 198689473808622.7500 | 0.1531 | 0.2131 | 0.0018 | 0.8109 |
| asinh_mad | 3027 | 0.7072 | 0.7160 | 12195189255.7614 | 0.1583 | 0.2506 | 0.0025 | 0.8456 |
| asinh_mad | 4027 | 0.6791 | 0.6983 | 101962181240.9228 | 0.1488 | 0.2162 | 0.0022 | 0.8154 |
| copula_repair | 2027 | 0.7380 | 0.7284 | 0.1964 | 0.0898 | 0.0000 | 0.0023 | 0.5884 |
| copula_repair | 3027 | 0.7035 | 0.6996 | 0.1299 | 0.0615 | 0.0000 | 0.0010 | 0.5666 |
| copula_repair | 4027 | 0.6880 | 0.6986 | 0.1338 | 0.0869 | 0.0000 | 0.0019 | 0.5627 |
| full_prior | 2027 | 0.7383 | 0.7450 | 0.3049 | 0.1843 | 0.0550 | 0.0020 | 0.8722 |
| full_prior | 3027 | 0.6991 | 0.7126 | 0.2037 | 0.1489 | 0.0497 | 0.0024 | 0.8475 |
| full_prior | 4027 | 0.7076 | 0.6886 | 0.2223 | 0.1932 | 0.0596 | 0.0015 | 0.8788 |

## Component costs

| arm | count | mean | min | max | sum |
| --- | --- | --- | --- | --- | --- |
| asinh_mad | 9 | 5.8347 | 2.2989 | 7.9697 | 52.5120 |
