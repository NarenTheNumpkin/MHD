# Iteration 3: measured constraint-strength experiments

**46 new model training runs; 174 distinct sampling/evaluation configurations; 94 fitted checkpoints used.** Some checkpoints were reused from iteration 2. Resampling is not new model training.

All outcomes are development validation results. Outer fold 0 remains reserved. Seeds and overlapping inner partitions are not independent patient cohorts, and the utility tolerances are engineering screens rather than clinical noninferiority margins.

`rf` and `logistic` are synthetic-trained classifier AUROCs (higher better); `violation` is the fraction violating a supplied rule, `w` normalized numeric Wasserstein and `assoc` association RMSE (lower better). Read per-rule coverage and per-feature variance before interpreting compliance.

## Frozen-settings replication

Each row averages three generator seeds on one development partition. Every selected setting is retained, including failures. No setting was retuned after replication.

| dataset | inner_seed | setting | rf | logistic | violation | w | assoc | parity |
|---|---|---|---|---|---|---|---|---|
| ilpd | 2027 | base_v_l0_g0 | 0.694 | 0.671 | 0.971 | 0.435 | 0.277 | — |
| ilpd | 2027 | base_v_l0_g10 | 0.701 | 0.683 | 0.096 | 0.401 | 0.228 | — |
| ilpd | 2027 | base_x0_l0_g0 | 0.701 | 0.726 | 0.727 | 0.410 | 0.202 | — |
| ilpd | 2027 | dense_v_l0p01_g10 | 0.754 | 0.740 | 0.057 | 0.244 | 0.117 | — |
| ilpd | 2027 | dense_v_l0p1_g10 | 0.759 | 0.738 | 0.057 | 0.244 | 0.117 | — |
| ilpd | 2027 | full_v_l0p01_g10 | 0.716 | 0.753 | 0.058 | 0.294 | 0.179 | — |
| ilpd | 2027 | full_v_l0p1_g10 | 0.716 | 0.749 | 0.058 | 0.296 | 0.180 | — |
| ilpd | 2027 | full_x0_l0p1_g0p2 | 0.726 | 0.753 | 0.153 | 0.391 | 0.203 | — |
| ilpd | 3027 | base_v_l0_g0 | 0.723 | 0.685 | 0.935 | 0.262 | 0.217 | — |
| ilpd | 3027 | base_v_l0_g10 | 0.722 | 0.725 | 0.068 | 0.269 | 0.161 | — |
| ilpd | 3027 | base_x0_l0_g0 | 0.694 | 0.718 | 0.761 | 0.306 | 0.242 | — |
| ilpd | 3027 | dense_v_l0p01_g10 | 0.634 | 0.685 | 0.068 | 0.208 | 0.133 | — |
| ilpd | 3027 | dense_v_l0p1_g10 | 0.639 | 0.683 | 0.066 | 0.208 | 0.132 | — |
| ilpd | 3027 | full_v_l0p01_g10 | 0.680 | 0.703 | 0.060 | 0.227 | 0.147 | — |
| ilpd | 3027 | full_v_l0p1_g10 | 0.674 | 0.702 | 0.056 | 0.229 | 0.146 | — |
| ilpd | 3027 | full_x0_l0p1_g0p2 | 0.683 | 0.676 | 0.191 | 0.408 | 0.200 | — |
| ilpd | 4027 | base_v_l0_g0 | 0.702 | 0.691 | 0.956 | 0.303 | 0.245 | — |
| ilpd | 4027 | base_v_l0_g10 | 0.715 | 0.695 | 0.084 | 0.291 | 0.200 | — |
| ilpd | 4027 | base_x0_l0_g0 | 0.692 | 0.680 | 0.779 | 0.282 | 0.228 | — |
| ilpd | 4027 | dense_v_l0p01_g10 | 0.704 | 0.689 | 0.070 | 0.192 | 0.142 | — |
| ilpd | 4027 | dense_v_l0p1_g10 | 0.694 | 0.691 | 0.066 | 0.190 | 0.139 | — |
| ilpd | 4027 | full_v_l0p01_g10 | 0.712 | 0.694 | 0.066 | 0.242 | 0.182 | — |
| ilpd | 4027 | full_v_l0p1_g10 | 0.706 | 0.650 | 0.064 | 0.230 | 0.184 | — |
| ilpd | 4027 | full_x0_l0p1_g0p2 | 0.661 | 0.639 | 0.277 | 0.303 | 0.237 | — |
| toy | 2027 | base_x0_l0_g0 | 0.790 | 0.833 | 0.991 | 0.270 | 0.105 | 0.474 |
| toy | 2027 | base_x0_l0_g0p2 | 0.797 | 0.837 | 0.867 | 0.250 | 0.098 | 0.473 |
| toy | 2027 | full_x0_l0p1_g0p2 | 0.691 | 0.812 | 0.728 | 0.288 | 0.078 | 0.122 |
| toy | 3027 | base_x0_l0_g0 | 0.731 | 0.807 | 0.987 | 0.302 | 0.096 | 0.463 |
| toy | 3027 | base_x0_l0_g0p2 | 0.745 | 0.803 | 0.874 | 0.280 | 0.073 | 0.459 |
| toy | 3027 | full_x0_l0p1_g0p2 | 0.667 | 0.797 | 0.714 | 0.355 | 0.055 | 0.140 |
| toy | 4027 | base_x0_l0_g0 | 0.795 | 0.784 | 0.992 | 0.343 | 0.059 | 0.461 |
| toy | 4027 | base_x0_l0_g0p2 | 0.778 | 0.781 | 0.894 | 0.319 | 0.067 | 0.463 |
| toy | 4027 | full_x0_l0p1_g0p2 | 0.778 | 0.803 | 0.839 | 0.304 | 0.084 | 0.131 |

## Selected settings against matching backbones

Passing the engineering conditions on seed means is descriptive; it is not a confidence interval, statistical test or independent validation.

| dataset | inner_seed | setting | rf_delta | logistic_delta | rule_reduction | w_ratio | assoc_ratio | passes_screen_on_means |
|---|---|---|---|---|---|---|---|---|
| ilpd | 2027 | full_v_l0p1_g10 | 0.022 | 0.078 | 0.913 | 0.681 | 0.648 | True |
| ilpd | 3027 | full_v_l0p1_g10 | -0.049 | 0.017 | 0.879 | 0.873 | 0.676 | False |
| ilpd | 4027 | full_v_l0p1_g10 | 0.004 | -0.041 | 0.892 | 0.759 | 0.749 | False |
| ilpd | 2027 | full_v_l0p01_g10 | 0.022 | 0.082 | 0.913 | 0.676 | 0.644 | True |
| ilpd | 3027 | full_v_l0p01_g10 | -0.043 | 0.017 | 0.875 | 0.869 | 0.677 | False |
| ilpd | 4027 | full_v_l0p01_g10 | 0.010 | 0.003 | 0.890 | 0.800 | 0.741 | True |
| toy | 2027 | base_x0_l0_g0p2 | 0.007 | 0.004 | 0.124 | 0.926 | 0.928 | True |
| toy | 3027 | base_x0_l0_g0p2 | 0.014 | -0.004 | 0.113 | 0.928 | 0.763 | True |
| toy | 4027 | base_x0_l0_g0p2 | -0.017 | -0.003 | 0.099 | 0.932 | 1.122 | False |

## Complete seed-0 screen

| dataset | setting | rf | logistic | violation | w | assoc |
|---|---|---|---|---|---|---|
| ilpd | base_v_l0_g0 | 0.671 | 0.661 | 0.963 | 0.418 | 0.259 |
| ilpd | base_v_l0_g0_repair | 0.673 | 0.664 | 0.000 | 0.461 | 0.226 |
| ilpd | base_v_l0_g0p2 | 0.695 | 0.665 | 0.911 | 0.408 | 0.238 |
| ilpd | base_v_l0_g10 | 0.699 | 0.672 | 0.095 | 0.406 | 0.213 |
| ilpd | base_v_l0_g2 | 0.686 | 0.672 | 0.602 | 0.397 | 0.208 |
| ilpd | base_x0_l0_g0 | 0.665 | 0.735 | 0.751 | 0.385 | 0.196 |
| ilpd | base_x0_l0_g0_repair | 0.655 | 0.729 | 0.000 | 0.354 | 0.171 |
| ilpd | base_x0_l0_g0p2 | 0.668 | 0.737 | 0.748 | 0.387 | 0.195 |
| ilpd | base_x0_l0_g10 | 0.698 | 0.754 | 0.069 | 0.397 | 0.197 |
| ilpd | base_x0_l0_g2 | 0.679 | 0.740 | 0.275 | 0.395 | 0.201 |
| ilpd | full_v_l0_g0 | 0.760 | 0.785 | 0.579 | 0.280 | 0.156 |
| ilpd | full_v_l0_g0p2 | 0.746 | 0.774 | 0.524 | 0.279 | 0.155 |
| ilpd | full_v_l0_g10 | 0.742 | 0.798 | 0.060 | 0.276 | 0.155 |
| ilpd | full_v_l0_g2 | 0.745 | 0.782 | 0.169 | 0.278 | 0.154 |
| ilpd | full_v_l0p01_g0 | 0.756 | 0.793 | 0.559 | 0.278 | 0.154 |
| ilpd | full_v_l0p01_g0p2 | 0.754 | 0.789 | 0.516 | 0.277 | 0.153 |
| ilpd | full_v_l0p01_g10 | 0.743 | 0.797 | 0.057 | 0.275 | 0.155 |
| ilpd | full_v_l0p01_g2 | 0.741 | 0.788 | 0.163 | 0.277 | 0.152 |
| ilpd | full_v_l0p1_g0 | 0.737 | 0.759 | 0.547 | 0.284 | 0.157 |
| ilpd | full_v_l0p1_g0p2 | 0.729 | 0.757 | 0.504 | 0.284 | 0.156 |
| ilpd | full_v_l0p1_g10 | 0.731 | 0.786 | 0.054 | 0.282 | 0.157 |
| ilpd | full_v_l0p1_g2 | 0.715 | 0.765 | 0.135 | 0.284 | 0.155 |
| ilpd | full_v_l1_g0 | 0.619 | 0.704 | 0.547 | 0.281 | 0.156 |
| ilpd | full_v_l1_g0p2 | 0.620 | 0.698 | 0.499 | 0.280 | 0.155 |
| ilpd | full_v_l1_g10 | 0.630 | 0.734 | 0.060 | 0.276 | 0.156 |
| ilpd | full_v_l1_g2 | 0.629 | 0.696 | 0.178 | 0.280 | 0.155 |
| ilpd | full_x0_l0_g0 | 0.771 | 0.717 | 0.020 | 0.339 | 0.223 |
| ilpd | full_x0_l0_g0p2 | 0.771 | 0.717 | 0.020 | 0.339 | 0.223 |
| ilpd | full_x0_l0_g10 | 0.768 | 0.717 | 0.000 | 0.340 | 0.223 |
| ilpd | full_x0_l0_g2 | 0.767 | 0.718 | 0.006 | 0.340 | 0.223 |
| ilpd | full_x0_l0p01_g0 | 0.770 | 0.717 | 0.020 | 0.339 | 0.223 |
| ilpd | full_x0_l0p01_g0p2 | 0.770 | 0.717 | 0.020 | 0.339 | 0.223 |
| ilpd | full_x0_l0p01_g10 | 0.768 | 0.716 | 0.000 | 0.340 | 0.223 |
| ilpd | full_x0_l0p01_g2 | 0.766 | 0.717 | 0.006 | 0.340 | 0.223 |
| ilpd | full_x0_l0p1_g0 | 0.769 | 0.717 | 0.020 | 0.339 | 0.223 |
| ilpd | full_x0_l0p1_g0p2 | 0.768 | 0.717 | 0.020 | 0.339 | 0.223 |
| ilpd | full_x0_l0p1_g10 | 0.768 | 0.717 | 0.000 | 0.339 | 0.223 |
| ilpd | full_x0_l0p1_g2 | 0.767 | 0.717 | 0.006 | 0.339 | 0.223 |
| ilpd | full_x0_l1_g0 | 0.761 | 0.717 | 0.017 | 0.338 | 0.220 |
| ilpd | full_x0_l1_g0p2 | 0.761 | 0.717 | 0.017 | 0.338 | 0.220 |
| ilpd | full_x0_l1_g10 | 0.764 | 0.715 | 0.000 | 0.338 | 0.220 |
| ilpd | full_x0_l1_g2 | 0.762 | 0.716 | 0.003 | 0.338 | 0.220 |
| toy | base_v_l0_g0 | 0.856 | 0.834 | 1.000 | 0.181 | 0.157 |
| toy | base_v_l0_g0_repair | 0.831 | 0.806 | 0.000 | 1.062 | 0.149 |
| toy | base_v_l0_g0p2 | 0.826 | 0.837 | 0.833 | 0.188 | 0.061 |
| toy | base_v_l0_g10 | 0.818 | 0.829 | 1.000 | 8.228 | 0.073 |
| toy | base_v_l0_g2 | 0.844 | 0.841 | 1.000 | 0.476 | 0.146 |
| toy | base_x0_l0_g0 | 0.800 | 0.828 | 0.980 | 0.210 | 0.093 |
| toy | base_x0_l0_g0_repair | 0.784 | 0.798 | 0.000 | 0.191 | 0.094 |
| toy | base_x0_l0_g0p2 | 0.803 | 0.830 | 0.846 | 0.203 | 0.090 |
| toy | base_x0_l0_g10 | 0.820 | 0.829 | 0.997 | 5.712 | 0.111 |
| toy | base_x0_l0_g2 | 0.825 | 0.828 | 0.964 | 0.510 | 0.165 |
| toy | full_v_l0_g0 | 0.822 | 0.838 | 0.990 | 0.213 | 0.033 |
| toy | full_v_l0_g0p2 | 0.781 | 0.840 | 0.752 | 0.196 | 0.038 |
| toy | full_v_l0_g10 | 0.817 | 0.824 | 1.000 | 7.407 | 0.061 |
| toy | full_v_l0_g2 | 0.804 | 0.817 | 0.997 | 0.885 | 0.137 |
| toy | full_v_l0p01_g0 | 0.821 | 0.839 | 0.993 | 0.214 | 0.033 |
| toy | full_v_l0p01_g0p2 | 0.784 | 0.840 | 0.752 | 0.196 | 0.038 |
| toy | full_v_l0p01_g10 | 0.826 | 0.829 | 1.000 | 7.351 | 0.065 |
| toy | full_v_l0p01_g2 | 0.811 | 0.819 | 1.000 | 0.883 | 0.137 |
| toy | full_v_l0p1_g0 | 0.808 | 0.839 | 0.990 | 0.217 | 0.033 |
| toy | full_v_l0p1_g0p2 | 0.781 | 0.843 | 0.748 | 0.200 | 0.040 |
| toy | full_v_l0p1_g10 | 0.827 | 0.825 | 1.000 | 7.725 | 0.067 |
| toy | full_v_l0p1_g2 | 0.817 | 0.838 | 1.000 | 0.941 | 0.138 |
| toy | full_v_l1_g0 | 0.824 | 0.844 | 0.993 | 0.187 | 0.037 |
| toy | full_v_l1_g0p2 | 0.820 | 0.845 | 0.784 | 0.180 | 0.049 |
| toy | full_v_l1_g10 | 0.847 | 0.828 | 1.000 | 7.936 | 0.056 |
| toy | full_v_l1_g2 | 0.809 | 0.837 | 0.987 | 0.601 | 0.137 |
| toy | full_x0_l0_g0 | 0.603 | 0.771 | 0.967 | 0.261 | 0.059 |
| toy | full_x0_l0_g0p2 | 0.589 | 0.783 | 0.709 | 0.273 | 0.059 |
| toy | full_x0_l0_g10 | 0.758 | 0.798 | 0.944 | 2.569 | 0.147 |
| toy | full_x0_l0_g2 | 0.624 | 0.806 | 0.964 | 0.398 | 0.080 |
| toy | full_x0_l0p01_g0 | 0.612 | 0.771 | 0.967 | 0.261 | 0.059 |
| toy | full_x0_l0p01_g0p2 | 0.596 | 0.781 | 0.712 | 0.273 | 0.060 |
| toy | full_x0_l0p01_g10 | 0.767 | 0.797 | 0.941 | 2.654 | 0.145 |
| toy | full_x0_l0p01_g2 | 0.633 | 0.803 | 0.964 | 0.392 | 0.079 |
| toy | full_x0_l0p1_g0 | 0.607 | 0.775 | 0.980 | 0.265 | 0.061 |
| toy | full_x0_l0p1_g0p2 | 0.611 | 0.776 | 0.693 | 0.276 | 0.061 |
| toy | full_x0_l0p1_g10 | 0.742 | 0.800 | 0.918 | 2.649 | 0.145 |
| toy | full_x0_l0p1_g2 | 0.644 | 0.804 | 0.951 | 0.372 | 0.075 |
| toy | full_x0_l1_g0 | 0.824 | 0.826 | 0.964 | 0.488 | 0.102 |
| toy | full_x0_l1_g0p2 | 0.830 | 0.826 | 0.631 | 0.485 | 0.112 |
| toy | full_x0_l1_g10 | 0.798 | 0.832 | 0.967 | 3.966 | 0.184 |
| toy | full_x0_l1_g2 | 0.783 | 0.658 | 0.915 | 0.509 | 0.116 |
