# Step 2 - signal quality and uncertainty budget
## Sampling (whole file)

- dt_median_ms: 10.07
- dt_min_ms: 8.452
- dt_max_ms: 12.11
- dt_std_ms: 0.01586
- jitter_p99_ms: 0.04482
- gaps_over_1p5x: 0
- samples_per_cycle_mean: 1094
- cycle_duration_mean_s: 11.01

## Axis 1

### Resolution and noise floor per signal (healthy cycles, at rest)

| signal | units | quant_step | effective_bits | sigma_rest | sigma_white |
|---|---|---|---|---|---|
| joint_position | rad | 5.96e-06 | 17.89 | 9.979e-05 | 9e-06 |
| joint_velocity | rad/s | 0 | nan | 0.001996 | 0.002 |
| motor_position | rad | 2.98e-08 | 25.53 | 9.912e-05 | 3.127e-06 |
| motor_velocity | rad/s | 0 | nan | 0.001087 | 0.001039 |
| torque_sensor_a | N m | 0.01122 | 11.5 | 0.5107 | 0.1646 |
| torque_sensor_b | N m | 0.01123 | 11.46 | 0.4912 | 0.1627 |
| motor_torque | N m | 2.515e-08 | 32.11 | 0.7662 | 0.4897 |
| motor_iq | A | 0 | nan | 0.05022 | 0.03353 |
| motor_id | A | 0 | nan | 0.008283 | 0.008369 |
| computed_torque | N m | 0 | nan | 0.4401 | 0.04886 |
| target_position | rad | 7.451e-09 | 27.51 | 0 | 0 |
| target_velocity | rad/s | 0 | nan | 0 | 0 |
| target_acceleration | rad/s^2 | 0 | nan | 1.46e-08 | 1.045e-08 |
| robot_current | A | 0.01465 | 8.562 | 0.3371 | 0.1922 |
### Systematic errors

| comparison | units | bias_mean | cycle_to_cycle_std | within_cycle_std |
|---|---|---|---|---|
| torque sensor A - B, at rest | N m | 0.06608 | 0.1294 | 0.2683 |
| A - B vs torque level: slope (gain mismatch) | N m per N m | nan | nan | 0.2832 |
| torque sensor A - controller model, at rest | N m | 0.06087 | 0.1322 | nan |
| motor torque - torque sensor A, at rest | N m | 1.588 | 2.457 | nan |
| joint position - commanded, at rest | rad | -0.0001282 | 7.369e-05 | nan |
| joint position - commanded, while moving (rms) | rad | 0.006878 | 5.215e-05 | nan |
### Healthy state per recording batch

| variant | load_coulomb | load_c0 | load_viscous | gear_coulomb | gear_viscous | sensor_ab_mean | torque_mean | duration | cycles |
|---|---|---|---|---|---|---|---|---|---|
| BETWEEN_A | 0.1026 | 0.0255 | 0.6983 | 11.25 | 7.424 | -0.1202 | -0.5373 | 11.06 | 37 |
| BETWEEN_B | 0.0828 | 0.1307 | 0.7183 | 11.05 | 9.04 | 0.1667 | -0.7513 | 11.11 | 44 |
| BETWEEN_C | 0.0673 | 0.026 | 0.6926 | 11.48 | 7.91 | -0.0565 | -0.6468 | 11.04 | 19 |
| PRE_A | 0.1661 | 0.0313 | 0.6336 | 10.57 | 7.257 | 0.0155 | -0.7044 | 10.96 | 948 |
| PRE_B | 0.1221 | 0.003 | 0.7043 | 11.38 | 6.762 | -0.1126 | -0.9037 | 10.9 | 319 |
### Drift: between-batch vs within-batch spread

| quantity | within_batch_std | between_batch_std | ratio_between_over_within |
|---|---|---|---|
| load_coulomb | 0.0579 | 0.03837 | 0.6628 |
| load_c0 | 0.03443 | 0.05006 | 1.454 |
| load_viscous | 0.05122 | 0.03266 | 0.6377 |
| gear_coulomb | 0.362 | 0.3585 | 0.9902 |
| gear_viscous | 0.3825 | 0.8643 | 2.26 |
| sensor_ab_mean | 0.09322 | 0.1184 | 1.27 |
| torque_mean | 0.6182 | 0.1352 | 0.2187 |
| duration | 0.2156 | 0.08483 | 0.3934 |
### Propagation to the Coulomb friction estimate

- fit_residual_std_Nm: 1.458
- n_moving_samples: 305.9
- se_coulomb_white_Nm: 0.1432
- se_coulomb_corrected_Nm: 0.3292
- residual_lag1_corr: 0.679
- empirical_std_within_batch_Nm: 0.0579
- empirical_std_all_healthy_Nm: 0.08458
- quantization_contribution_Nm: 0.0001852
- ratio_empirical_over_corrected_se: 0.1759
### Spectral content (healthy cycles)

| signal | f95_hz | f99_hz | power_above_40hz |
|---|---|---|---|
| torque_sensor_a | 4.297 | 10.94 | 0.001613 |
| motor_torque | 2.344 | 11.33 | 0.002203 |
| motor_iq | 2.734 | 24.61 | 0.003584 |
| joint_velocity | 1.172 | 1.172 | 1.07e-05 |

## Axis 2

### Resolution and noise floor per signal (healthy cycles, at rest)

| signal | units | quant_step | effective_bits | sigma_rest | sigma_white |
|---|---|---|---|---|---|
| joint_position | rad | 5.96e-06 | 15.99 | 9.76e-05 | 8.319e-06 |
| joint_velocity | rad/s | 0 | nan | 0.001946 | 0.001955 |
| motor_position | rad | 1.192e-07 | 21.63 | 0.0001008 | 2.244e-06 |
| motor_velocity | rad/s | 0 | nan | 0.001112 | 0.001087 |
| torque_sensor_a | N m | 0.011 | 12.48 | 0.1648 | 0.07406 |
| torque_sensor_b | N m | 0.011 | 12.47 | 0.1619 | 0.07287 |
| motor_torque | N m | 9.537e-07 | 26.97 | 0.7981 | 0.6607 |
| motor_iq | A | 0 | nan | 0.0569 | 0.04517 |
| motor_id | A | 0 | nan | 0.01052 | 0.01065 |
| computed_torque | N m | 1.907e-06 | 24.79 | 3.009e-08 | 2.151e-08 |
| target_position | rad | 1.192e-07 | 21.61 | 0 | 0 |
| target_velocity | rad/s | 0 | nan | 0 | 0 |
| target_acceleration | rad/s^2 | 0 | nan | 1.096e-08 | 7.836e-09 |
| robot_current | A | 0.01465 | 8.562 | 0.1841 | 0.182 |
### Systematic errors

| comparison | units | bias_mean | cycle_to_cycle_std | within_cycle_std |
|---|---|---|---|---|
| torque sensor A - B, at rest | N m | 0.7407 | 0.15 | 0.4706 |
| A - B vs torque level: slope (gain mismatch) | N m per N m | 0.05687 | nan | 0.3715 |
| torque sensor A - controller model, at rest | N m | -7.209 | 0.1399 | nan |
| motor torque - torque sensor A, at rest | N m | 24.4 | 1.511 | nan |
| joint position - commanded, at rest | rad | -0.0001592 | 5.747e-05 | nan |
| joint position - commanded, while moving (rms) | rad | 0.002813 | 4.04e-05 | nan |
### Healthy state per recording batch

| variant | load_coulomb | load_c0 | load_viscous | gear_coulomb | gear_viscous | sensor_ab_mean | torque_mean | duration | cycles |
|---|---|---|---|---|---|---|---|---|---|
| BETWEEN_A | 0.5485 | -6.989 | 1.83 | 10.58 | 17.1 | 0.3998 | 61.33 | 11.06 | 37 |
| BETWEEN_B | 0.611 | -6.918 | 1.74 | 11.22 | 21.75 | 0.5596 | 61.12 | 11.11 | 44 |
| BETWEEN_C | 0.5373 | -7.071 | 1.922 | 10.71 | 17.57 | 0.4123 | 61.31 | 11.04 | 19 |
| PRE_A | 0.5868 | -6.889 | 1.795 | 10.8 | 17.73 | 0.5014 | 61.13 | 10.96 | 948 |
| PRE_B | 0.5576 | -6.965 | 1.824 | 10.53 | 16.31 | 0.3638 | 61.02 | 10.9 | 319 |
### Drift: between-batch vs within-batch spread

| quantity | within_batch_std | between_batch_std | ratio_between_over_within |
|---|---|---|---|
| load_coulomb | 0.1667 | 0.03015 | 0.1809 |
| load_c0 | 0.08673 | 0.07028 | 0.8102 |
| load_viscous | 0.2637 | 0.06634 | 0.2516 |
| gear_coulomb | 0.1849 | 0.2712 | 1.467 |
| gear_viscous | 0.692 | 2.116 | 3.058 |
| sensor_ab_mean | 0.05912 | 0.08063 | 1.364 |
| torque_mean | 0.49 | 0.1329 | 0.2712 |
| duration | 0.2156 | 0.08483 | 0.3934 |
### Propagation to the Coulomb friction estimate

- fit_residual_std_Nm: 2.301
- n_moving_samples: 430.8
- se_coulomb_white_Nm: 0.1711
- se_coulomb_corrected_Nm: 0.543
- residual_lag1_corr: 0.8182
- empirical_std_within_batch_Nm: 0.1667
- empirical_std_all_healthy_Nm: 0.172
- quantization_contribution_Nm: 0.000153
- ratio_empirical_over_corrected_se: 0.3069
### Spectral content (healthy cycles)

| signal | f95_hz | f99_hz | power_above_40hz |
|---|---|---|---|
| torque_sensor_a | 5.469 | 8.594 | 0.0002565 |
| motor_torque | 3.516 | 6.641 | 0.0006797 |
| motor_iq | 3.906 | 10.94 | 0.001652 |
| joint_velocity | 2.344 | 3.516 | 4.369e-05 |

## Axis 3

### Resolution and noise floor per signal (healthy cycles, at rest)

| signal | units | quant_step | effective_bits | sigma_rest | sigma_white |
|---|---|---|---|---|---|
| joint_position | rad | 5.96e-06 | 16.24 | 9.994e-05 | 8.429e-06 |
| joint_velocity | rad/s | 0 | nan | 0.001975 | 0.001991 |
| motor_position | rad | 1.192e-07 | 21.88 | 0.000101 | 2.022e-06 |
| motor_velocity | rad/s | 0 | nan | 0.001274 | 0.001273 |
| torque_sensor_a | N m | 0.01095 | 11.46 | 0.1769 | 0.06289 |
| torque_sensor_b | N m | 0.01094 | 11.49 | 0.179 | 0.06518 |
| motor_torque | N m | 4.768e-07 | 27.46 | 0.8768 | 0.757 |
| motor_iq | A | 0 | nan | 0.07316 | 0.06203 |
| motor_id | A | 0 | nan | 0.01032 | 0.01038 |
| computed_torque | N m | 1.907e-06 | 23.46 | 1.082e-05 | 7.739e-06 |
| target_position | rad | 1.192e-07 | 21.87 | 0 | 0 |
| target_velocity | rad/s | 0 | nan | 0 | 0 |
| target_acceleration | rad/s^2 | 0 | nan | 2.36e-06 | 1.688e-06 |
| robot_current | A | 0.01465 | 8.562 | 0.1824 | 0.1819 |
### Systematic errors

| comparison | units | bias_mean | cycle_to_cycle_std | within_cycle_std |
|---|---|---|---|---|
| torque sensor A - B, at rest | N m | -0.5058 | 0.1427 | 0.426 |
| A - B vs torque level: slope (gain mismatch) | N m per N m | 0.07369 | nan | 0.359 |
| torque sensor A - controller model, at rest | N m | 5.253 | 0.1649 | nan |
| motor torque - torque sensor A, at rest | N m | -5.138 | 1.272 | nan |
| joint position - commanded, at rest | rad | 3.478e-05 | 0.0001013 | nan |
| joint position - commanded, while moving (rms) | rad | 0.0028 | 0.0001201 | nan |
### Healthy state per recording batch

| variant | load_coulomb | load_c0 | load_viscous | gear_coulomb | gear_viscous | sensor_ab_mean | torque_mean | duration | cycles |
|---|---|---|---|---|---|---|---|---|---|
| BETWEEN_A | -0.6709 | 4.806 | 2.005 | 8.231 | 19.72 | -0.3942 | -38.67 | 11.06 | 37 |
| BETWEEN_B | -0.5774 | 4.713 | 1.915 | 8.554 | 22.67 | -0.2317 | -38.38 | 11.11 | 44 |
| BETWEEN_C | -0.6173 | 4.769 | 1.869 | 8.43 | 20.47 | -0.3272 | -38.65 | 11.04 | 19 |
| PRE_A | -0.5973 | 4.805 | 1.883 | 8.332 | 21.31 | -0.326 | -38.59 | 10.96 | 948 |
| PRE_B | -0.5897 | 4.843 | 1.893 | 8.161 | 19.39 | -0.4137 | -38.66 | 10.9 | 319 |
### Drift: between-batch vs within-batch spread

| quantity | within_batch_std | between_batch_std | ratio_between_over_within |
|---|---|---|---|
| load_coulomb | 0.08573 | 0.03672 | 0.4283 |
| load_c0 | 0.0264 | 0.04883 | 1.85 |
| load_viscous | 0.2086 | 0.054 | 0.2589 |
| gear_coulomb | 0.2443 | 0.1564 | 0.6402 |
| gear_viscous | 1.022 | 1.322 | 1.293 |
| sensor_ab_mean | 0.08243 | 0.07151 | 0.8675 |
| torque_mean | 0.1847 | 0.1236 | 0.6693 |
| duration | 0.2156 | 0.08483 | 0.3934 |
### Propagation to the Coulomb friction estimate

- fit_residual_std_Nm: 1.915
- n_moving_samples: 455.6
- se_coulomb_white_Nm: 0.1637
- se_coulomb_corrected_Nm: 0.3009
- residual_lag1_corr: 0.5334
- empirical_std_within_batch_Nm: 0.08573
- empirical_std_all_healthy_Nm: 0.09384
- quantization_contribution_Nm: 0.0001481
- ratio_empirical_over_corrected_se: 0.2849
### Spectral content (healthy cycles)

| signal | f95_hz | f99_hz | power_above_40hz |
|---|---|---|---|
| torque_sensor_a | 13.28 | 33.98 | 0.004867 |
| motor_torque | 5.078 | 26.95 | 0.003675 |
| motor_iq | 5.859 | 33.59 | 0.005485 |
| joint_velocity | 1.953 | 2.734 | 5.66e-05 |

## Axis 4

### Resolution and noise floor per signal (healthy cycles, at rest)

| signal | units | quant_step | effective_bits | sigma_rest | sigma_white |
|---|---|---|---|---|---|
| joint_position | rad | 5.722e-06 | 16.44 | 4.562e-05 | 7.693e-06 |
| joint_velocity | rad/s | 0 | nan | 0.001754 | 0.001726 |
| motor_position | rad | 2.384e-07 | 21.03 | 4.086e-05 | 3.377e-06 |
| motor_velocity | rad/s | 0 | nan | 0.001067 | 0.0009569 |
| torque_sensor_a | N m | 0.002855 | 11.47 | 0.07616 | 0.01989 |
| torque_sensor_b | N m | 0.002857 | 11.49 | 0.07628 | 0.02017 |
| motor_torque | N m | 5.96e-08 | 28.25 | 0.363 | 0.2599 |
| motor_iq | A | 0 | nan | 0.03979 | 0.02815 |
| motor_id | A | 0 | nan | 0.008929 | 0.008976 |
| computed_torque | N m | 4.768e-07 | 22.93 | 0 | 0 |
| target_position | rad | 2.384e-07 | 21 | 0 | 0 |
| target_velocity | rad/s | 0 | nan | 0 | 0 |
| target_acceleration | rad/s^2 | 0 | nan | 0 | 0 |
| robot_current | A | 0.01465 | 8.562 | 0.1816 | 0.1828 |
### Systematic errors

| comparison | units | bias_mean | cycle_to_cycle_std | within_cycle_std |
|---|---|---|---|---|
| torque sensor A - B, at rest | N m | 0.1153 | 0.03501 | 0.08467 |
| A - B vs torque level: slope (gain mismatch) | N m per N m | nan | nan | 0.06896 |
| torque sensor A - controller model, at rest | N m | 1.558 | 0.03302 | nan |
| motor torque - torque sensor A, at rest | N m | -1.934 | 0.1263 | nan |
| joint position - commanded, at rest | rad | -8.861e-07 | 4.352e-06 | nan |
| joint position - commanded, while moving (rms) | rad | 0.005656 | 5.581e-05 | nan |
### Healthy state per recording batch

| variant | load_coulomb | load_c0 | load_viscous | gear_coulomb | gear_viscous | sensor_ab_mean | torque_mean | duration | cycles |
|---|---|---|---|---|---|---|---|---|---|
| BETWEEN_A | 0.1958 | 1.506 | -0.2807 | 1.675 | 3.271 | 0.1639 | -7.244 | 11.06 | 37 |
| BETWEEN_B | 0.2087 | 1.44 | -0.3068 | 1.73 | 3.985 | 0.0626 | -7.216 | 11.11 | 44 |
| BETWEEN_C | 0.2198 | 1.483 | -0.2968 | 1.666 | 3.5 | 0.1285 | -7.221 | 11.04 | 19 |
| PRE_A | 0.1963 | 1.491 | -0.2828 | 1.664 | 3.515 | 0.1225 | -7.193 | 10.96 | 948 |
| PRE_B | 0.2061 | 1.536 | -0.2835 | 1.641 | 3.196 | 0.182 | -7.198 | 10.9 | 319 |
### Drift: between-batch vs within-batch spread

| quantity | within_batch_std | between_batch_std | ratio_between_over_within |
|---|---|---|---|
| load_coulomb | 0.02525 | 0.009898 | 0.392 |
| load_c0 | 0.01482 | 0.03497 | 2.359 |
| load_viscous | 0.03049 | 0.01127 | 0.3698 |
| gear_coulomb | 0.04171 | 0.0333 | 0.7984 |
| gear_viscous | 0.1282 | 0.3084 | 2.405 |
| sensor_ab_mean | 0.01217 | 0.04594 | 3.775 |
| torque_mean | 0.05121 | 0.02038 | 0.3981 |
| duration | 0.2156 | 0.08483 | 0.3934 |
### Propagation to the Coulomb friction estimate

- fit_residual_std_Nm: 0.4857
- n_moving_samples: 510.7
- se_coulomb_white_Nm: 0.03448
- se_coulomb_corrected_Nm: 0.04358
- residual_lag1_corr: 0.2246
- empirical_std_within_batch_Nm: 0.02525
- empirical_std_all_healthy_Nm: 0.02656
- quantization_contribution_Nm: 3.647e-05
- ratio_empirical_over_corrected_se: 0.5793
### Spectral content (healthy cycles)

| signal | f95_hz | f99_hz | power_above_40hz |
|---|---|---|---|
| torque_sensor_a | 39.06 | 46.88 | 0.04529 |
| motor_torque | 6.25 | 37.89 | 0.008166 |
| motor_iq | 7.031 | 38.67 | 0.00885 |
| joint_velocity | 1.953 | 2.734 | 0.0002596 |

## Axis 5

### Resolution and noise floor per signal (healthy cycles, at rest)

| signal | units | quant_step | effective_bits | sigma_rest | sigma_white |
|---|---|---|---|---|---|
| joint_position | rad | 5.722e-06 | 15.56 | 2.484e-05 | 7.87e-06 |
| joint_velocity | rad/s | 0 | nan | 0.001757 | 0.001771 |
| motor_position | rad | 1.192e-07 | 21.15 | 2.351e-05 | 1.861e-06 |
| motor_velocity | rad/s | 0 | nan | 0.0008722 | 0.0008411 |
| torque_sensor_a | N m | 0.002934 | 10.16 | 0.0387 | 0.02182 |
| torque_sensor_b | N m | 0.002924 | 10.19 | 0.03888 | 0.02183 |
| motor_torque | N m | 1.49e-08 | 29.59 | 0.322 | 0.1882 |
| motor_iq | A | 0 | nan | 0.03488 | 0.02037 |
| motor_id | A | 0 | nan | 0.008027 | 0.008074 |
| computed_torque | N m | 0 | nan | 0.0006929 | 8.278e-05 |
| target_position | rad | 1.192e-07 | 21.13 | 0 | 0 |
| target_velocity | rad/s | 0 | nan | 0 | 0 |
| target_acceleration | rad/s^2 | 0 | nan | 1.226e-10 | 1.248e-10 |
| robot_current | A | 0.01465 | 8.562 | 0.2015 | 0.1843 |
### Systematic errors

| comparison | units | bias_mean | cycle_to_cycle_std | within_cycle_std |
|---|---|---|---|---|
| torque sensor A - B, at rest | N m | 0.01292 | 0.01842 | 0.06628 |
| A - B vs torque level: slope (gain mismatch) | N m per N m | nan | nan | 0.02107 |
| torque sensor A - controller model, at rest | N m | 0.06089 | 0.01441 | nan |
| motor torque - torque sensor A, at rest | N m | 0.4992 | 0.1223 | nan |
| joint position - commanded, at rest | rad | -9.071e-06 | 3.313e-06 | nan |
| joint position - commanded, while moving (rms) | rad | 0.002462 | 9.968e-05 | nan |
### Healthy state per recording batch

| variant | load_coulomb | load_c0 | load_viscous | gear_coulomb | gear_viscous | sensor_ab_mean | torque_mean | duration | cycles |
|---|---|---|---|---|---|---|---|---|---|
| BETWEEN_A | -0.0282 | -0.0357 | 0.1376 | 1.201 | 2.67 | -0.0405 | -0.0168 | 11.06 | 37 |
| BETWEEN_B | -0.0148 | 0.004 | 0.1163 | 1.283 | 3.859 | 0.0059 | 0.0087 | 11.11 | 44 |
| BETWEEN_C | -0.0163 | -0.0152 | 0.1097 | 1.273 | 3.154 | -0.0345 | -0.0061 | 11.04 | 19 |
| PRE_A | -0.0218 | -0.019 | 0.131 | 1.25 | 3.256 | -0.0323 | 0.0008 | 10.96 | 948 |
| PRE_B | -0.0189 | -0.0331 | 0.1285 | 1.228 | 2.786 | -0.0603 | -0.0181 | 10.9 | 319 |
### Drift: between-batch vs within-batch spread

| quantity | within_batch_std | between_batch_std | ratio_between_over_within |
|---|---|---|---|
| load_coulomb | 0.01881 | 0.005289 | 0.2811 |
| load_c0 | 0.01031 | 0.01594 | 1.547 |
| load_viscous | 0.04321 | 0.01135 | 0.2626 |
| gear_coulomb | 0.05709 | 0.03361 | 0.5887 |
| gear_viscous | 0.2023 | 0.4681 | 2.313 |
| sensor_ab_mean | 0.00815 | 0.02404 | 2.95 |
| torque_mean | 0.0621 | 0.01144 | 0.1842 |
| duration | 0.2156 | 0.08483 | 0.3934 |
### Propagation to the Coulomb friction estimate

- fit_residual_std_Nm: 0.2272
- n_moving_samples: 207.6
- se_coulomb_white_Nm: 0.0271
- se_coulomb_corrected_Nm: 0.02719
- residual_lag1_corr: 0.003787
- empirical_std_within_batch_Nm: 0.01881
- empirical_std_all_healthy_Nm: 0.01739
- quantization_contribution_Nm: 5.879e-05
- ratio_empirical_over_corrected_se: 0.6918
### Spectral content (healthy cycles)

| signal | f95_hz | f99_hz | power_above_40hz |
|---|---|---|---|
| torque_sensor_a | 30.86 | 45.31 | 0.03267 |
| motor_torque | 13.28 | 42.97 | 0.01526 |
| motor_iq | 25.39 | 45.31 | 0.02132 |
| joint_velocity | 1.953 | 2.344 | 0.0007711 |

## Axis 6

### Resolution and noise floor per signal (healthy cycles, at rest)

| signal | units | quant_step | effective_bits | sigma_rest | sigma_white |
|---|---|---|---|---|---|
| joint_position | rad | 5.722e-06 | 18.01 | 2.994e-05 | 7.197e-06 |
| joint_velocity | rad/s | 0 | nan | 0.001735 | 0.00173 |
| motor_position | rad | 2.98e-08 | 25.59 | 2.78e-05 | 2.132e-06 |
| motor_velocity | rad/s | 0 | nan | 0.0006853 | 0.0006171 |
| torque_sensor_a | N m | 0.002763 | 8.492 | 0.01371 | 0.01342 |
| torque_sensor_b | N m | 0.00276 | 8.611 | 0.0135 | 0.01338 |
| motor_torque | N m | 1.49e-08 | 30.44 | 0.2557 | 0.1495 |
| motor_iq | A | 0 | nan | 0.02789 | 0.01618 |
| motor_id | A | 0 | nan | 0.009086 | 0.009193 |
| computed_torque | N m | 0 | nan | 1.029e-05 | 1.196e-06 |
| target_position | rad | 3.725e-09 | 28.58 | 0 | 0 |
| target_velocity | rad/s | 0 | nan | 0 | 0 |
| target_acceleration | rad/s^2 | 0 | nan | 2.595e-08 | 1.849e-08 |
| robot_current | A | 0.01465 | 8.562 | 0.352 | 0.1945 |
### Systematic errors

| comparison | units | bias_mean | cycle_to_cycle_std | within_cycle_std |
|---|---|---|---|---|
| torque sensor A - B, at rest | N m | 0.1626 | 0.01976 | 0.06394 |
| A - B vs torque level: slope (gain mismatch) | N m per N m | nan | nan | 0.06613 |
| torque sensor A - controller model, at rest | N m | 0.01458 | 0.0128 | nan |
| motor torque - torque sensor A, at rest | N m | -0.06787 | 0.126 | nan |
| joint position - commanded, at rest | rad | -8.793e-07 | 3.279e-06 | nan |
| joint position - commanded, while moving (rms) | rad | 0.006637 | 6.042e-05 | nan |
### Healthy state per recording batch

| variant | load_coulomb | load_c0 | load_viscous | gear_coulomb | gear_viscous | sensor_ab_mean | torque_mean | duration | cycles |
|---|---|---|---|---|---|---|---|---|---|
| BETWEEN_A | 0.1136 | 0.0223 | -0.003 | 1.752 | 2.585 | 0.1567 | -0.0412 | 11.06 | 37 |
| BETWEEN_B | 0.1218 | -0.0084 | -0.0026 | 1.883 | 3.032 | 0.1192 | -0.0303 | 11.11 | 44 |
| BETWEEN_C | 0.1092 | 0.0045 | -0.0038 | 1.754 | 2.706 | 0.1435 | 0.0552 | 11.04 | 19 |
| PRE_A | 0.1207 | 0.0046 | -0.0079 | 1.831 | 2.608 | 0.1268 | -0.0142 | 10.96 | 948 |
| PRE_B | 0.1166 | 0.0188 | -0.0068 | 1.78 | 2.438 | 0.1529 | -0.0128 | 10.9 | 319 |
### Drift: between-batch vs within-batch spread

| quantity | within_batch_std | between_batch_std | ratio_between_over_within |
|---|---|---|---|
| load_coulomb | 0.006543 | 0.00519 | 0.7932 |
| load_c0 | 0.006554 | 0.0124 | 1.892 |
| load_viscous | 0.00748 | 0.002398 | 0.3206 |
| gear_coulomb | 0.05171 | 0.05653 | 1.093 |
| gear_viscous | 0.1037 | 0.2222 | 2.143 |
| sensor_ab_mean | 0.01928 | 0.01632 | 0.8466 |
| torque_mean | 0.08078 | 0.0376 | 0.4654 |
| duration | 0.2156 | 0.08483 | 0.3934 |
### Propagation to the Coulomb friction estimate

- fit_residual_std_Nm: 0.09664
- n_moving_samples: 326.7
- se_coulomb_white_Nm: 0.008729
- se_coulomb_corrected_Nm: 0.009092
- residual_lag1_corr: 0.03986
- empirical_std_within_batch_Nm: 0.006543
- empirical_std_all_healthy_Nm: 0.007059
- quantization_contribution_Nm: 4.414e-05
- ratio_empirical_over_corrected_se: 0.7197
### Spectral content (healthy cycles)

| signal | f95_hz | f99_hz | power_above_40hz |
|---|---|---|---|
| torque_sensor_a | 45.7 | 49.22 | 0.1097 |
| motor_torque | 1.953 | 33.2 | 0.005609 |
| motor_iq | 2.734 | 36.33 | 0.007246 |
| joint_velocity | 1.172 | 1.172 | 2.978e-05 |
