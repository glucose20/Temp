# Kết quả so sánh: log vs log_seed

## Tóm tắt

| Thư mục | MSE | CI | R² | Pearson | Spearman |
|---------|-----|----|----|---------|----------|
| **log** | 0.474 ± 0.010 | 0.879 ± 0.006 | 0.719 ± 0.017 | 0.849 ± 0.010 | 0.676 ± 0.009 |
| **log_seed** | 0.481 ± 0.014 | 0.880 ± 0.008 | 0.711 ± 0.010 | 0.844 ± 0.005 | 0.678 ± 0.008 |

## Chi tiết log/ (lưu trong results_log/)

```
MSE       : 0.473891 ± 0.010157
CI        : 0.879136 ± 0.006402
R2        : 0.718964 ± 0.016632
Pearson   : 0.849262 ± 0.009591
Spearman  : 0.676341 ± 0.009212
```

Kết quả từng fold:
| Fold | MSE | CI | R² | Pearson | Spearman |
|------|-----|----|----|---------|----------|
| 0 | 0.465 | 0.886 | 0.723 | 0.853 | 0.681 |
| 1 | 0.462 | 0.877 | 0.743 | 0.862 | 0.672 |
| 2 | 0.477 | 0.870 | 0.716 | 0.847 | 0.666 |
| 3 | 0.481 | 0.879 | 0.697 | 0.836 | 0.673 |
| 4 | 0.485 | 0.884 | 0.717 | 0.847 | 0.690 |

## Chi tiết log_seed/ (lưu trong results_log_seed/)

```
MSE       : 0.480687 ± 0.013588
CI        : 0.880132 ± 0.007881
R2        : 0.711203 ± 0.009784
Pearson   : 0.843998 ± 0.005377
Spearman  : 0.677668 ± 0.007812
```

Kết quả từng fold:
| Fold | MSE | CI | R² | Pearson | Spearman |
|------|-----|----|----|---------|----------|
| 0 | 0.472 | 0.885 | 0.714 | 0.845 | 0.678 |
| 1 | 0.484 | 0.883 | 0.717 | 0.847 | 0.677 |
| 2 | 0.493 | 0.866 | 0.696 | 0.835 | 0.667 |
| 3 | 0.462 | 0.881 | 0.721 | 0.849 | 0.678 |
| 4 | 0.492 | 0.885 | 0.709 | 0.843 | 0.689 |

## Nhận xét

1. **Hiệu suất tương đương**: Cả hai thư mục cho kết quả gần giống nhau
   - MSE: ~0.47-0.48 (chênh lệch 0.007)
   - CI: ~0.88 (chênh lệch 0.001)
   - R²: ~0.71-0.72 (chênh lệch 0.008)

2. **Độ ổn định**:
   - `log`: Std thấp hơn ở R² (0.017 vs 0.010) - dao động nhiều hơn
   - `log_seed`: Std thấp hơn ở Pearson (0.005 vs 0.010) - ổn định hơn

3. **Kết quả tốt nhất**:
   - `log fold 1`: MSE=0.462, CI=0.877, R²=0.743 ⭐
   - `log_seed fold 3`: MSE=0.462, CI=0.881, R²=0.721 ⭐

## File đã lưu

### Thư mục log/
- `results_log/davis_warm_summary.csv` - Kết quả chi tiết từng fold
- `results_log/davis_warm_summary_stats.csv` - Thống kê trung bình ± std

### Thư mục log_seed/
- `results_log_seed/davis_warm_summary.csv` - Kết quả chi tiết từng fold
- `results_log_seed/davis_warm_summary_stats.csv` - Thống kê trung bình ± std
