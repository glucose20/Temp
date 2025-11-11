# HƯỚNG DẪN TRAIN VỚI THAM SỐ DÒNG LỆNH

## 🎯 Thay đổi chính

File `train.py` đã được sửa đổi để **nhận fold index từ tham số dòng lệnh** thay vì dùng vòng for. Điều này cho phép:

1. ✅ **Chạy song song nhiều fold** trên nhiều GPU
2. ✅ **Linh hoạt hơn** trong việc train từng fold riêng lẻ
3. ✅ **Dễ debug** khi một fold gặp lỗi
4. ✅ **Tối ưu tài nguyên GPU** khi có nhiều card

---

## 📖 Cách sử dụng

### **1. Train một fold đơn lẻ**

```powershell
# Train fold 0 trên GPU 0
python code/train.py --fold 0 --cuda "0"

# Train fold 1 trên GPU 1
python code/train.py --fold 1 --cuda "1"

# Train fold 2, sử dụng CUDA device từ hyperparameter.py
python code/train.py --fold 2
```

**Tham số:**
- `--fold`: **(Bắt buộc)** Index của fold cần train (0-4 cho 5-fold CV)
- `--cuda`: **(Tùy chọn)** GPU device ID, ghi đè giá trị trong `hyperparameter.py`

---

### **2. Train tất cả các fold tuần tự**

```powershell
# Windows PowerShell
for ($i=0; $i -lt 5; $i++) {
    python code/train.py --fold $i --cuda "0"
}
```

```bash
# Linux/Mac Bash
for fold in {0..4}; do
    python code/train.py --fold $fold --cuda "0"
done
```

---

### **3. Train tất cả các fold song song (KHUYẾN NGHỊ)**

#### **Trên Windows:**

```powershell
# Sử dụng script có sẵn
.\scripts\train_all_folds.ps1
```

**Hoặc thủ công:**
```powershell
# Start tất cả folds trong background
Start-Job -ScriptBlock { python code/train.py --fold 0 --cuda "0" }
Start-Job -ScriptBlock { python code/train.py --fold 1 --cuda "0" }
Start-Job -ScriptBlock { python code/train.py --fold 2 --cuda "0" }
Start-Job -ScriptBlock { python code/train.py --fold 3 --cuda "0" }
Start-Job -ScriptBlock { python code/train.py --fold 4 --cuda "0" }

# Xem trạng thái
Get-Job

# Xem output
Receive-Job -Id 1
```

#### **Trên Linux/Mac:**

```bash
# Sử dụng script có sẵn
bash scripts/train_all_folds.sh
```

**Hoặc thủ công:**
```bash
# Start tất cả folds trong background
python code/train.py --fold 0 --cuda "0" &
python code/train.py --fold 1 --cuda "1" &
python code/train.py --fold 2 --cuda "2" &
python code/train.py --fold 3 --cuda "3" &
python code/train.py --fold 4 --cuda "0" &

# Wait cho tất cả hoàn thành
wait
```

---

### **4. Train với nhiều GPU**

Nếu bạn có 4 GPUs, phân bổ như sau:

```powershell
# Windows
Start-Job -ScriptBlock { python code/train.py --fold 0 --cuda "0" }
Start-Job -ScriptBlock { python code/train.py --fold 1 --cuda "1" }
Start-Job -ScriptBlock { python code/train.py --fold 2 --cuda "2" }
Start-Job -ScriptBlock { python code/train.py --fold 3 --cuda "3" }
Start-Job -ScriptBlock { python code/train.py --fold 4 --cuda "0" }  # Quay lại GPU 0
```

```bash
# Linux/Mac
python code/train.py --fold 0 --cuda "0" &
python code/train.py --fold 1 --cuda "1" &
python code/train.py --fold 2 --cuda "2" &
python code/train.py --fold 3 --cuda "3" &
python code/train.py --fold 4 --cuda "0" &
wait
```

---

## 📊 Tổng hợp kết quả

Sau khi train xong tất cả các fold, chạy script tổng hợp:

```powershell
python code/aggregate_results.py --dataset davis --running_set novel-pair
```

**Output:**
```
Found 5 fold result files:
  - ./log/Test-davis-novel-pair-fold0-Nov11_10-30-45.csv
  - ./log/Test-davis-novel-pair-fold1-Nov11_10-31-12.csv
  - ...

============================================================
SUMMARY: davis-novel-pair
============================================================
   fold     mse    rmse      ci      r2  pearson  spearman
      0  0.421   0.649   0.856   0.712    0.844     0.838
      1  0.438   0.662   0.849   0.698    0.835     0.829
      2  0.415   0.644   0.861   0.718    0.848     0.842
      3  0.429   0.655   0.852   0.705    0.839     0.833
      4  0.423   0.650   0.858   0.714    0.845     0.839

============================================================
STATISTICS (Mean ± Std)
============================================================
mse       : 0.425200 ± 0.009154 (var=0.000084)
rmse      : 0.652000 ± 0.007280 (var=0.000053)
ci        : 0.855200 ± 0.004658 (var=0.000022)
r2        : 0.709400 ± 0.007958 (var=0.000063)
pearson   : 0.842200 ± 0.005070 (var=0.000026)
spearman  : 0.836200 ± 0.005263 (var=0.000028)
============================================================

Aggregated results saved to: ./log/Test-davis-novel-pair-AGGREGATED.csv
Summary statistics saved to: ./log/Test-davis-novel-pair-SUMMARY.csv
```

---

## 📁 Cấu trúc file output

```
log/
├── Nov11_10-30-45-davis-novel-pair-fold0.csv          # Training log fold 0
├── Nov11_10-31-12-davis-novel-pair-fold1.csv          # Training log fold 1
├── ...
├── Test-davis-novel-pair-fold0-Nov11_10-30-45.csv     # Test result fold 0
├── Test-davis-novel-pair-fold1-Nov11_10-31-12.csv     # Test result fold 1
├── ...
├── Test-davis-novel-pair-AGGREGATED.csv               # Tất cả folds
└── Test-davis-novel-pair-SUMMARY.csv                  # Thống kê tổng hợp

savemodel/
├── davis-novel-pair-fold0-Nov11_10-30-45.pth
├── davis-novel-pair-fold1-Nov11_10-31-12.pth
└── ...
```

---

## ⚡ Ví dụ thực tế

### **Scenario 1: Train nhanh 1 fold để test**

```powershell
python code/train.py --fold 0 --cuda "0"
```

### **Scenario 2: Train full 5-fold CV song song trên 2 GPUs**

```powershell
# GPU 0: fold 0, 2, 4
# GPU 1: fold 1, 3

Start-Job -ScriptBlock { python code/train.py --fold 0 --cuda "0" }
Start-Job -ScriptBlock { python code/train.py --fold 1 --cuda "1" }
Start-Job -ScriptBlock { python code/train.py --fold 2 --cuda "0" }
Start-Job -ScriptBlock { python code/train.py --fold 3 --cuda "1" }
Start-Job -ScriptBlock { python code/train.py --fold 4 --cuda "0" }

# Theo dõi tiến độ
Get-Job | Format-Table Id, State, Command

# Xem output real-time của job 1
Receive-Job -Id 1 -Keep

# Đợi tất cả hoàn thành
Get-Job | Wait-Job

# Tổng hợp kết quả
python code/aggregate_results.py --dataset davis --running_set novel-pair
```

### **Scenario 3: Chỉ train lại fold bị lỗi**

```powershell
# Giả sử fold 2 bị lỗi
python code/train.py --fold 2 --cuda "0"

# Sau đó tổng hợp lại
python code/aggregate_results.py --dataset davis --running_set novel-pair
```

---

## 🔧 Cấu hình nâng cao

### **Chỉnh sửa script `train_all_folds.ps1`:**

```powershell
# Dòng 7-9: Thay đổi dataset và task
$DATASET = "kiba"                  # davis, kiba, metz
$RUNNING_SET = "novel-drug"        # warm, novel-drug, novel-prot, novel-pair
$NUM_FOLDS = 5

# Dòng 14: Phân bổ GPU
$GPU_DEVICES = @("0", "1", "2", "3", "0")  # 4 GPUs available
```

### **Chỉnh sửa `hyperparameter.py`:**

```python
# Thay đổi mặc định
self.dataset = 'kiba'
self.running_set = 'novel-drug'
self.cuda = "0"  # GPU mặc định nếu không truyền --cuda
```

---

## 🚨 Lưu ý quan trọng

1. **Validation fold index**: Script sẽ kiểm tra `0 <= fold < kfold` trước khi train
2. **Memory**: Mỗi fold tải riêng embeddings vào RAM → cần đủ RAM nếu chạy song song
3. **GPU memory**: Mỗi fold cần ~6-8 GB VRAM → tối đa 1-2 folds/GPU (tùy card)
4. **Timestamp**: Mỗi lần chạy tạo timestamp mới → không ghi đè file cũ
5. **Early stopping**: Vẫn hoạt động bình thường với `max_patience=20`

---

## 📞 Hỗ trợ

Nếu gặp lỗi:

```powershell
# Kiểm tra fold index có hợp lệ không
python code/train.py --fold 5  # ERROR: Fold must be 0-4

# Kiểm tra GPU available
python -c "import torch; print(torch.cuda.is_available())"

# Xem log chi tiết
cat ./log/fold_0_console.log  # Linux
Get-Content ./log/fold_0_console.log  # Windows
```

---

**Thời gian ước tính:**
- 1 fold: ~30-60 phút (tùy GPU và early stopping)
- 5 folds tuần tự: ~2.5-5 giờ
- 5 folds song song (1 GPU): ~30-60 phút (giống 1 fold)
- 5 folds song song (5 GPUs): ~30-60 phút (nhanh nhất)
