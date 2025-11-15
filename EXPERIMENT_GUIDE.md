# 🚀 HƯỚNG DẪN CHẠY EXPERIMENTS TRÊN SERVER

## 📋 Tổng quan

Chạy tất cả 60 experiments cho LLMDTA:
- **3 datasets**: davis, kiba, metz
- **4 settings**: warm, novel-drug, novel-prot, novel-pair
- **5 folds**: 0, 1, 2, 3, 4
- **Total**: 60 runs với `--epochs 200 --batch_size 16`

---

## 🛠️ Chuẩn bị (Setup trên Server)

### 1. Clone repository và download data

```bash
# Clone repo
git clone https://github.com/glucose20/Temp.git
cd Temp

# Download dataset từ Kaggle (hoặc copy từ local)
# Cài kagglehub nếu cần
pip install kagglehub

# Download và giải nén data
python -c "
import kagglehub
import shutil
import os

# Download pretrained features
path = kagglehub.dataset_download('christang0002/llmdta')
pretrain_dir = f'{path}/pretrain-feature/pretrained-feature'

# Copy to project
for dataset in ['davis', 'kiba', 'metz']:
    src = os.path.join(pretrain_dir, dataset)
    dst = f'./data/{dataset}'
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f'Copied {dataset}')
"

# Giải nén 5-fold datasets
tar -xzf ./data/dta-5fold-dataset/davis.tar.gz -C ./data/dta-5fold-dataset/
tar -xzf ./data/dta-5fold-dataset/kiba.tar.gz -C ./data/dta-5fold-dataset/
tar -xzf ./data/dta-5fold-dataset/metz.tar.gz -C ./data/dta-5fold-dataset/
```

### 2. Cài đặt dependencies

```bash
pip install numpy pandas scipy scikit-learn torch tqdm gensim matplotlib mol2vec fair-esm rdkit
```

### 3. Cấp quyền thực thi cho scripts

```bash
chmod +x scripts/*.sh
```

---

## 🎯 Cách chạy Experiments

### **Option 1: Chạy tuần tự TẤT CẢ 60 runs (Sequential)**

**Dùng khi:** Chỉ có 1 GPU hoặc muốn chạy an toàn

```bash
# Chạy tất cả
bash scripts/run_all_experiments.sh
```

**Thời gian ước tính:** ~30-60 giờ (tùy GPU và early stopping)

**Ưu điểm:**
- ✅ An toàn, ít lỗi
- ✅ Dễ theo dõi log
- ✅ Không tốn nhiều RAM

**Nhược điểm:**
- ❌ Rất chậm

---

### **Option 2: Chạy song song với nhiều GPU (Parallel - KHUYẾN NGHỊ)**

**Dùng khi:** Có 2-4 GPUs

```bash
# Chỉnh sửa file trước (dòng 12-13)
nano scripts/run_all_experiments_parallel.sh

# Sửa:
NUM_GPUS=4
GPU_DEVICES=(0 1 2 3)  # IDs của GPUs bạn có

# Chạy
bash scripts/run_all_experiments_parallel.sh
```

**Thời gian ước tính:** 
- 2 GPUs: ~15-30 giờ
- 4 GPUs: ~8-15 giờ

**Ưu điểm:**
- ✅ Nhanh gấp N lần (N = số GPU)
- ✅ Tận dụng tối đa hardware

**Nhược điểm:**
- ❌ Tốn nhiều RAM (mỗi process load embeddings riêng)
- ❌ Khó debug nếu có lỗi

---

### **Option 3: Chạy từng dataset riêng lẻ**

**Dùng khi:** Muốn chạy từng dataset một, hoặc test trước

```bash
# Chạy DAVIS (20 runs: 4 settings × 5 folds)
bash scripts/run_single_dataset.sh davis

# Chạy KIBA
bash scripts/run_single_dataset.sh kiba

# Chạy METZ
bash scripts/run_single_dataset.sh metz
```

**Thời gian mỗi dataset:** ~10-20 giờ

---

### **Option 4: Chạy thủ công từng run (Debug)**

```bash
# Test nhanh với 1 epoch
python code/train.py --fold 0 --cuda "0" --dataset davis --running_set warm --epochs 1 --batch_size 16

# Chạy thật 1 run
python code/train.py --fold 0 --cuda "0" --dataset davis --running_set novel-pair --epochs 200 --batch_size 16
```

---

## 📊 Theo dõi tiến độ

### **Xem log real-time**

```bash
# Xem master log
tail -f ./results/experiment_master_log_*.txt

# Xem log của 1 run cụ thể
tail -f ./log/*davis*novel-pair*fold0*.log
```

### **Kiểm tra GPU usage**

```bash
# Xem real-time
watch -n 1 nvidia-smi

# Hoặc
nvidia-smi -l 1
```

### **Đếm số runs đã hoàn thành**

```bash
# Đếm model files
ls -1 ./savemodel/*.pth | wc -l

# Đếm test result files
ls -1 ./log/Test-*-fold*.csv | wc -l
```

---

## 📈 Tổng hợp kết quả

### **Tổng hợp từng dataset-setting**

```bash
# Tự động chạy sau mỗi setting (nếu dùng script)
# Hoặc chạy thủ công:
python code/aggregate_results.py --dataset davis --running_set warm
python code/aggregate_results.py --dataset davis --running_set novel-pair
# ... (làm tương tự cho tất cả)
```

### **Tạo báo cáo tổng hợp cuối cùng**

```bash
python code/generate_final_report.py
```

**Output:**
- `./log/FINAL_SUMMARY_REPORT_<timestamp>.csv`
- Console output với bảng so sánh và best results

---

## 🔧 Cấu hình nâng cao

### **Thay đổi GPU trong script**

```bash
# Edit script
nano scripts/run_all_experiments.sh

# Dòng 9: Thay đổi GPU
CUDA_DEVICE="1"  # Chuyển sang GPU 1
```

### **Thay đổi hyperparameters**

```bash
# Edit script
nano scripts/run_all_experiments.sh

# Dòng 7-8: Thay đổi
EPOCHS=100       # Giảm xuống 100 epochs
BATCH_SIZE=32    # Tăng batch size
```

### **Chạy subset của experiments**

```bash
# Edit script để chỉ chạy một vài settings
nano scripts/run_all_experiments.sh

# Dòng 13: Bỏ bớt settings
SETTINGS=("warm" "novel-pair")  # Chỉ chạy 2 settings thay vì 4
```

---

## 🚨 Xử lý lỗi

### **Nếu một run bị lỗi:**

```bash
# Xem log chi tiết
cat ./results/run_davis_warm_fold0.log

# Chạy lại run đó
python code/train.py --fold 0 --cuda "0" --dataset davis --running_set warm --epochs 200 --batch_size 16

# Tổng hợp lại results
python code/aggregate_results.py --dataset davis --running_set warm
```

### **Nếu hết VRAM (Out of Memory):**

```bash
# Giảm batch size
python code/train.py --fold 0 --cuda "0" --dataset davis --running_set warm --epochs 200 --batch_size 8
```

### **Nếu server bị disconnect:**

Dùng `tmux` hoặc `screen` để chạy background:

```bash
# Sử dụng tmux (khuyến nghị)
tmux new -s llmdta
bash scripts/run_all_experiments.sh

# Detach: Ctrl+B, D
# Reattach: tmux attach -t llmdta

# Hoặc dùng nohup
nohup bash scripts/run_all_experiments.sh > experiment.log 2>&1 &

# Xem tiến độ
tail -f experiment.log
```

---

## 📁 Cấu trúc output

Sau khi chạy xong:

```
log/
├── experiment_master_log_<timestamp>.txt          # Master log
├── Nov12_10-30-45-davis-warm-fold0.csv           # Training curves
├── Test-davis-warm-fold0-Nov12_10-30-45.csv      # Individual fold results
├── Test-davis-warm-AGGREGATED.csv                 # Aggregated 5 folds
├── Test-davis-warm-SUMMARY.csv                    # Statistics
├── ... (tương tự cho 12 combinations)
└── FINAL_SUMMARY_REPORT_<timestamp>.csv           # Final report

savemodel/
├── davis-warm-fold0-Nov12_10-30-45.pth
├── davis-warm-fold1-Nov12_10-35-12.pth
└── ... (60 model files total)

results/
├── experiment_master_log_<timestamp>.txt
├── run_davis_warm_fold0.log
└── ... (60 individual run logs)
```

---

## 📊 Kết quả mẫu

```
============================================================
LLMDTA - Final Experiment Summary Report
============================================================

DAVIS - warm
------------------------------------------------------------
  mse       : 0.421000 ± 0.009154
  rmse      : 0.649000 ± 0.007280
  ci        : 0.856000 ± 0.004658
  r2        : 0.712000 ± 0.007958
  pearson   : 0.844000 ± 0.005070
  spearman  : 0.838000 ± 0.005263

[... tương tự cho 11 combinations khác ...]

Comparison Table (Mean MSE)
============================================================
setting     warm  novel-drug  novel-prot  novel-pair
dataset                                              
davis      0.421       0.512       0.634       0.789
kiba       0.345       0.445       0.556       0.667
metz       0.398       0.498       0.598       0.698

Best Results by Metric
============================================================
Best MSE: kiba - warm = 0.345000
Best CI:  davis - warm = 0.856000
Best R²:  davis - warm = 0.712000
```

---

## ⏱️ Thời gian ước tính

| Method | GPUs | Time per run | Total time |
|--------|------|--------------|------------|
| Sequential | 1 | ~30-60 min | ~30-60 hours |
| Parallel (2 GPUs) | 2 | ~30-60 min | ~15-30 hours |
| Parallel (4 GPUs) | 4 | ~30-60 min | ~8-15 hours |

*Thời gian thực tế tùy thuộc vào GPU, early stopping, và data size*

---

## 💡 Tips

1. **Test trước với 1-2 epochs**: 
   ```bash
   python code/train.py --fold 0 --cuda "0" --dataset davis --running_set warm --epochs 2
   ```

2. **Chạy overnight với tmux**: Server có thể disconnect nhưng job vẫn chạy

3. **Backup định kỳ**: Copy `./log/` và `./savemodel/` về local

4. **Monitor GPU**: Đảm bảo GPU utilization ~80-100%

5. **Check disk space**: Mỗi model ~100MB, tổng cộng ~6GB

---

## 📞 Troubleshooting

| Lỗi | Giải pháp |
|-----|-----------|
| `CUDA out of memory` | Giảm `--batch_size` xuống 8 hoặc 4 |
| `File not found` | Kiểm tra đường dẫn data, chạy setup lại |
| `Permission denied` | `chmod +x scripts/*.sh` |
| Script dừng giữa chừng | Dùng `tmux` hoặc `nohup` |
| Kết quả không aggregated | Chạy `python code/aggregate_results.py` thủ công |

---

**Happy Training! 🚀**
