# 🚀 Quick Start - Chạy 60 Experiments trên Server

## Setup ban đầu (chỉ 1 lần)

```bash
# 1. Clone repo
git clone https://github.com/glucose20/Temp.git
cd Temp

# 2. Chạy script setup tự động
bash scripts/setup_server.sh

# 3. Test nhanh (2 epochs để kiểm tra)
python code/train.py --fold 0 --cuda "0" --dataset davis --running_set warm --epochs 2
```

---

## Chạy TẤT CẢ 60 experiments

### **Cách 1: Sequential (1 GPU, an toàn nhất)**
```bash
# Dùng tmux để tránh disconnect
tmux new -s llmdta
bash scripts/run_all_experiments.sh

# Detach: Ctrl+B, D
# Reattach: tmux attach -t llmdta
```

### **Cách 2: Parallel (nhiều GPU, nhanh nhất)**
```bash
# Edit số GPU trước
nano scripts/run_all_experiments_parallel.sh
# Sửa dòng 12-13: NUM_GPUS=4 và GPU_DEVICES=(0 1 2 3)

# Chạy
tmux new -s llmdta
bash scripts/run_all_experiments_parallel.sh
```

### **Cách 3: Từng dataset (chia nhỏ)**
```bash
# Terminal 1 (GPU 0)
CUDA_VISIBLE_DEVICES=0 bash scripts/run_single_dataset.sh davis

# Terminal 2 (GPU 1) 
CUDA_VISIBLE_DEVICES=1 bash scripts/run_single_dataset.sh kiba

# Terminal 3 (GPU 2)
CUDA_VISIBLE_DEVICES=2 bash scripts/run_single_dataset.sh metz
```

---

## Theo dõi tiến độ

```bash
# Xem log real-time
tail -f ./results/experiment_master_log_*.txt

# GPU usage
watch -n 1 nvidia-smi

# Đếm số runs hoàn thành
ls -1 ./savemodel/*.pth | wc -l  # Mục tiêu: 60 files
```

---

## Tổng hợp kết quả cuối

```bash
# Tạo báo cáo tổng hợp
python code/generate_final_report.py

# Xem kết quả
cat ./log/FINAL_SUMMARY_REPORT_*.csv
```

---

## Thời gian ước tính

| Setup | Thời gian |
|-------|-----------|
| 1 GPU sequential | ~30-60 giờ |
| 2 GPUs parallel | ~15-30 giờ |
| 4 GPUs parallel | ~8-15 giờ |

*Mỗi run: ~30-60 phút (tùy early stopping)*

---

## Troubleshooting nhanh

```bash
# Out of memory → giảm batch size
# Edit dòng 8 trong script: BATCH_SIZE=8

# Xem lỗi chi tiết
cat ./results/run_davis_warm_fold0.log

# Chạy lại 1 run cụ thể
python code/train.py --fold 0 --cuda "0" --dataset davis --running_set warm --epochs 200 --batch_size 16
```

---

**Đọc thêm:** `EXPERIMENT_GUIDE.md` để biết chi tiết
