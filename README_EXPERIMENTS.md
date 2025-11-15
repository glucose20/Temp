# 📦 Files Created for Server Experiments

## Shell Scripts (executable)

### 1. **scripts/run_all_experiments.sh** ⭐ MAIN
- Chạy TẤT CẢ 60 experiments tuần tự (sequential)
- Phù hợp: 1 GPU, chạy an toàn
- Thời gian: ~30-60 giờ
- Tự động: log tracking, error handling, aggregation

**Chạy:**
```bash
bash scripts/run_all_experiments.sh
```

### 2. **scripts/run_all_experiments_parallel.sh** ⚡ FAST
- Chạy song song với nhiều GPU
- Phù hợp: 2-4 GPUs
- Thời gian: ~8-30 giờ (tùy số GPU)
- Tự động phân bổ jobs lên GPUs

**Chạy:**
```bash
# Edit NUM_GPUS và GPU_DEVICES trước
bash scripts/run_all_experiments_parallel.sh
```

### 3. **scripts/run_single_dataset.sh** 🎯 FOCUSED
- Chạy 1 dataset (20 runs: 4 settings × 5 folds)
- Phù hợp: test hoặc chia nhỏ workload
- Thời gian: ~10-20 giờ/dataset

**Chạy:**
```bash
bash scripts/run_single_dataset.sh davis
bash scripts/run_single_dataset.sh kiba
bash scripts/run_single_dataset.sh metz
```

### 4. **scripts/setup_server.sh** 🛠️ SETUP
- Setup tự động: dependencies, data download, extract
- Chỉ chạy 1 lần khi setup server mới

**Chạy:**
```bash
bash scripts/setup_server.sh
```

---

## Python Scripts

### 5. **code/aggregate_results.py** 📊 (ĐÃ CÓ SẴN)
- Tổng hợp kết quả từ 5 folds
- Tự động gọi trong shell scripts
- Có thể chạy thủ công

**Chạy:**
```bash
python code/aggregate_results.py --dataset davis --running_set warm
```

### 6. **code/generate_final_report.py** 📈 NEW
- Tạo báo cáo tổng hợp cuối cùng cho TẤT CẢ 60 runs
- So sánh giữa datasets và settings
- Xuất CSV và console output đẹp

**Chạy:**
```bash
python code/generate_final_report.py
```

---

## Documentation

### 7. **EXPERIMENT_GUIDE.md** 📖 DETAILED
- Hướng dẫn CHI TIẾT đầy đủ
- Setup, cách chạy, theo dõi, troubleshooting
- Ví dụ và tips

### 8. **QUICK_START_SERVER.md** ⚡ QUICK
- Quick start ngắn gọn
- Copy-paste commands
- Troubleshooting nhanh

### 9. **README_EXPERIMENTS.md** 📋 THIS FILE
- Tổng quan tất cả files
- Cái nào dùng khi nào

---

## Decision Tree - Chọn script nào?

```
Bạn có bao nhiêu GPU?
│
├─ 1 GPU
│  │
│  ├─ Muốn chạy tất cả 60 runs?
│  │  └─ YES → run_all_experiments.sh (sequential)
│  │
│  └─ Chỉ muốn test 1 dataset?
│     └─ run_single_dataset.sh davis
│
└─ 2+ GPUs
   │
   ├─ Muốn nhanh nhất?
   │  └─ run_all_experiments_parallel.sh (parallel)
   │
   └─ Muốn chia thủ công?
      └─ Mở 3 terminals, mỗi cái chạy 1 dataset:
         - Terminal 1: run_single_dataset.sh davis  (GPU 0)
         - Terminal 2: run_single_dataset.sh kiba   (GPU 1)
         - Terminal 3: run_single_dataset.sh metz   (GPU 2)
```

---

## Workflow hoàn chỉnh

```bash
# 1. Setup (1 lần duy nhất)
bash scripts/setup_server.sh

# 2. Test nhanh
python code/train.py --fold 0 --cuda "0" --dataset davis --running_set warm --epochs 2

# 3. Chạy TẤT CẢ experiments
tmux new -s llmdta
bash scripts/run_all_experiments.sh  # hoặc _parallel.sh

# Detach: Ctrl+B, D
# Check: tmux attach -t llmdta

# 4. Theo dõi
tail -f ./results/experiment_master_log_*.txt
watch -n 1 nvidia-smi

# 5. Tổng hợp kết quả (sau khi xong)
python code/generate_final_report.py

# 6. Download kết quả về local
scp -r user@server:/path/to/Temp/log ./
scp -r user@server:/path/to/Temp/savemodel ./
```

---

## File Structure sau khi chạy xong

```
Temp/
├── scripts/
│   ├── run_all_experiments.sh          ← MAIN: Sequential
│   ├── run_all_experiments_parallel.sh ← FAST: Parallel
│   ├── run_single_dataset.sh           ← FOCUSED: 1 dataset
│   └── setup_server.sh                 ← SETUP: Initialize
│
├── code/
│   ├── train.py                        ← Core training (modified)
│   ├── aggregate_results.py            ← Aggregate 5 folds
│   └── generate_final_report.py        ← Final report (NEW)
│
├── log/
│   ├── experiment_master_log_*.txt     ← Master tracking
│   ├── *-fold*.csv                     ← Individual runs (60 files)
│   ├── Test-*-AGGREGATED.csv           ← Per setting (12 files)
│   ├── Test-*-SUMMARY.csv              ← Statistics (12 files)
│   └── FINAL_SUMMARY_REPORT_*.csv      ← Final comparison
│
├── savemodel/
│   └── *.pth                           ← 60 model checkpoints
│
├── results/
│   ├── experiment_master_log_*.txt
│   └── run_*.log                       ← Per-run logs (60 files)
│
├── EXPERIMENT_GUIDE.md                 ← Detailed guide
├── QUICK_START_SERVER.md               ← Quick start
└── README_EXPERIMENTS.md               ← This file
```

---

## Checklist hoàn chỉnh

- [ ] Setup server: `bash scripts/setup_server.sh`
- [ ] Test run: `python code/train.py --fold 0 --cuda "0" --dataset davis --running_set warm --epochs 2`
- [ ] Chọn strategy (sequential/parallel/single)
- [ ] Start tmux session
- [ ] Run experiments script
- [ ] Monitor progress (tail -f log)
- [ ] Wait for completion (~8-60 hours)
- [ ] Generate final report: `python code/generate_final_report.py`
- [ ] Download results to local
- [ ] Celebrate! 🎉

---

## Support & Troubleshooting

**Common issues:**

1. **Out of memory**: Edit script, change `BATCH_SIZE=8`
2. **CUDA not available**: Check PyTorch installation
3. **File not found**: Run `setup_server.sh` again
4. **Script stops**: Use `tmux` or `nohup`
5. **Wrong GPU**: Edit `CUDA_DEVICE` in script

**Get help:**
- Read `EXPERIMENT_GUIDE.md` section "🚨 Xử lý lỗi"
- Check individual run logs in `./results/`
- Verify GPU: `nvidia-smi`
- Test single run first before batch

---

**Good luck with your experiments! 🚀**
