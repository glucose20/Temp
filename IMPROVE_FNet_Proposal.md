# Đề xuất cải tiến vượt FNet Fourier Mixing hiện tại

**Dựa trên:** `related works/` (2 paper mới thêm vào repo) + `res19_7/` (kết quả FNet hiện tại) + research bổ sung
**Repo tham chiếu:** `glucose20org/Temp` (nhánh `quantum`) — kiến trúc hiện tại: `LLMDTA_FNet` (`code/train_fnet.py`), thay `CrossAttention` bằng `FourierCrossMixing` không tham số (`torch.fft.fft` 2 chiều, lấy phần thực)

---

## 1. Hiện trạng: kết quả FFT (FNet) hiện tại

Từ `res19_7/results.csv` và `results_kb.csv` (checkpoint `fnet_moe`, `code/train_fnet.py`):

| Dataset | Setting | MSE ↓ | CI ↑ | R² | #runs |
|---|---|---|---|---|---|
| Davis | warm | 0.214 | 0.891 | 0.732 | 5 |
| Davis | novel-drug | 0.669 | 0.708 | 0.129 | 5 |
| Davis | novel-prot | 0.489 | 0.801 | 0.370 | 5 |
| Davis | novel-pair | 0.738 | 0.697 | 0.145 | 5 |
| KIBA | warm | 0.202 (results_kb) / 0.222 (results) | 0.858–0.870 | 0.68–0.71 | 3–5 |
| KIBA | novel-prot | 0.310–0.399 | 0.76–0.77 | 0.41–0.56 | 2–5 |
| Metz | warm | 0.310 | 0.807 | 0.662 | 3 |

**Quan sát quan trọng — cần cảnh giác trước khi kết luận FFT "thắng":**
- `results_kb.csv` cho thấy baseline (CrossAttention) trên KIBA-warm có `mse=0.826, r2=-0.17, mse_std=1.20` — dấu hiệu **huấn luyện không ổn định** (có thể là 1 seed bị divergence kéo lệch trung bình), không phải bằng chứng chắc chắn rằng CrossAttention kém hơn FFT một cách bản chất.
- Nhiều dòng có `num_runs = 1–2` (kiba novel-drug, novel-pair, metz novel-pair) — chưa đủ để paired t-test như protocol trong `REPORT_Fourier_Mixing_DTA.md` §4.3 yêu cầu (≥3 seed × 5-fold).

→ **Khuyến nghị 0 (bắt buộc trước khi tối ưu tiếp):** chạy lại baseline KIBA-warm với ≥3 seed, loại bỏ nghi ngờ divergence, rồi mới dùng làm mốc so sánh "cao hơn FFT hiện tại". Nếu không, mọi cải tiến sau đều so sánh với một baseline nhiễu.

---

## 2. Research bổ sung

### 2.1 Hai paper trong `related works/` (đọc trực tiếp từ PDF)

**[MixingDTA](https://github.com/rokieplayer20/MixingDTA)** (Kim et al., *Bioinformatics* 2025, ISMB/ECCB Supplement) — chính là repo GitHub bạn gửi:
- **MEETA backbone**: thay pretrain embedding mol2vec/ESM2 hiện tại bằng **MolFormer** (drug, transformer hoá học pretrain) + **ESM3** (protein). Riêng việc đổi backbone này đã cho **tới 19% cải thiện MSE** so với SOTA cũ, *trước khi* thêm augmentation.
- **AFA (Attention-Free Aggregation)**: thay multi-head cross-attention bằng phép tổng hợp không cần dot-product (`softmax(K)⊙V` theo token), độ phức tạp **O(T)** — rẻ hơn cả attention O(T²) lẫn FFT O(T log T), và trong ablation của paper AFA cho CI/MSE tốt hơn MHA (Table 4: MHA CI=0.87 → AFA CI=0.89).
- **GBA-Mixup**: augmentation dữ liệu dựa trên nguyên lý "guilt-by-association" — mixup không phải giữa cặp D-T ngẫu nhiên, mà giữa các cặp có **chung drug hoặc chung protein**, trọng số mixup dựa trên độ tương đồng affinity. Đóng góp thêm **+8.4% MSE** trên nền MEETA, và quan trọng: **model-agnostic**, áp dụng được cho bất kỳ backbone nào (đã test trên DeepDTA, AttentionDTA, MEETA — tất cả đều cải thiện, kể cả cold-start: MSE giảm tới −16.9% cho ML-DTI).
- Case study cho thấy GBA-Mixup giúp mô hình định vị đúng residue liên kết (binding site) tốt hơn AttentionDTA dù không có input cấu trúc 3D.

**DCI-SiteDTA** (Zhang et al., *BMC Bioinformatics* 2026):
- **Multi-scale feature fusion**: Conv1D (kernel k) + Self-Attention song song rồi concat — sweep k ∈ {9,7,5,3}, **k=5 là tối ưu** trên cả Davis và KIBA (ACC, CI, RMSE đều tốt nhất). Đây là finding trực tiếp áp dụng được: `Encoder` hiện tại trong `code/LLMDTA.py:200` đang **cố định `kernel_size=7`**.
- **Binding-site-guided masking**: dùng xác suất binding-site dự đoán để soft-mask + giữ lại Top-K context ngoài vùng tin cậy cao (tránh mất thông tin do hard mask), sau đó **gated weighted pooling** (trọng số hoá theo binding score thay vì average pooling thường).
- **Dual cross-interaction**: 2 pathway MHA riêng (ligand-view: drug là query; protein-view: protein là query) rồi concat — thay vì 1 chiều cross-attention.
- Giới hạn: cần **binding-site ground truth** (từ PDBbind/scPDB/BioLiP), hiện repo Temp chưa có nhãn này → chi phí triển khai cao hơn.

### 2.2 Paper tìm thêm (research bổ sung theo yêu cầu)

| Công trình | Ý tưởng chính | Liên hệ trực tiếp tới `FourierCrossMixing` hiện tại |
|---|---|---|
| **[GFNet — Global Filter Networks](https://arxiv.org/abs/2107.00645)** (Rao et al., NeurIPS 2021 / T-PAMI 2023) | Thay vì FFT thuần "không tham số" như FNet, GFNet nhân phổ Fourier với **filter học được** (element-wise, per-frequency) trước khi biến đổi ngược. Vẫn O(L log L), chỉ thêm ~d tham số phức mỗi tầng. | Đây **chính là** "biến thể bán tham số" mà `REPORT_Fourier_Mixing_DTA.md` §6 đã tự đề xuất nhưng chưa cài đặt. Sửa trực tiếp trong `FourierCrossMixing.forward()`. |
| **[AFNO — Adaptive Fourier Neural Operator](https://arxiv.org/abs/2111.13587)** (Guibas et al., ICLR 2022) | Token-mixing trong miền tần số với **block-diagonal weight matrix** (trộn kênh, không chỉ scale từng tần số) + **soft-thresholding/shrinkage** để triệt tiêu tần số nhiễu. Complexity vẫn quasi-linear. | Mạnh hơn GFNet (mixing chứ không chỉ gating), và cơ chế shrinkage đặc biệt phù hợp với protein sequence dài (tới 1022–2048 residue) có nhiều vùng không liên quan tới binding site — đóng vai trò lọc nhiễu tương tự binding-site masking của DCI-SiteDTA nhưng **không cần nhãn**. |
| **[FFTNet (2025)](https://arxiv.org/html/2502.18394v1)** — "The FFT Strikes Back: An Efficient Alternative to Self-Attention" | Chỉ ra hạn chế cốt lõi của FNet: FFT tĩnh không thích ứng theo input/task. Đề xuất **adaptive spectral filtering** — filter phụ thuộc vào input thay vì filter cố định (khác GFNet ở chỗ GFNet học filter *tĩnh* sau train, còn FFTNet sinh filter *động* theo từng sample). | Xác nhận đúng giới hạn mà FourierCrossMixing hiện tại gặp phải (0 tham số ⇒ không thích ứng); là bằng chứng thời sự (2025) cho hướng "thêm một chút adaptivity vào FFT" thay vì bỏ FFT hoàn toàn. |
| **[FNet-Hybrid layouts](https://arxiv.org/abs/2105.03824)** (trong chính paper FNet gốc, phần ablation) | Không cần thay *toàn bộ* attention bằng FFT: đặt 1–2 tầng attention thật ở **TOP** (gần đầu ra) trong khi các tầng còn lại dùng FFT cho kết quả tốt nhất về accuracy/speed trade-off, tốt hơn cả full-attention và full-FFT ở một số cấu hình. | Gợi ý: **không cần chọn giữa FFT và CrossAttention "một trong hai"** như ablation hiện tại (`ablation_fft.py` có variant `attn_mixing` thay hẳn FFT bằng attention) — thử kết hợp cả hai theo layout thay vì thay thế. |
| **[Frequency Enhanced Hybrid Attention (2023)](https://arxiv.org/abs/2304.09184)** | Kiến trúc recommendation kết hợp song song 1 nhánh frequency-domain + 1 nhánh attention, **gate học được** quyết định trọng số trộn 2 nhánh theo từng sample. | Blueprint cụ thể cho biến thể "Gated Fourier-Attention" ở Tier 2 bên dưới — khác GFNet/AFNO ở chỗ giữ nguyên cả 2 nhánh full thay vì sửa nội bộ FFT. |
| **[CS-DTA (2026)](https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2026.1834317/full)**, **[KANPM-DTA (2026)](https://academic.oup.com/bib/article/27/2/bbag112/8516647)** | SOTA cold-start DTA 2026: đều dùng **ChemBERTa-2/ESM-C** (KANPM-DTA) hoặc **ChemBERTa+ESM2** (CS-DTA) làm embedding pretrain, không dùng FFT/spectral mixing — cải thiện đến từ embedding chất lượng cao + kiến trúc fusion tốt, không phải từ cơ chế mixing. | Xác nhận thêm (độc lập với MixingDTA) rằng **nâng cấp embedding** là đòn bẩy lớn, đúng hướng `ESMC_MIGRATION_GUIDE.md` đã chuẩn bị sẵn trong repo nhưng **chưa từng chạy để lấy số liệu**. |

---

## 3. Đề xuất cải tiến — xếp theo chi phí/lợi ích

### Tier 1 — Chi phí thấp, khả năng thắng cao, làm trước

**(1) Bật ESM-C + MolFormer cho FNet model (chưa từng đo)**
Hạ tầng đã có sẵn (`ESMC_MIGRATION_GUIDE.md`, `code_prepareEmb/_PreparePretrain_ESMC.ipynb`) nhưng **chưa có số liệu DTA thực tế** — file `res19_7` chỉ có kết quả với ESM2/mol2vec. Đây là thay đổi rẻ nhất về công sức kỹ sư (đã code sẵn) và theo MixingDTA có thể mang lại cải thiện lớn nhất trong tất cả đề xuất (tới 19% MSE ở paper gốc, dù backbone khác nên cần tự đo). MolFormer cho drug hiện chưa có sẵn migration guide — cần thêm tương tự ESM-C.
→ Việc cần làm: sinh `davis/kiba/metz_esmc_pretrain.pkl`, set `use_esmc=True`, chạy lại `train_fnet.py` trên davis-warm trước (rẻ nhất) để có tín hiệu nhanh.

**(2) GBA-Mixup (từ MixingDTA) — augmentation, không đổi kiến trúc**
Cắm trực tiếp vào loop hiện tại của `code/train_fnet.py` mà **không cần sửa `FourierCrossMixing`**: mở rộng `MyDataset.py`/collate để, với xác suất p, mix 2 cặp D-T chung drug hoặc chung protein (dùng `pairs.csv` đã có sẵn để build neighbor index theo drug_id/prot_id), nội suy `(x_mix, y_mix) = λx_i+(1-λ)x_j`. Vì mục tiêu chính của LLMDTA/report là **cold-start (novel-drug/prot/pair)**, và GBA-Mixup được thiết kế đặc biệt cho đúng bài toán "cặp lân cận chia sẻ drug/protein" — đây là hướng khớp nhất với mục tiêu đã nêu trong `REPORT_Fourier_Mixing_DTA.md`.
→ Ưu tiên áp GBA-Mixup vào 3 setting đang yếu nhất hiện nay: `davis novel-drug` (CI=0.708), `davis novel-pair` (CI=0.697) — đây là chỗ MixingDTA báo cáo mức cải thiện lớn nhất (cold-drug).

**(3) Learnable spectral filter (GFNet-style) thay `FourierCrossMixing` thuần FFT**
Sửa tối thiểu trong `train_fnet.py`:
```python
class LearnableFourierMixing(nn.Module):
    def __init__(self, hidden_dim, seq_len, dropout=0.1):
        super().__init__()
        # filter phức học được, riêng theo mỗi vị trí tần số x kênh
        self.complex_weight = nn.Parameter(torch.randn(seq_len, hidden_dim // 2 + 1, 2) * 0.02)
        self.out_ln = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key_value):
        seq_q = query.shape[1]
        combined = torch.cat([query, key_value], dim=1)
        x_freq = torch.fft.rfft(combined, dim=-1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_freq = x_freq * weight[:combined.shape[1]].unsqueeze(0)
        mixed = torch.fft.irfft(x_freq, n=combined.shape[-1], dim=-1, norm='ortho')
        mixed_query = self.dropout(mixed[:, :seq_q, :])
        return self.out_ln(mixed_query + query)
```
Chi phí tham số: `seq_len × (d/2+1)` số phức mỗi khối — với `d=128`, `seq_len≈1122` (drug+prot) ⇒ ~73K tham số/khối, tương đương baseline CrossAttention cũ nhưng **vẫn O(L log L)**. Đây là bước "dễ nhất về mặt lý thuyết" theo đúng gợi ý đã có sẵn trong báo cáo cũ (§6) — nên làm ablation `attn_mixing` hiện có trong `ablation_fft.py` thành 3 nhánh so sánh: `fft` (hiện tại) vs `learnable_fft` (mới) vs `attn` (CrossAttention cũ).

**(4) Sweep kernel size của `Encoder` (k=5 thay vì k=7 cố định)**
Rẻ nhất trong tất cả — chỉ đổi `self.kernel_size = 7` → thử `{3,5,7,9}` trong `code/LLMDTA.py:200`, dựa trên finding thực nghiệm của DCI-SiteDTA rằng k=5 tối ưu cho cả Davis và KIBA trong đúng bài toán binding-affinity. Không cần binding-site label vì đây chỉ là hyperparameter sweep, không phải toàn bộ module multi-scale fusion của họ.

### Tier 2 — Chi phí trung bình, cần thiết kế/code mới

**(5) AFNO-style mixing** (thay Tier-1-(3) nếu learnable filter đơn giản không đủ): thêm block-diagonal weight (chia `hidden_dim` thành `B` block, mỗi block có ma trận trộn tần số riêng) + soft-shrinkage (`sign(x)·relu(|x|-λ)`) trên phổ trước khi biến đổi ngược, để tự động triệt tiêu tần số nhiễu — đóng vai trò lọc nhiễu không giám sát, gần tương đương lợi ích của binding-site masking (DCI-SiteDTA) mà **không cần nhãn binding site**.

**(6) Gated Fourier-Attention (song song, không thay thế)**
```
h_fft  = FourierCrossMixing(query, key_value)
h_attn = CrossAttention(query, key_value)          # module cũ, giữ nguyên
gate   = sigmoid(Linear(query.mean(dim=1)))          # scalar/vector gate theo sample
output = gate * h_fft + (1 - gate) * h_attn
```
Theo finding của FNet-Hybrid (giữ 1 phần attention cho kết quả tốt hơn full-FFT) và Frequency-Enhanced-Hybrid-Attention. Chi phí tính toán tăng (chạy cả 2 nhánh) nhưng vẫn rẻ hơn nhiều so với thêm cả MoE thứ hai; đáng thử vì đây là hướng duy nhất "vừa giữ ưu điểm tốc độ của FFT, vừa lấy lại phần adaptivity mà FFT thuần đánh mất" — đúng giới hạn mà FFTNet (2025) chỉ ra.

### Tier 3 — Chi phí cao / dài hạn (đã có sẵn trong report cũ, giữ nguyên định hướng)

**(7) Binding-site-guided masking kiểu DCI-SiteDTA** — cần binding-site label (PDBbind/scPDB/BioLiP), phù hợp nếu sau này mở rộng dataset.
**(8) Quantum-inspired MoE gating / Variational Quantum Circuit thật** — như đã nêu ở `REPORT_Fourier_Mixing_DTA.md` §6, vẫn là hướng xa nhất, giữ nguyên vị trí ưu tiên thấp nhất.

---

## 4. Lộ trình thực nghiệm đề xuất (ưu tiên theo ROI / chi phí compute)

1. **Sửa baseline KIBA-warm không ổn định** (≥3 seed) → xác nhận lại mốc so sánh trước khi làm gì khác.
2. **Sweep kernel_size {3,5,7,9}** trên `davis warm` (rẻ, ~vài giờ/GPU) — nếu k=5 thắng, áp dụng cho toàn bộ thí nghiệm sau.
3. **Sinh ESM-C + thử MolFormer** cho drug, chạy lại `train_fnet.py` trên `davis warm` để có tín hiệu nhanh trước khi chạy full 5-fold × 3 datasets.
4. **Learnable Fourier filter (GFNet-style)** thay `FourierCrossMixing`, so sánh CI/MSE với bản FFT thuần trên cùng seed/split (ablation trong `ablation_fft.py`).
5. **GBA-Mixup**, ưu tiên áp trên `novel-drug`/`novel-pair` (nơi FFT hiện yếu nhất: CI 0.70/0.70) — đo riêng phần đóng góp của mixup (độc lập với kiến trúc mixing) bằng cách bật/tắt trên cùng 1 backbone (FNet đã chọn ở bước 4).
6. Nếu bước 4–5 đã cộng dồn cải thiện có ý nghĩa thống kê (paired t-test theo đúng protocol §4.3 cũ) → thử **Gated Fourier-Attention** (Tier 2) như bước tinh chỉnh cuối, vì tốn compute nhất trong các đề xuất ngắn hạn.
7. Ghi nhận kết quả theo đúng format `res19_7/results.csv` hiện có để so sánh trực tiếp cột-với-cột.

**Kỳ vọng tác động** (ước lượng định tính, cần đo thực tế): (1) và (4) là 2 đòn bẩy độc lập lớn nhất (embedding chất lượng + spectral mixing có tham số); (2) GBA-Mixup nhắm đúng cold-start — mục tiêu chính của paper LLMDTA gốc mà FFT hiện tại đang yếu nhất; (3) là "quick win" gần như miễn phí về công sức.

---

## 5. Tài liệu tham khảo bổ sung (ngoài 9 mục đã có trong `REPORT_Fourier_Mixing_DTA.md`)

10. Kim, Y. et al. "MixingDTA: improved drug–target affinity prediction by extending mixup with guilt-by-association." *Bioinformatics* 2025;41:i105–i114. https://doi.org/10.1093/bioinformatics/btaf238 — code: https://github.com/rokieplayer20/MixingDTA
11. Zhang, J. et al. "DCI-SiteDTA: drug-target affinity prediction based on binding sites detection and site-aware dual cross-interaction block." *BMC Bioinformatics* 2026;27:122. https://doi.org/10.1186/s12859-026-06446-8
12. Rao, Y. et al. "Global Filter Networks for Image Classification." NeurIPS 2021 / *IEEE T-PAMI* 2023. https://arxiv.org/abs/2107.00645
13. Guibas, J. et al. "Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers." ICLR 2022. https://arxiv.org/abs/2111.13587
14. "The FFT Strikes Back: An Efficient Alternative to Self-Attention" (FFTNet). 2025. https://arxiv.org/html/2502.18394v1
15. "Frequency Enhanced Hybrid Attention Network for Sequential Recommendation." 2023. https://arxiv.org/abs/2304.09184
16. "CS-DTA: a language model-driven framework for robust drug-target affinity prediction under strict cold-start scenarios." *Frontiers in Chemistry* 2026. https://www.frontiersin.org/journals/chemistry/articles/10.3389/fchem.2026.1834317/full
17. "KANPM-DTA: improving drug–target affinity prediction with Kolmogorov–Arnold networks and pretrained models." *Briefings in Bioinformatics* 2026;27(2):bbag112. https://academic.oup.com/bib/article/27/2/bbag112/8516647
