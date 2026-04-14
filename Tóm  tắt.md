

### 1. Cấu Trúc cần nộp cho cô
```text
File zip GROUP_ID này bao gồm:

├── [Hỗ trợ - Không sửa]
│   ├── run_llama.py         (Trung tâm điều khiển)
│   ├── base_llama.py
│   ├── config.py
│   ├── tokenizer.py
│   ├── utils.py
│   ├── setup.sh
│   └── sanity_check.py      (Kiểm tra nhanh hệ thống)
│
├── [Mã nguồn chính - CẦN LÀM]
│   ├── llama.py             (Kiến trúc Transformer chính)
│   ├── rope.py              (Nhúng vị trí)
│   ├── classifier.py        (Lớp phân loại cảm xúc)
│   └── optimizer.py         (Thuật toán AdamW)
│
├── [Dữ liệu & Tài liệu]
│   ├── README.md
│   ├── structure.md
│   ├── sanity_check.data
│   └── Group_ID-report.pdf  (Báo cáo tối đa 3 trang)
│
└── [File Output sinh ra sau khi chạy]
    ├── generated-sentence-temp-0.txt        (Sinh văn bản)
    ├── generated-sentence-temp-1.txt        (Sinh văn bản)
    ├── sst-dev-prompting-output.txt         (Zero-shot SST)
    ├── sst-test-prompting-output.txt        (Zero-shot SST)
    ├── sst-dev-finetuning-output.txt        (Finetune SST - Quan trọng)
    ├── sst-test-finetuning-output.txt       (Finetune SST - Quan trọng)
    ├── cfimdb-dev-prompting-output.txt      (Zero-shot CFIMDB)
    ├── cfimdb-test-prompting-output.txt     (Zero-shot CFIMDB)
    ├── cfimdb-dev-finetuning-output.txt     (Finetune CFIMDB - Quan trọng)
    ├── cfimdb-test-finetuning-output.txt    (Finetune CFIMDB - Quan trọng)
    │
    └── [Tùy chọn - Nếu làm nâng cao]
        ├── sst-dev-advanced-output.txt
        ├── sst-test-advanced-output.txt
        ├── cfimdb-dev-advanced-output.txt
        └── cfimdb-test-advanced-output.txt
```

---

### 2. Hướng dẫn Dự án: Minimalist Llama2

#### A. Mục tiêu & Yêu cầu Tham chiếu (Benchmark)
* **Mục tiêu:** Tái cấu trúc kiến trúc Llama2, dùng trọng số `stories42M.pt` để sinh văn bản và phân loại cảm xúc (SST, CFIMDB).
* **Zero-Shot Prompting:**
    * SST: Dev ~ 0.213 | Test ~ 0.224
    * CFIMDB: Dev ~ 0.498 (hoặc 0.502)
* **Classification Fine-tuning:**
    * SST: Dev ~ 0.414 | Test ~ 0.418
    * CFIMDB: Dev ~ 0.800 (có thể lên 0.882 tùy seed)
* *Lưu ý:* Tập test của CFIMDB có nhãn giả định (-1) nên accuracy sẽ rất thấp, không cần lo lắng.

#### B. Chi tiết triển khai mã nguồn (#TODO)
**1. Kiến trúc mô hình:**
* `rope.py`: Hoàn thiện `apply_rotary_emb` để gán vị trí cho vector.
* `llama.py`:
    * `Attention.forward`: Triển khai Grouped-Query Attention (GQA).
    * `RMSNorm.norm`: Triển khai chuẩn hóa RMS.
    * `Llama.forward`: Xây dựng luồng dữ liệu qua encoder.
    * `Llama.generate`: Sinh văn bản bằng temperature sampling.

**2. Huấn luyện và Fine-tuning:**
* `classifier.py`: Hoàn thiện `LlamaSentClassifier`. Lấy hidden representation từ **từ cuối cùng** của câu, áp dụng dropout và đưa qua linear layer để dự đoán.
* `optimizer.py`: Viết hàm `step()` cho AdamW. Chú ý bias correction và tích hợp learning rate vào weight decay.

#### C. Quy trình Kiểm tra (Unit Tests)
Chạy các lệnh sau và đảm bảo không có lỗi trước khi huấn luyện thực sự:
```bash
!python rope_test.py
!python optimizer_test.py
!python sanity_check.py
```

#### D. Các lệnh chạy trên Google Colab (ví dụ)
*(Đã bao gồm cờ `--use_gpu`, hãy đảm bảo Colab đang bật GPU)*

**1. Sinh văn bản (Text Continuation)**
```bash
!python run_llama.py --option generate
```

**2. Zero-Shot Prompting**
```bash
# Cho dữ liệu SST:
!python run_llama.py --option prompt --batch_size 10 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt --use_gpu

# Cho dữ liệu CFIMDB:
!python run_llama.py --option prompt --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-prompting-output.txt --test_out cfimdb-test-prompting-output.txt --use_gpu
```

**3. Fine-tuning**
```bash
# Cho dữ liệu SST:
!python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 80 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-finetuning-output.txt --test_out sst-test-finetuning-output.txt --use_gpu

# Cho dữ liệu CFIMDB:
!python run_llama.py --option finetune --epochs 5 --lr 2e-5 --batch_size 10 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt --use_gpu
```

#### E. Quy định bắt buộc
1.  **Không dùng thư viện ngoài:** Chỉ dùng cài đặt từ `setup.sh` (không import `transformers`).
2.  **Giữ nguyên tên biến:** Không đổi tên các tham số Llama2 để tránh lỗi nạp file `stories42M.pt`.Chào bạn, việc tự tay triển khai các mô hình ngôn ngữ lớn như Llama2 sẽ giúp củng cố rất sâu các kiến thức về kiến trúc NLP và Deep Learning mà bạn đang nghiên cứu. 

Để đảm bảo khi bạn copy và dán thuần văn bản (Ctrl + Shift + V) không bị vỡ định dạng, mình đã gom nhóm, căn chỉnh lại khoảng trắng và làm nổi bật các phần mã lệnh. Bạn có thể sao chép toàn bộ nội dung bên dưới:



#### F. Công việc
```bash

Dưới đây là nội dung mã **`.md`** được thiết kế riêng cho phần giao nhiệm vụ của nhóm bạn, dựa trên các yêu cầu từ tài liệu và kế hoạch đã thảo luận:

```markdown
# Phân chia Nhiệm vụ Nhóm: Dự án Phát triển Mô hình Llama2 Tối giản

## 1. Mục tiêu và Chỉ số Độ chính xác (Benchmarks)
Nhóm cần phối hợp để đạt được các con số tham chiếu sau đây:

*   **Zero-Shot Prompting (SST):** Dev Acc ~ **0.213**.
*   **Zero-Shot Prompting (CFIMDB):** Dev Acc ~ **0.498**.
*   **Fine-tuning (SST):** Dev Acc ~ **0.414**.
*   **Fine-tuning (CFIMDB):** Dev Acc ~ **0.800** (có thể đạt 0.882 tùy seed).

---

## 2. Giai đoạn 1: Xây dựng nền móng (Người 1 & A - Ưu tiên làm trước)
Mục tiêu của giai đoạn này là tạo ra "xương sống" ổn định cho toàn bộ mô hình.

### Người 1: Kỹ sư Hệ thống & Nền tảng
*   **File sửa:** `llama.py`.
*   **Nhiệm vụ:**
    *   **Thiết lập:** Chạy `setup.sh` để chuẩn bị môi trường và nạp trọng số `stories42M.pt`.
    *   **Triển khai:** Viết mã cho **`llama.RMSNorm.norm`** và **`llama.Llama.forward`** (kết nối token embeddings, encoder layers và projection layer).
*   **Trách nhiệm:** Đảm bảo vượt qua bài test tích hợp **`sanity_check.py`**.

### Người A + 1: Chuyên gia Kiến trúc (Xử lý phần khó nhất)
*   **File sửa:** `rope.py`, `llama.py`.
*   **Nhiệm vụ:**
    *   **Triển khai:** Viết hàm **`rope.apply_rotary_emb`** (nhúng vị trí quay) – đây là phần rất lắt léo.
    *   **Triển khai:** Viết mã cho **`llama.Attention.forward`** theo cơ chế **Grouped-Query Attention (GQA)**.
*   **Trách nhiệm:** Đảm bảo vượt qua bài test **`RoPE_test.py`**.

---

## 3. Giai đoạn 2: Bộ máy & Ứng dụng (Người B & C - Làm song song sau khi có khung) 
Giai đoạn này tập trung vào việc giúp mô hình "học" và áp dụng vào bài toán phân loại.

### Người B: Chuyên gia Tối ưu
*   **File sửa:** `optimizer.py`.
*   **Nhiệm vụ:**
    *   **Triển khai:** Hoàn thiện hàm **`optimizer.AdamW.step`** với hiệu chỉnh độ lệch (bias correction) và tích hợp learning rate vào weight decay.
*   **Trách nhiệm:** Đảm bảo vượt qua bài test **`optimizer_test.py`**.

### Người C: Chuyên gia Ứng dụng & Huấn luyện
*   **File sửa:** `classifier.py`, `llama.py`.
*   **Nhiệm vụ:**
    *   **Triển khai:** Viết lớp **`LlamaSentClassifier`** để lấy hidden representation từ từ cuối cùng của câu và thực hiện phân loại.
    *   **Triển khai:** Viết hàm **`llama.Llama.generate`** sử dụng kỹ thuật **temperature sampling**.
*   **Trách nhiệm:** Đạt độ chính xác mục tiêu khi chạy lệnh `--option finetune` và xuất các file `.txt` kết quả.

---
Sau khi người B và C hoàn thành xong 2 file. Up code lên git cho mọi người. Nếu chạy ko thành công báo cho A và 1 cùng nhau làm thử bộ tham số khác

## 4. Quy trình chung: Chạy - Kiểm tra - Cải tiến - viết báo cáo (Nhiệm vụ của cả nhóm)
Việc chạy ra các file `.txt` là cách duy nhất để kiểm tra tính hiệu quả của mã nguồn và tìm hướng cải thiện:

1.  **Chạy lệnh sinh văn bản:** Kiểm tra tính trôi chảy của tiếng Anh để xác nhận kiến trúc (Người 1 & A).
2.  **Chạy lệnh Fine-tuning:** Theo dõi Accuracy. Nếu kết quả thấp hơn tham chiếu, cả nhóm cùng rà soát lại bộ tối ưu (Người B) và lớp phân loại (Người C).
3.  **Tối ưu siêu tham số:** Cùng thử nghiệm các mức `learning rate`, `batch size` hoặc `epochs` khác nhau để tìm ra kết quả tốt nhất cho báo cáo.




```
####  G. Quy trình Phối hợp và Cải tiến Dự án (Cập nhật)
```bash
 Tạo git chung và đẩy file code đổi lên đó

 
## 1. Giai đoạn Hoàn thiện và Chia sẻ (Người B & C)
Sau khi **Người B** hoàn thiện `optimizer.py` (AdamW) và **Người C** hoàn thiện `classifier.py` (LlamaSentClassifier), hai thành viên thực hiện các bước sau,:
*   **Đẩy mã nguồn lên Git:** Tải toàn bộ mã nguồn đã hoàn thiện lên kho lưu trữ chung (Git) để tất cả thành viên (1, A, B, C) có bản cập nhật mới nhất.
*   **Đồng bộ hóa:** Người 1 và Người A nạp mã mới về môi trường làm việc của mình (Colab/Kaggle) để sẵn sàng hỗ trợ kiểm tra tính tương thích với "xương sống" của mô hình,.

## 2. Quy trình Xử lý khi Huấn luyện không thành công
Nếu quá trình chạy Fine-tuning (Setting 3) không đạt được độ chính xác tham chiếu hoặc phát sinh lỗi, nhóm sẽ kích hoạt quy trình phối hợp,:

*   **Báo cáo phản hồi:** Người B và C báo cáo ngay cho **Người 1 và Người A**.
*   **Phối hợp rà soát:**
    *   **Người 1 & A:** Kiểm tra xem lỗi có nằm ở kiến trúc mô hình (`llama.py`, `rope.py`) khiến dữ liệu đầu vào của bộ phân loại bị sai lệch hay không,.
    *   **Người B & C:** Kiểm tra lại logic tính toán trong bộ tối ưu AdamW (đặc biệt là bias correction) và lớp phân loại (`LlamaSentClassifier`),.

## 3. Thử nghiệm Bộ tham số mới (Hyperparameter Tuning)
Khi cấu trúc mã đã đúng nhưng độ chính xác chưa đạt mục tiêu, cả 4 thành viên cùng phối hợp thử nghiệm các **siêu tham số (hyperparameters)** khác với mặc định để tìm ra kết quả tốt nhất,:

*   **Các tham số cần thử nghiệm:**
    *   **Tốc độ học (`--lr`):** Thử các mức quanh giá trị `2e-5`,.
    *   **Kích thước lô (`--batch_size`):** Thử thay đổi để tối ưu hóa bộ nhớ GPU và tốc độ hội tụ (ví dụ: SST dùng 80, CFIMDB dùng 10),.
    *   **Số lượng epoch (`--epochs`):** Mặc định là 5, có thể điều chỉnh để tránh quá khớp (overfitting) hoặc chưa khớp (underfitting),.
*   **Công cụ hỗ trợ:** Nhóm được khuyến khích sử dụng **Colab hoặc GPU cá nhân** để có thể chạy nhiều thí nghiệm song song và "lặp lại quy trình thực nghiệm nhanh hơn" (iterate more quickly),.

## 4. Mục tiêu nghiệm thu (Benchmarks)
Nhóm chỉ dừng việc thử nghiệm khi đạt được (hoặc vượt) các con số trung bình sau,:
*   **Fine-tuning SST:** Dev Accuracy ~ **0.414** .
*   **Fine-tuning CFIMDB:** Dev Accuracy ~ **0.800** .


```
