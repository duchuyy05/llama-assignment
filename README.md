# Assignment 1: Develop Minimalist of The Llama2

Triển khai một số thành phần cốt lõi của Llama2 để:

- sinh văn bản từ mô hình ngôn ngữ pretrained,
- phân loại cảm xúc bằng zero-shot prompting,
- fine-tune mô hình cho bài toán sentiment classification trên `SST` và `CFIMDB`.

## Tổng quan

Các phần được hiện thực trong project:

- `RoPE` trong [rope.py]
- `RMSNorm`, `Attention`, `Transformer block`, `generate()` trong [llama.py]
- classifier head trong [classifier.py]
- optimizer `AdamW` trong [optimizer.py]

Mô hình sử dụng checkpoint pretrained `stories42M.pt` để chạy inference và fine-tuning.

Cấu trúc chính:

```text
.
├── data/
│   ├── sst-*.txt
│   ├── cfimdb-*.txt
│   └── *-label-mapping.json
├── run_llama.py
├── llama.py
├── rope.py
├── classifier.py
├── optimizer.py
├── base_llama.py
├── config.py
├── tokenizer.py
├── utils.py
├── sanity_check.py
├── rope_test.py
├── optimizer_test.py
├── structure.md
├── setup.sh
├── README.md
├── generated-sentence-temp-0.txt
├── generated-sentence-temp-1.txt
├── sst-dev-prompting-output.txt
├── sst-test-prompting-output.txt
├── sst-dev-finetuning-output.txt
├── sst-test-finetuning-output.txt
├── cfimdb-dev-prompting-output.txt
├── cfimdb-test-prompting-output.txt
├── cfimdb-dev-finetuning-output.txt
├── cfimdb-test-finetuning-output.txt
└── *-advanced-output.txt
```

Dataset:

- `SST`: 5 lớp cảm xúc, label names: `awful`, `bad`, `average`, `good`, `excellent`
- `CFIMDB`: 2 lớp cảm xúc, label names: `bad`, `good`

Số lượng mẫu hiện có trong repo:

- `sst-train`: 8544
- `sst-dev`: 1101
- `sst-test`: 2210
- `cfimdb-train`: 1707
- `cfimdb-dev`: 245
- `cfimdb-test`: 488

## Cài đặt và chạy
Project cung cấp [setup.sh] để tạo môi trường và tải checkpoint
```bash
bash setup.sh
```

Script này sẽ:

- tạo môi trường `conda` tên `llama_hw`,
- cài `PyTorch`, `scikit-learn`, `sentencepiece`, `tokenizers`, ...
- tải file `stories42M.pt`.

Kiểm tra cài đặt:

Chạy 3 bài test sau:

```bash
python rope_test.py
python optimizer_test.py
python sanity_check.py
```

Nếu mọi thứ đúng, bạn sẽ thấy:

- `Rotary embedding test passed!`
- `Optimizer test passed!`
- `Your Llama implementation is correct!`

### 1. Sinh văn bản

```bash
python run_llama.py --option generate
```

Lệnh này sẽ sinh 2 file:

- `generated-sentence-temp-0.txt`
- `generated-sentence-temp-1.txt`

### 2. Zero-shot prompting

SST:

```bash
python run_llama.py \
  --option prompt \
  --train data/sst-train.txt \
  --dev data/sst-dev.txt \
  --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-prompting-output.txt \
  --test_out sst-test-prompting-output.txt
```

CFIMDB:

```bash
python run_llama.py \
  --option prompt \
  --train data/cfimdb-train.txt \
  --dev data/cfimdb-dev.txt \
  --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-prompting-output.txt \
  --test_out cfimdb-test-prompting-output.txt
```

### 3. Fine-tuning classifier

SST:

```bash
python run_llama.py \
  --option finetune \
  --epochs 5 \
  --lr 2e-5 \
  --batch_size 80 \
  --train data/sst-train.txt \
  --dev data/sst-dev.txt \
  --test data/sst-test.txt \
  --label-names data/sst-label-mapping.json \
  --dev_out sst-dev-finetuning-output.txt \
  --test_out sst-test-finetuning-output.txt
```

CFIMDB:

```bash
python run_llama.py \
  --option finetune \
  --epochs 5 \
  --lr 2e-5 \
  --batch_size 10 \
  --train data/cfimdb-train.txt \
  --dev data/cfimdb-dev.txt \
  --test data/cfimdb-test.txt \
  --label-names data/cfimdb-label-mapping.json \
  --dev_out cfimdb-dev-finetuning-output.txt \
  --test_out cfimdb-test-finetuning-output.txt
```

Nếu có GPU:

```bash
python run_llama.py --option finetune --use_gpu
```

### 4. Advanced (tùy chọn)

`run_llama.py` hiện chỉ hỗ trợ `--option generate|prompt|finetune` (không có `--option advanced`).
Vì vậy, setting `advanced` trong báo cáo được tạo bằng một bước hậu xử lý nhanh:
- huấn luyện `TF-IDF + LinearSVM` trên tập train,
- dùng confidence-based override để thay đổi dự đoán của finetune khi classifier cổ điển đủ tự tin.

Chạy tạo file advanced:

```bash
python3 build_advanced_outputs.py
```

Kết quả advanced hiện tại:
- SST Dev/Test: `0.4187 / 0.4448` (thay đổi `229` mẫu dev, `421` mẫu test so với finetune)
- CFIMDB Dev/Test: `0.9184 / 0.5020` (thay đổi `17` mẫu dev, `36` mẫu test so với finetune)

Các tham số chính:

Một số cờ CLI đang có trong [run_llama.py]:

- `--option`: `generate`, `prompt`, `finetune`
- `--pretrained-model-path`: đường dẫn tới `stories42M.pt`
- `--epochs`
- `--lr`
- `--batch_size`
- `--use_gpu`
- `--dev_out`, `--test_out`

Checkpoint fine-tuned sẽ được lưu theo mẫu:

```text
{option}-{epochs}-{lr}.pt
```

Ví dụ: `finetune-5-2e-05.pt`

[Link các phiên bản model nhóm đã finetune](https://drive.google.com/drive/u/3/folders/1QZIJKa7WZN2N_3J655M5OGiPde6WZnV9)
## Output và ghi chú

- Không dùng `transformers`; project chỉ dùng các thư viện được cài trong `setup.sh`.
- `sanity_check.py` phụ thuộc trực tiếp vào checkpoint `stories42M.pt`.
- Với bài toán zero-shot prompting, accuracy thường thấp hơn đáng kể so với fine-tuning.
- `*-advanced-output.txt` là phần tùy chọn. Bản hiện tại dùng TF-IDF + LinearSVM confidence override để khác biệt rõ so với finetune nhưng vẫn chạy rất nhanh.
- Project đi kèm [LICENSE] theo giấy phép MIT.
