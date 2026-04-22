# Project Structure

## 1. Core Source Files

- `run_llama.py`: Entry point to run `generate`, `prompt`, and `finetune` modes.
- `base_llama.py`: Base pretrained-model wrapper and shared helpers.
- `llama.py`: Main Llama implementation (RMSNorm, attention layer stack, forward, generate, checkpoint loading).
- `rope.py`: Rotary positional embedding implementation.
- `classifier.py`: Zero-shot classifier and embedding-based finetuning classifier head.
- `optimizer.py`: AdamW optimizer implementation.
- `config.py`: Model/config definitions.
- `tokenizer.py`: SentencePiece tokenizer wrapper.
- `utils.py`: Utility helpers (I/O, cache/config helpers).

## 2. Validation / Tests

- `sanity_check.py`: Integration sanity check for Llama forward pass.
- `rope_test.py`: Unit test for RoPE implementation.
- `optimizer_test.py`: Unit test for AdamW step implementation.
- `sanity_check.data`: Reference tensors for sanity check.
- `rotary_embedding_actual.data`: Reference tensors for RoPE unit test.
- `optimizer_test.npy`: Reference data for optimizer test.

## 3. Data

- `data/sst-train.txt`, `data/sst-dev.txt`, `data/sst-test.txt`
- `data/cfimdb-train.txt`, `data/cfimdb-dev.txt`, `data/cfimdb-test.txt`
- `data/sst-label-mapping.json`, `data/cfimdb-label-mapping.json`

## 4. Model Weights / Checkpoints

- `stories42M.pt`: Pretrained TinyStories Llama checkpoint.
- `finetune-1-2e-05.pt`: Finetuned classifier checkpoint (1 epoch, CFIMDB setup).
- `finetune-5-2e-05.pt`: Finetuned classifier checkpoint (5 epochs, CFIMDB setup).

## 5. Generated Outputs

- Text generation:
  - `generated-sentence-temp-0.txt`
  - `generated-sentence-temp-1.txt`
- Prompting outputs:
  - `sst-dev-prompting-output.txt`
  - `sst-test-prompting-output.txt`
  - `cfimdb-dev-prompting-output.txt`
  - `cfimdb-test-prompting-output.txt`
- Finetuning outputs:
  - `sst-dev-finetuning-output.txt`
  - `sst-test-finetuning-output.txt`
  - `cfimdb-dev-finetuning-output.txt`
  - `cfimdb-test-finetuning-output.txt`

## 6. How To Run

1. Setup:
   - `bash setup.sh`
2. Generate mode:
   - `python3 run_llama.py --option generate --pretrained-model-path stories42M.pt`
3. Prompting mode:
   - `python3 run_llama.py --option prompt --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt --label-names data/sst-label-mapping.json --dev_out sst-dev-prompting-output.txt --test_out sst-test-prompting-output.txt`
4. Finetune mode:
   - `python3 run_llama.py --option finetune --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --label-names data/cfimdb-label-mapping.json --epochs 5 --lr 2e-5 --batch_size 10 --dev_out cfimdb-dev-finetuning-output.txt --test_out cfimdb-test-finetuning-output.txt`

## 7. Submission Notes

- Required report file path/name: `Group_ID-report.pdf`
- This repository now includes:
  - `structure.md` (this file)
  - `Group_ID-report.pdf`
