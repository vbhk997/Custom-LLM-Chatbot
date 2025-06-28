# 🧠 Falcon LegalBot — Fine-Tuning Falcon with LoRA for Indian Legal QA

This project implements a custom legal question-answering chatbot using the Falcon-RW-1B language model, fine-tuned with LoRA (Parameter-Efficient Fine-Tuning) on a dataset of legal instructions and responses.

## 📌 Highlights

- ✅ Fine-tuned using [LoRA (PEFT)](https://huggingface.co/docs/peft) on top of `tiiuae/falcon-rw-1b`
- ✅ Used 4-bit quantization via `bitsandbytes` for low-memory training
- ✅ Trained on a custom dataset of 10,000 legal Q&A examples
- ✅ Achieved stable convergence in under 3 epochs (loss ≈ 3.02)
- ✅ Optimized for low-cost training on Colab with `fp16`, batching & gradient accumulation

---

## 🚀 Quick Overview

### Dataset
Custom-built legal Q&A dataset in the following format:
```json
{
  "instruction": "Explain Section 498A of the IPC.",
  "output": "Section 498A deals with cruelty to a woman by her husband or his relatives..."
}
```

### Training Framework
- Hugging Face `transformers`
- `peft` with LoRA (8- or 16-rank adapters)
- PyTorch + `accelerate` for hardware efficiency

---

## 🧪 Inference Example

```python
prompt = "### Instruction:\nWhat is the punishment for dowry death?\n\n### Response:\n"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=150)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## 📦 Installation

```bash
pip install -r requirements.txt
```

You’ll also need a CUDA-enabled GPU if not running on Colab.

---

## 🗃️ Files Included

- `legal_bot_training.ipynb` — Full Colab notebook for training
- `legal_bot_dataset.json` — Custom dataset (10,000 entries)
- `requirements.txt` — All dependencies
- `README.md` — This file

---

## 🤖 Future Work (Optional Enhancements)

- Integrate RAG for external legal document lookup
- Add Gradio UI for interactive legal chatbot
- Deploy model to Hugging Face Spaces or Streamlit
