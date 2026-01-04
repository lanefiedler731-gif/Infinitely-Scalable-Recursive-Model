
# GGUF Conversion Workflow

To reliably convert the `SmallLM` (Qwen2-based) model to GGUF for Ollama:

## 1. Export to HuggingFace Format
Run the export script to convert the PyTorch checkpoint to a HF-compatible directory. This handles config mapping and tensor renaming.
```bash
.venv/bin/python export_to_hf.py
```
This creates a `hf_export/` directory.

## 2. Convert to GGUF
Use the official `llama.cpp` conversion script.
Note: You may need an older version of the script (e.g. from commit `b3500`) if the latest `gguf` pip package is incompatible with the master branch script.
```bash
wget https://raw.githubusercontent.com/ggerganov/llama.cpp/b3500/convert_hf_to_gguf.py -O convert_hf_to_gguf.py
.venv/bin/python convert_hf_to_gguf.py hf_export/ --outfile outputs/best_model.gguf --outtype f32
```

## 3. Import to Ollama
Create a model using the `Modelfile`.
```bash
ollama create qwen-custom -f Modelfile
ollama run qwen-custom "Hello!"
```
