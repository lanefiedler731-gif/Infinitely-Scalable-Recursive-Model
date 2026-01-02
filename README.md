# TestLabs Model (SmallLM)

**Note: This project is for fun/educational purposes. Model versions will be updated frequently.**

## Overview
This is a custom-built, experimental language model (~800M parameters) trained from scratch. It utilizes a "Deep and Thin" architecture designed to maximize reasoning depth while maintaining a small parameter footprint.

## Model Architecture
*   **Type:** LLaMA-style Transformer
*   **Parameters:** ~800M
*   **Layers:** 57
*   **Hidden Dimension:** 1024
*   **Context Window:** 1024 tokens
*   **Tokenizer:** Qwen 2.5 (151k vocab)
*   **Training Composition:**
    *   FP8 Training on NVIDIA Blackwell (RTX 5090)
    *   RoPE (Rotary Embeddings)
    *   SwiGLU Activations
    *   Grouped Query Attention (GQA)

## Training Data
The model is trained on a mix of high-quality reasoning and instruction datasets, including:
*   TeichAI Reasoning Datasets (MiMo, GLM, Claude-4.5 traces)
*   Open-Orca
*   OpenThoughts (1.2M reasoning examples)
*   Alpaca (Instruction tuning)

## Usage (Ollama)

Run this model directly with Ollama:

```bash
ollama run lanefiedler731/TestLabs
```

### Manual Installation (GGUF)
If you have the GGUF file locally:

1.  Create a `Modelfile`:
    ```dockerfile
    FROM ./best_model.gguf
    TEMPLATE """{{ if .System }}<|im_start|>system
    {{ .System }}<|im_end|>
    {{ end }}{{ if .Prompt }}<|im_start|>user
    {{ .Prompt }}<|im_end|>
    {{ end }}<|im_start|>assistant
    """
    PARAMETER stop "<|im_end|>"
    ```

2.  Create and run:
    ```bash
    ollama create testlabs -f Modelfile
    ollama run testlabs
    ```

## Development
This model is under active development. Expect regular updates as training strategies and datasets are refined.
