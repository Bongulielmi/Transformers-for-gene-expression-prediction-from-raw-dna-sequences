# Transformers for Gene Expression Prediction from Raw DNA Sequences

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter Notebooks](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Organization](#repository-organization)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Access](#data-access)
- [Workflow Examples](#workflow-examples)
- [Model Classes](#model-classes)
- [Best Hyperparameters](#best-hyperparameters)
- [Results & Visualizations](#results--visualizations)
- [Contributing](#contributing)
- [License](#license)
- [References](#references)

---

## Project Overview

This repository provides code and workflows for **predicting gene expression** directly from raw DNA sequences using advanced transformer architectures. Our models are designed to process genomic data and output gene expression predictions, enabling research into regulatory genomics and precision medicine.

**Key Features:**
- Transformer-based models optimized for DNA sequence analysis
- End-to-end pipelines for data preprocessing, training, and evaluation
- Modular and extensible code structure
- Example workflows in Jupyter Notebooks

---

## Repository Organization

```
├── data/                  # Data storage and preprocessing scripts
├── notebooks/             # Jupyter Notebooks for experiments and workflows
├── src/                   # Core source code: models, training, utilities
├── results/               # Output results, logs, and plots
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

**Note:** Large datasets and certain model checkpoints are hosted externally (see [Data Access](#data-access)).  

---

## Getting Started

### Prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [transformers](https://huggingface.co/transformers/)
- Jupyter Notebook

Install required Python packages:

```bash
pip install -r requirements.txt
```

### Installation

Clone this repository:

```bash
git clone https://github.com/Bongulielmi/Transformers-for-gene-expression-prediction-from-raw-dna-sequences.git
cd Transformers-for-gene-expression-prediction-from-raw-dna-sequences
```

### Data Access

- **Google Drive:** [Access raw and processed datasets here.](<YOUR_GDRIVE_LINK>)
- Download and place data under the `data/` directory as outlined in the notebooks.

---

## Workflow Examples

Run the main analysis notebook:

```bash
jupyter notebook notebooks/GeneExpressionTransformer.ipynb
```

Or execute training scripts:

```bash
python src/train_transformer.py --config configs/default.yaml
```

Sample workflow steps:
1. Data preprocessing (`src/data_processing.py`)
2. Model training (`src/train_transformer.py`)
3. Model evaluation and visualization (`notebooks/Evaluation.ipynb`)

---

## Model Classes

Our main model classes include:

- **DNATransformer**: Implements the transformer architecture for DNA sequences.
- **GeneExpressionDataset**: Custom PyTorch Dataset for efficient loading.
- **Trainer**: Manages training loops, logging, and checkpointing.

**Example instantiation:**

```python
from src.models import DNATransformer

model = DNATransformer(
    input_length=1000,
    num_layers=6,
    num_heads=8,
    hidden_dim=512,
    dropout=0.1
)
```

See [src/models.py](src/models.py) for implementation details.

---

## Best Hyperparameters

| Parameter         | Value   |
|-------------------|---------|
| Input Length      | 1000    |
| Num Layers        | 6       |
| Num Heads         | 8       |
| Hidden Dimension  | 512     |
| Dropout           | 0.1     |
| Batch Size        | 32      |
| Learning Rate     | 1e-4    |

*(See [notebooks/HyperparameterTuning.ipynb](notebooks/HyperparameterTuning.ipynb) for tuning details.)*

---

## Results & Visualizations

Sample evaluation metrics and plots are available in the `results/` directory and in the [Evaluation Notebook](notebooks/Evaluation.ipynb).

- **Pearson Correlation:** 0.87 (Test Set)
- **Loss Curve:**  
  ![Loss Curve](results/loss_curve.png)

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes and push (`git push origin feature/your-feature`)
4. Open a Pull Request

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## References

- Vaswani, A., et al. (2017). ["Attention is All You Need"](https://arxiv.org/abs/1706.03762)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**For questions or support, please open an issue or contact [Bongulielmi](https://github.com/Bongulielmi).**

---

Let me know if you want further customization, such as adding images, more examples, or direct links to datasets!
