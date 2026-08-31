# 1.Описание задачи
Решаю задачу многоклассовой классификации с 10 классами. Данные fashion-MNIST картинки размером 28x28 пикселей с одним серым каналом, таргеты от 0 до 9, число отражает название одежды для соотвествующего класса. 

По результатам EDA - классы распределены равномерно.  

Используемые модели в эксперименте:  
SIMPLE MLP, MLP TUNNED,   
ALEXNET,   
RESNET, RESNET TUNNED,   
TRANSFORMER SWIN, TRANSFORMER SWIN FINETUNNED.    

# 2.Выводы эксперимента:  
1. Использования сверточных сетей для CV задачи и этого датасета дало прирост по точности 3.5% с 88.77% до 92.32%.
2. Каждое зменение гиперпараметров нейроситей давали хоть и небольшой но прирост. В одном случае с 91.31% до 92.32%, в другом 87.93% до 88.77%. Да улучшение незначитальные, но они были подобраны в ручную, думаю при помощи более продвинутых методов подробора можно было бы получить больший прирост.
3. Трансформер отработал хуже, чем более старый RESNET, даже файнтюнинг с инизиализацией весов не повысил качество. Говорит о том, что под конкретную задачу важна более детальная настройка модели и её архитектуры, а не просто взять самую новую модель. 
4. Применяемые модели не предназначены для размера картинки 28х28, из-за размеров яред, карт активаций, пуллингов в их свертках, поэтому не было сильного прироста качества.
5. В идеале под эту задачу нужно подобрать архитектуры свертки для 28х28. Либо как вариант масштабировать картинку до больших размеров подходящих для Alexnet и других моделей.
6. Даже базовая MLP справилась хорошо, значит данные достаточно чистые и даже слабенькая сеть может уловить закономерность.

| Model | Train Accuracy | Test Accuracy |
|:------|---------------:|--------------:|
| **ResNet Tuned** | **95.12%** | **92.32%** |
| AlexNet AS IS | 95.24% | 91.86% |
| ResNet AS IS | 95.73% | 91.31% |
| Swin Transformer | 95.91% | 90.96% |
| Swin Transformer Finetuned | 95.37% | 90.87% |
| Tuned MLP | 93.14% | 88.77% |
| Simple MLP | 91.05% | 87.93% |
# 3. DEPENDENCIES
## Installation

### Requirements

* Python 3.12
* [uv](https://docs.astral.sh/uv/)

### 1. Clone the repository

```bash
git clone https://github.com/ivanfalkov/fashion-MNIST.git
cd fashion-MNIST
```

### 2. Install Python 3.12

If Python 3.12 is not installed:

```bash
uv python install 3.12
```

Set Python 3.12 as the project Python version:

```bash
uv python pin 3.12
```

This creates a `.python-version` file in the project.

### 3. Create a virtual environment

```bash
uv venv
```

This creates a `.venv` directory with an isolated Python environment for the project.

### 4. Install dependencies

Install all project dependencies from `pyproject.toml` and `uv.lock`:

```bash
uv sync
```

### 5. Jupyter Notebook

The project uses `ipykernel` to run Jupyter notebooks directly in Cursor.

Open a `.ipynb` file in Cursor and select the project's `.venv` as the Python kernel.

To verify the selected environment:

```python
import sys

print(sys.executable)
```

The path should point to:

```text
.../fashion-MNIST/.venv/...
```
### 6. Code Formatting and Linting

Format the project with Ruff:

uv run ruff format .

Check the code for linting issues:

uv run ruff check .

Automatically fix supported linting issues:

uv run ruff check . --fix

### 7. Project Structure

```text
fashion-MNIST/
├── configs/              # YAML configuration files
├── notebooks/            # Jupyter notebooks
├── scripts/              # Project scripts
├── src/
│   └── data/             # Dataset files   
│   └── metrics/          # Metrics file
│   └── checkpoints/      # Models 
├── .gitignore
├── pyproject.toml        # Project configuration and dependencies
├── uv.lock               # Locked dependency versions
└── README.md
```

### 8. Dependencies

The main project dependencies include:

* `torch` — deep learning framework
* `torchvision` — computer vision datasets and transformations
* `matplotlib` — data visualization
* `seaborn` — statistical data visualization
* `pyyaml` — parsing and working with YAML configuration files
* `click` — creating command-line interfaces (CLI) and handling command-line arguments

Development dependencies include:

* `ipykernel` — Jupyter Notebook kernel for working with `.ipynb` files in Cursor
* `ruff` — linting and code formatting
* `torchinfo` — displaying model architecture and parameter information

All dependencies are specified in `pyproject.toml` and locked in `uv.lock`.
