# SFT Dataset Sampler

This project is used to sample from existing SFT datasets, supporting the separation of **easy / hard / thinking** subsets.  
It generates sampled data files as well as statistics and plots of the most common bash tools.

## Environment Setup

Use the default Python environment (Python 3.9+ recommended).

Install dependencies:
```bash
pip install -r requirements.txt
```

## Directory Structure

Example:

```
project_root/
├── data/                        # Original datasets
│   ├── bashtrain/               
│   │   ├── direct.jsonl
│   │   └── tool.jsonl
│   ├── rlvr-bash-edited/
│   │   ├── direct.jsonl
│   │   └── tool.jsonl
│   ├── ...                      
│   ├── shellm-V3-simple-unix_thinking_0818_7452s_single_turn.jsonl # original data
│   └── shellm-V3-simple-unix_thinking_0818_7452s_single_turn_openhands.jsonl
│
├── sample_dataset/              # Generated sampled datasets
│   └── ...                      
│
├── sample_dataset.py            # Main script
├── requirements.txt
└── README.md
```

## Usage

Run the sampling script:

```bash
python sample_dataset.py --input_dir <dataset_dir> --orig_size <num> --new_size <num> --easy_ratio <float> --thinking_ratio <float>
```

Arguments:

* `--input_dir` : Root dataset directory (contains dataset folders and original jsonl files)  
* `--orig_size` : Number of samples to draw from original files  
* `--new_size` : Number of samples to draw from new dataset folders  
* `--easy_ratio` : Ratio of easy data in new samples (0–1)  
* `--thinking_ratio` : Ratio of thinking data within hard samples (0–1)  

Output:

* Sampled dataset: `sample_dataset/orig{orig_size}_new{new_size}_easy{easy_ratio}_think{thinking_ratio}.jsonl`  
* Bash tool distribution plot: same name with `.png` extension  

## Examples

Balanced easy/hard, 50% thinking in hard:
```bash
python sample_dataset.py --input_dir data --orig_size 0 --new_size 10000 --easy_ratio 0.5 --thinking_ratio 0.5
```

All easy:
```bash
python sample_dataset.py --input_dir data --orig_size 0 --new_size 10000 --easy_ratio 1.0 --thinking_ratio 0.0
```

All hard with all thinking:
```bash
python sample_dataset.py --input_dir data --orig_size 0 --new_size 10000 --easy_ratio 0.0 --thinking_ratio 1.0
```

20% easy, 80% hard, among hard 70% with thinking:
```bash
python sample_dataset.py --input_dir data --orig_size 0 --new_size 10000 --easy_ratio 0.2 --thinking_ratio 0.7
```

---

## Dataset Sampling Script – Parameters Overview

Ruochen Deng 84389121 2025-08-28 19:45  

* `--input_dir` : Path to input dataset directory  
* `--orig_size` : Number of samples from original dataset  
* `--new_size` : Number of samples from new datasets  
* `--easy_ratio` : Fraction (0–1) of *easy* samples in new data  
  * Applies only to new data  
  * *Easy* contains no thinking  
* `--thinking_ratio` : Fraction (0–1) of *thinking* samples inside the *hard* portion of new data  
  * Effective proportion = `(1 – easy_ratio) × thinking_ratio`  
* `--direct_ratio` : Ratio of *direct* samples (raw command). Default `0.5`  
  * *Tool* samples (openhand) ratio = `1 – direct_ratio`  

**Final dataset size** = `orig_size + new_size`  

**Output from script:**  
1. A single dataset with exactly `orig_size + new_size` samples  
2. A plot of the **Top 30 most frequent commands**
