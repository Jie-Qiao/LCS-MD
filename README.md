# Identification of Causal Structure in the Presence of Missing Data with Additive Noise Model

This repository contains implementation of the paper: "Identification of Causal Structure in the Presence of Missing Data with Additive Noise Model" for handling both SMAR (Self-Masking Missing at Random) and SMNAR (Self-Masking Missing Not at Random) mechanisms.


## Requirements

- Required Python packages:
  - pandas
  - numpy
  - tqdm
  - gcastle==1.0.4rc1
  - scipy
  - xgboost
  - networkx 
  - pydot


## Installation

Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

The main experiment can be run by executing the `main.py` Python script:
```shell
python main.py
```

Results are saved in CSV files in the `result/orient` and `result/skeleton` directory, separated by whether skeleton learning or full DAG learning was performed.


## Citation

If you use this code in your research, please cite the following paper:

```bibtex
@inproceedings{qiao2024identification,
  title={Identification of causal structure in the presence of missing data with additive noise model},
  author={Qiao, Jie and Chen, Zhengming and Yu, Jianhua and Cai, Ruichu and Hao, Zhifeng},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={38},
  number={18},
  pages={20516--20523},
  year={2024}
}
```
