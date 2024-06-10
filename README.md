# Conformal Prediction for Deep Classifier via Label Ranking

This repository is the official implementation
of [Conformal Prediction for Deep Classifier via Label Ranking](https://arxiv.org/abs/2310.06430) at ICML'2024

## How to Install

This code is built on the awesome toolbox [TorchCP](https://github.com/ml-stat-Sustech/torchCP) that you need to install
first. We use Python 3.9, and TorchCP can be installed by:

```
pip install torchcp
```

Simply follow the instructions described here to install TorchCP as well as PyTorch. After that, other packages can be
installed by:

```
pip install -r requirements.txt
```

Then, you are ready to go.

## How to Run

Producing the prediction sets:

```
python main.py --dataset imagenet  --trials 10
```

with the following arguments:

- dataset: the name of the dataset.
- trials: the number of trials.

## Citation

If you find this useful in your research, please consider citing:

    @misc{huang2023conformal,
      title={Conformal Prediction for Deep Classifier via Label Ranking}, 
      author={Jianguo Huang and Huajun Xi and Linjun Zhang and Huaxiu Yao and Yue Qiu and Hongxin Wei},
      year={2023},
      eprint={2310.06430},
      archivePrefix={arXiv},
      primaryClass={cs.LG}}

