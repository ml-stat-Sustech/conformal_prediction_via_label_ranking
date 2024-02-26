# Conformal Prediction for Deep Classifier via Label Ranking
This repository is the offical  implementation  for the paper: [Conformal Prediction for Deep Classifier via Label Ranking](https://arxiv.org/abs/2310.06430).

## Usage
We use Python 3.9, and other packages can be installed by:
```
pip install -r requirements.txt
```

Producing the prediction sets:
```
python main.py --dataset  imagenet  --trials 10
```
with the following arguments:
 - dataset: the name of dataset.
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

