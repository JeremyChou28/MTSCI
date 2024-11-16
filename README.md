<!--
 * @Description:
 * @Author: Jianping Zhou
 * @Email: jianpingzhou0927@gmail.com
 * @Date: 2024-08-07 11:26:38
-->

# MTSCI: Multivariate Time Series Consistent Imputation

It's a pytorch implementation of paper "[MTSCI: A Conditional Diffusion Model for Consistent Imputation in Incomplete Time Series](https://dl.acm.org/doi/10.1145/3627673.3679532)" accepted in CIKM2024.

## Requirements

```shell
pip install -r requirements.txt
```

## Datasets

- ETT
- Weather
- METR-LA (you should unzip the METRLA.tar.gz `tar -zxvf METRLA.tar.gz`)

## How to run

### Train MTSCI from scratch

```shell
cd scripts/
bash ETT_point.sh
bash ETT_block.sh
bash Weather_point.sh
bash Weather_block.sh
bash METRLA_point.sh
bash METRLA_block.sh
```

### Test MTSCI from trained model

First, set the scratch is `False` in scripts.
Then, run these scripts.

```shell
bash ETT_point.sh
bash ETT_block.sh
bash Weather_point.sh
bash Weather_block.sh
bash METRLA_point.sh
bash METRLA_block.sh
```

## Citation

If you find this repo useful, please cite our paper. Thanks for your attention.

```
@inproceedings{zhou2024mtsci,
  title={MTSCI: A Conditional Diffusion Model for Multivariate Time Series Consistent Imputation},
  author={Zhou, Jianping and Li, Junhao and Zheng, Guanjie and Wang, Xinbing and Zhou, Chenghu},
  booktitle={Proceedings of the 33rd ACM International Conference on Information and Knowledge Management},
  pages={3474--3483},
  year={2024}
}
```
