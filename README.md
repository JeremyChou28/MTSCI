<!--
 * @Description:
 * @Author: Jianping Zhou
 * @Email: jianpingzhou0927@gmail.com
 * @Date: 2024-05-22 14:53:04
-->

# MTSCI: Multivariate Time Series Consistent Imputation

It's a pytorch implementation of paper "MTSCI: A Conditional Diffusion Model for Consistent Imputation in Incomplete Time Series" under review in CIKM2024.

## Requirements

```shell
pip install -r requirements.txt
```

## Datasets

- ETT
- Weather
- METR-LA (you should unzip the METRLA.tar.gz `tar -zxvf METRLA.tar.gz`)

## How to run

```shell
bash ./scripts/ETT_point.sh
bash ./scripts/ETT_block.sh
bash ./scripts/Weather_point.sh
bash ./scripts/Weather_block.sh
bash ./scripts/METRLA_point.sh
bash ./scripts/METRLA_block.sh
```
