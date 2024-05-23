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

### Train MTSCI from scratch

```shell
bash ./scripts/ETT_point.sh
bash ./scripts/ETT_block.sh
bash ./scripts/Weather_point.sh
bash ./scripts/Weather_block.sh
bash ./scripts/METRLA_point.sh
bash ./scripts/METRLA_block.sh
```

### Test MTSCI from trained model

First, set the scratch is `False` in scripts.
Then, run these scripts.

```shell
bash ./scripts/ETT_point.sh
bash ./scripts/ETT_block.sh
bash ./scripts/Weather_point.sh
bash ./scripts/Weather_block.sh
bash ./scripts/METRLA_point.sh
bash ./scripts/METRLA_block.sh
```
