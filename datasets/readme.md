<!--
 * @Description: 
 * @Author: Jianping Zhou
 * @Email: jianpingzhou0927@gmail.com
 * @Date: 2024-11-16 19:09:11
-->
# Dataset 

| Datasets              | ETT    | Weather | METR-LA |
| --------------------- | ------ | ------- | ------- |
| Time span             | 69,680 | 52,696  | 34,272  |
| Interval              | 15 min | 10 min  | 5 min   |
| Features              | 7      | 21      | 207     |
| Sequence length       | 24     | 24      | 24      |
| Original missing rate | 0%     | 0.017%  | 8.6%    |



## ETT
ETT is a series of datasets, including ETTh1, ETTm1, ETTh2, and ETTm2, where "1" and "2" represent different transformers. "h" and "m" represent different sampling frequency (every 15 minute and every hour). In this paper, we select ETTm1.

## Weather
Weather is recorded every 10 minutes for 2020 whole year, which contains 21 meteorological indicators, such as air temperature, humidity, etc. Kindly note that there are multiple versions of Weather dataset in different papers, such as AutoFormer and Informer. We choose the version of AutoFormer.

## METR-LA

The traffic speed dataset contains average vehicle speeds recorded by 207 detectors located on Los Angeles County highways. The data spans from March 1, 2012, to June 27, 2012, with a sampling rate of 5 minutes. This dataset is widely used for traffic flow prediction tasks due to its detailed and frequent measurements, providing a comprehensive view of traffic patterns in the Los Angeles area.

