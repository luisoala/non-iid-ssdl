# Dataset Similarity to Assess Semi-supervised Learning Under Distribution Mismatch Between the Labelled and Unlabelled Datasets

| [Paper](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9078688) | [ICLR 2021 RobustML Poster](https://github.com/luisoala/luisoala.github.io/blob/master/assets/img/repos/noniidssdl/Poster_ICLR_2021_v2%20(1).png) |
| --- | --- |
| [IEEE Transactions on Artificial Intelligence](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9078688), [arXiv open access](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=9078688) | <img src="https://github.com/luisoala/luisoala.github.io/blob/master/assets/img/repos/noniidssdl/Poster_ICLR_2021_v2%20(1).png" alt="drawing" width="400"/> |


[**Paper**](https://arxiv.org/abs/2104.10223) | [**Poster**](https://github.com/luisoala/luisoala.github.io/blob/master/assets/pdf/posters/Poster_ICLR_2021_v2%20(1).pdf)

## A short introduction
![Visual abstract](https://github.com/luisoala/luisoala.github.io/blob/master/assets/img/repos/noniidssdl/Screenshot_20220421-124636_Chrome.jpg)
Semi-supervised deep learning (SSDL) is a popular strategy to leverage unlabelled data for machine learning when labelled data is not readily available. In real-world scenarios, different unlabelled data sources are usually available, with varying degrees of distribution mismatch regarding the labelled datasets.  It begs the question which unlabelled dataset to choose for good SSDL outcomes. Oftentimes, semantic heuristics are used to match unlabelled data with labelled data. However, a quantitative and systematic approach to this selection problem would be preferable. In this work, we first test the SSDL MixMatch algorithm under various distribution mismatch configurations to study the impact on SSDL accuracy. Then, we propose a quantitative unlabelled dataset selection heuristic based on dataset dissimilarity measures. These are designed to systematically assess how distribution mismatch between the labelled and unlabelled datasets affects MixMatch performance. We refer to our proposed method as deep dataset dissimilarity measures (DeDiMs), designed to compare labelled and unlabelled datasets. They use the feature space of a generic Wide-ResNet, can be applied prior to learning, are quick to evaluate and model agnostic. The strong correlation in our tests between MixMatch accuracy and the proposed DeDiMs suggests that this approach can be a good fit for quantitatively ranking different unlabelled datasets prior to SSDL training.

## Data access
If you wish to reproduce any of the experiments data sets are automatically downloaded by the experiment script `ood_experiment_at_scale_script.sh` for your convenience based on which experiment you choose to run. An overview of the different data sets can be found below. Note that we used the training split of each data set as the basis to construct our own training and test splits for each experimental run. The Gaussian and Salt and Pepper data sets were created with the following parameters: a variance of 10 and mean 0 for the Gaussian noise, and an equal Bernoulli probability for 0 and 255 pixels, in the case of the Salt and Pepper noise.

![Data](https://github.com/luisoala/luisoala.github.io/blob/master/assets/img/repos/noniidssdl/Screenshot%20from%202021-09-11%2015-32-00.png)
## Code
We provice the experiment script `ood_experiment_at_scale_script.sh` for your convenience where you can select the types of experiments you would like to run.
## Cite as  
    @ARTICLE{non-iid-ssdl,
      author    = {Sa{\'{u}}l Calder{\'{o}}n Ram{\'{\i}}rez and Luis Oala and Jordina Torrents{-}Barrena and Shengxiang Yang and Armaghan Moemeni and   
                    Wojciech Samek and Miguel A. Molina{-}Cabello}
      journal={IEEE Transactions on Artificial Intelligence}, 
      title={Dataset Similarity to Assess Semi-supervised Learning Under Distribution Mismatch Between the Labelled and Unlabelled Datasets},  
      year={2022},
      volume={tbd},
      number={tbd},
      pages={tbd},
      doi={tbd}}
**For any questions feel free to open an issue or contact us**
