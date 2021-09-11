# More Than Meets The Eye: Semi-supervised Learning Under Non-IID Data
[**Paper**](https://arxiv.org/abs/2104.10223) | [**Poster**](https://github.com/luisoala/luisoala.github.io/blob/master/assets/pdf/posters/Poster_ICLR_2021_v2%20(1).pdf)

## A short introduction
A common heuristic in semi-supervised deep learning (SSDL) is to select unlabelled data based on a notion of semantic similarity to the labelled data. For example, labelled images of numbers should be paired with unlabelled images of numbers instead of, say, unlabelled images of cars. We refer to this practice as semantic data set matching. In this work, we demonstrate the limits of semantic data set matching. We show that it can sometimes even degrade the performance for a state of the art SSDL algorithm. We present and make available a comprehensive simulation sandbox, called non-IID-SSDL, for stress testing an SSDL algorithm under different degrees of distribution mismatch between the labelled and unlabelled data sets. In addition, we demonstrate that simple density based dissimilarity measures in the feature space of a generic classifier offer a promising and more reliable quantitative matching criterion to select unlabelled data before SSDL training.

![Poster](https://github.com/luisoala/luisoala.github.io/blob/master/assets/img/repos/noniidssdl/Poster_ICLR_2021_v2%20(1).png)
## Data access
If you wish to reproduce any of the experiments data sets are automatically downloaded by the experiment script `ood_experiment_at_scale_script.sh` for your convenience based on which experiment you choose to run. An overview of the different data sets can be found below. Note that we used the training split of each data set as the basis to construct our own training and test splits for each experimental run. The Gaussian and Salt and Pepper data sets were created with the following parameters: a variance of 10 and mean 0 for the Gaussian noise, and an equal Bernoulli probability for 0 and 255 pixels, in the case of the Salt and Pepper noise.

![Data](https://github.com/luisoala/luisoala.github.io/blob/master/assets/img/repos/noniidssdl/Screenshot%20from%202021-09-11%2015-32-00.png)
## Code
We provice the experiment script `ood_experiment_at_scale_script.sh` for your convenience where you can select the types of experiments you would like to run.
## Cite as

    @misc{calderonramirez2021meets,
      title={More Than Meets The Eye: Semi-supervised Learning Under Non-IID Data}, 
      author={Saul Calderon-Ramirez and Luis Oala},
      year={2021},
      eprint={2104.10223},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
    }
**For any questions feel free to open an issue or contact us**
