<!-- mathjax include -->
<!-- {% include mathjax.html %} -->
<!-- defining some tex commands that can be used throughout the page-->
# Abstract
In this work, we propose MixMOOD - a systematic approach to mitigate the effects of class distribution mismatch in semi-supervised deep learning (SSDL) with MixMatch. This work is divided into two components: (i) an extensive out of distribution (OOD) ablation test bed for SSDL and (ii) a quantitative unlabelled dataset selection heuristic referred to as MixMOOD. In the first part, we analyze the sensitivity of MixMatch accuracy under 90 different distribution mismatch scenarios across three multi-class classification tasks. These are designed to systematically understand how OOD unlabelled data affects MixMatch performance. In the second part, we propose an efficient and effective method, called deep dataset dissimilarity measures (DeDiMs), to compare labelled and unlabelled datasets. The proposed DeDiMs are quick to evaluate and model agnostic. They use the feature space of a generic Wide-ResNet and can be applied prior to learning. Our test results reveal that supposed semantic similarity between labelled and unlabelled data is not a good heuristic for unlabelled data selection. In contrast, strong correlation between MixMatch accuracy and the proposed DeDiMs allow us to quantitatively rank different unlabelled datasets ante hoc according to expected MixMatch accuracy. This is what we call MixMOOD. Furthermore, we argue that the MixMOOD approach can aid to standardize the evaluation of different semi-supervised learning techniques under real world scenarios involving out of distribution data.
# Highlights and Findings
We present the MixMatch approach to out of distribution data (MixMOOD). It entails the following contributions:
* A systematic OOD ablation test bed. We demonstrate that including OOD data in the unlabelled training dataset for the MixMatch algorithm can yield different degrees of accuracy degradation compared to the exclusive use of \gls{IOD} data. However, in most cases, using unlabelled data with \gls{OOD} contamination still improves the results when compared to the default fully supervised configuration.
* Markedly, OOD data that is supposedly semantically similar to the OOD labelled data does not always lead to the highest accuracy gain.
* We propose and evaluate four deep dataset dissimilarity measures (DeDiMs) that can be used to rank unlabelled data according to the expected accuracy gain _prior_ to SSDL training. They are cheap to compute and model-agnostic which make them amenable for practical application.
* Our test results reveal a strong correlation between the tested DeDiMs and MixMatch accuracy, making them informative for unlabelled dataset selection. This leads to MixMOOD which proposes the usage of tested \gls{DeDiM}s to select the unlabelled dataset for improved MixMatch accuracy.

$$ \begin{table}[t]
%\caption{\textbf{Results OOD Experiments.} IOD Data is the base data set that is considered IOD. OOD Type is the OOD configuration which can be \textit{half-half} (the model only learns to predict on half of the classes present in the data set while the data from the OH of classes is used as OOD data) or \textit{different} (data sets from a different data set than the IOD data are used as OOD data). OOD Data signifies the data set that was used as OOD data. OOD \% is the percentage of OOD data in the unlabeled data. $N$ Labelled is the number of labelled data points from IOD Data that were used during semi-supervised training. BL is the performance of a fully supervised baseline model trained on the entire data set of labelled IOD Data.}
\caption{Results for the class distribution mismatch experiment. Each result entry in the table represents the mean and variance of accuracy across ten random experimental runs per entry. For a detailed description of symbols and the experiment see \Cref{sec:testbed}.}
\label{tab:half-half-results}
%\begin{center}
%\begin{tiny}
\tiny
\centering
%l@{\qquad}c@{\;\;}c@{\qquad}c@{\;\;}c
\begin{tabular}{ccccccc}
\toprule
\multirow{2}{*}{$\mathbf{S_{IOD}}$} & \multirow{2}{*}{$\mathbf{T_{OOD}}$} & \multirow{2}{*}{$\mathbf{S_{uOOD}}$} & \multirow{2}{*}{$\mathbf{\%_{uOOD}}$} & \multicolumn{3}{c}{$\mathbf{n_l}$}\\

&  & &  & 60 & 100 & 150\\

\midrule

\multirow{12}{*}{\rotatebox[origin=c]{90}{\textbf{MNIST}}} 
& \multicolumn{3}{c}{Fully supervised baseline} & $0.457\pm0.108$ & $0.559\pm 0.125 $ & $0.645 \pm0.101$ \\

 & \multicolumn{3}{c}{SSDL baseline (no OOD data)} & $\mathbf{0.704 \pm 0.096}$ & $\mathbf{0.781\pm 0.065}$ & $\mathbf{0.831 \pm0.0626}$\\
 
 
 \cline{2-7}
 
 & \multirow{2}{*}{HH} & \multirow{2}{*}{OH} & 50 & $0.679\pm0.108$ & $0.769\pm0.066$ & $0.802\pm0.054$\\
 & & & 100 & $0.642 \pm 0.111$ & $0.746 \pm0.094$ & $0.798 \pm0.07$\\
 
 
 \cline{2-7}
 & \multirow{2}{*}{Sim} & \multirow{2}{*}{SVHN} & 50 & $0.637 \pm0.098$ & $0.745 \pm0.081$ & $0.801 \pm0.0699$\\
 
 & & & 100 & $0.482 \pm0.113$ & $0.719 \pm0.058$ & $0.765 \pm0.072$\\
 
  \cline{2-7}
 
  & \multirow{6}{*}{Dif}& \multirow{2}{*}{TI} & 50 & $0.642 \pm0.094$ & $0.739 \pm 0.074$ & $0.809 \pm0.066$\\
  
  & & & 100 & $0.637 \pm0.097$ & $0.732\pm 0.074$ & $0.804 \pm0.071$\\
  
  
 
 \cline{3-7}
 
  & & \multirow{2}{*}{G} & 50 & $0.606 \pm0.0989$ & $0.713 \pm0.087$ & $0.786\pm 0.065$\\
  
  & & & 100 & $0.442 \pm0.099$ & $0.461 \pm0.073$ & $0.542\pm 0.062$\\
  
 
  
   \cline{3-7}
 
  & & \multirow{2}{*}{SAP} & 50 & $0.631 \pm0.102$ & $0.735 \pm0.082$ & $0.813 \pm0.057$\\
  
  & & & 100 & $0.48\pm0.0951$ & $0.524 \pm0.09$ & $0.563 \pm0.095$\\
 
\hline

\multirow{12}{*}{\rotatebox[origin=c]{90}{\textbf{CIFAR-10}}} & \multicolumn{3}{c}{Fully supervised baseline} & $0.380\pm0.024$ & $0.445\pm0.042$ & $0.449\pm0.022$\\

 &  \multicolumn{3}{c}{SSDL baseline (no OOD data)} & $\mathbf{0.453\pm0.046}$ & $0.474\pm0.019$ & $0.501\pm0.033$\\
 
 
 \cline{2-7}


& \multirow{2}{*}{HH} & \multirow{2}{*}{OH} & 50 & $0.444\pm0.040$ & $0.472\pm0.039$ & $0.525\pm0.050$\\ 
 & & & 100 & $0.443\pm0.023$ & $0.472\pm0.047$ & $0.499\pm0.054$ \\
 \cline{2-7}
 & \multirow{2}{*}{Sim} & \multirow{2}{*}{TI} & 50 & $0.435\pm0.054$ & $0.473\pm0.039$ & $\mathbf{0.543\pm0.040}$\\ 
 & & & 100 & $0.417\pm0.020$ & $\mathbf{0.480\pm0.039}$ & $0.498\pm0.042$ \\
 
 \cline{2-7}
 
  & \multirow{6}{*}{Dif} & \multirow{2}{*}{SVHN} & 50 & $0.419\pm0.027$ & $0.464\pm0.044$ & $0.469\pm0.056$\\ 
 & & & 100 & $0.385\pm0.034$ & $0.418\pm0.035$ & $0.440\pm0.046$ \\
 
 \cline{3-7}
 
  & & \multirow{2}{*}{G} & 50 & $0.409\pm0.047$ & $0.454\pm0.048$ & $0.491\pm0.032$\\ 
 & & & 100 & $0.297\pm0.029$ & $0.306\pm0.034$ & $0.302\pm0.038$ \\
 
 \cline{3-7}
 
  & & \multirow{2}{*}{SAP}& 50 & $0.438\pm0.029$ & $0.455\pm0.037$ & $0.485\pm0.034$\\ 
 & & & 100 & $0.236\pm0.031$ & $0.246\pm0.032$ & $0.232\pm0.022$ \\
  
  \hline

\multirow{12}{*}{\rotatebox[origin=c]{90}{\textbf{FashionMNIST}}} & \multicolumn{3}{c}{Fully supervised baseline} & $0.571\pm0.073$ & $0.704\pm0.066$ & $0.720\pm0.093$\\

 & \multicolumn{3}{c}{SSDL baseline (no OOD data)} & $\mathbf{0.715\pm0.049}$ & $\mathbf{0.760\pm0.044}$ & $0.756\pm0.069$\\
 \cline{2-7}

& \multirow{2}{*}{HH} & \multirow{2}{*}{OH} & 50 & $0.714\pm0.049$ & $0.721\pm0.104$ & $0.765\pm0.053$\\ 
 & & & 100 & $0.660\pm0.061$ & $0.711\pm0.090$ & $0.747\pm0.061$ \\
 \cline{2-7}
 & \multirow{2}{*}{Sim} & \multirow{2}{*}{FP} & 50 & $0.707\pm0.039$ & $0.724\pm0.030$ & $0.778\pm0.078$\\ 
 & & & 100 & $0.546\pm0.101$ & $0.542\pm0.099$ & $0.540\pm0.105$ \\
 
 \cline{2-7}
 
  & \multirow{6}{*}{Dif}& \multirow{2}{*}{TI} & 50 & $0.690\pm0.065$ & $0.745\pm0.093$ & $0.792\pm0.058$\\ 
 & & & 100 & $0.690\pm0.073$ & $0.728\pm0.066$ & $\mathbf{0.794\pm0.056}$ \\
  \cline{3-7}
  & & \multirow{2}{*}{G} & 50 & $0.644\pm0.061$ & $0.689\pm0.075$ & $0.755\pm0.055$\\ 
 & & & 100 & $0.352\pm0.025$ & $0.366\pm0.065$ & $0.361\pm0.057$ \\
  \cline{3-7}
  & & \multirow{2}{*}{SAP} & 50 & $0.671\pm0.072$ & $0.708\pm0.095$ & $0.729\pm0.088$\\ 
 & & & 100 & $0.276\pm0.069$ & $0.297\pm0.046$ & $0.283\pm0.059$\\
 \bottomrule
\end{tabular}
%\end{tiny}
%\end{center}
\end{table}
$$
## Recommendations and Closing Thoughts

# Questions?
If you have questions regarding the code or the paper or if would like to discuss ideas please open an issue in the [repo](https://github.com/luisoala/mixmood).


