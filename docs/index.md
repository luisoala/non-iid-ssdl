<!-- mathjax include -->
{% include mathjax.html %}

<!-- some tex commands -->
$$ 
  \newcommand{\bfx}{\mathbf{x}}
  \newcommand{\bfy}{\mathbf{y}}
  \newcommand{\bfv}{\mathbf{v}}
  \newcommand{\bfg}{\mathbf{g}}
  \newcommand{\self}{\text{SSRL}}
  \newcommand{\super}{\text{SL}}
$$
# Preliminaries
We follow the notation by [Tian et al.](https://arxiv.org/abs/2005.10243) in describing the learning situation. It features three main steps: (i) producing augmented unlabelled data, (ii) learning representations from augmented unlabelled data, (iii) learning a supervised task using the represenations. 

## (i) Producing augmented unlabelled data
In the first step, the
* $$\bfx^{(i)}_u \in \mathbb{R}^{d}$$ unlabelled data and a set
* $$ \mathcal{G} = \{\bfg_1, ..., \bfg_k\} $$ of transormation functions
are used to create
* $$ \mathcal{V}^{(i)} = \{\bfv^{(i)}_1, ..., \bfv^{(i)}_K\} $$, a set of views for each $$\bfx^{(i)}_u$$ given by $$ \bfv^{(i)}_j = \bfg_j(\bfx^{(i)}_u)$$

## (ii) Learning representations from augmented unlabelled data
In the second step,
* $$\self(\mathcal{V})$$, a black-box self-supervised representation learning algorithm, is used to learn
* $$\mathcal{Z}^{i}$$, a set of representations for each $$\bfx^{(i)}_u$$

## (iii) Learning a supervised task using the represenations
In the third and final step, 
* $$\super(\mathcal{Z}, \mathcal{X}, \mathcal{Y})$$, a black-box supervised learning algorithm, is used to learn
* $$\mathbf{f}(\bfx)$$, a mapping between $$\bfx$$, e.g. images, and $$\bfy$$, e.g. classes, for
* $$\mathfrak{T}$$, some downstream task, e.g. classification of histopathological images
Note that the $$\bfx^{(i)}$$ may differ from $$\bfx^{(i)}_u$$ in some way <span title="Different how? Is $$p(x) \neq p(x_u)$$? Are they from different domains?"><sup>NOTE</sup></span>

# Questions
1. Given only $$\mathcal{(X, V)} $$ and a performance measure $$\mathbf{P}_{\mathfrak{T}}$$ for task $$\mathfrak{T}$$, can we make ante hoc predictions which view $$\bfv^{(i)}_j$$ is best to learn representations for $$\mathfrak{T}$$?
1. Can we use models of perceptual image quality in $$\self$$ to learn better representations $$\mathcal{Z}$$ for downstream task $$\mathfrak{T}$$?

# Ideas
## Regarding 1.
### Feature space distances
* [ ] Get samples for all possible views $$\mathcal{V}$$ created in [Chen et al.](https://arxiv.org/abs/2002.05709)
* [ ] Use [MixMOOD](https://github.com/peglegpete/mixmood) to calculate distances between $$\mathbf{h}(\mathcal{X}), \mathbf{h}(\mathcal{V}_j)$$
* [ ] Use $$I(\cdot, \cdot)$$ from [Tian et al.](https://arxiv.org/abs/2005.10243) to calculate distances between $$\mathbf{h}(\mathcal{X}), \mathbf{h}(\mathcal{V}_j)$$
* [ ] Calculate correlations between two distance sets and $$\mathbf{P}_{\mathfrak{T}}$$ from [Chen et al.](https://arxiv.org/abs/2002.05709)

## Regarding 2.
TODO

# Todos
[Task Board](https://github.com/users/peglegpete/projects/1)

# Resources
* [MI](https://en.wikipedia.org/wiki/Mutual_information)

# Results
TODO

