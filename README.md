# Meta Particle Flow

Pytorch implementation for:

[Meta Particle Flow for Sequential Bayesian Inference](https://arxiv.org/abs/1902.00640) [1]

Note: Our implementation is based on [ffjord](https://github.com/rtqichen/ffjord) [2].

# Install

  `pip install -e .`

  This package has the dependency over [torchdiffeq](https://github.com/rtqichen/torchdiffeq) [3], please install it first.


# References
[1] Xinshi Chen, Hanjun Dai, Le Song. "Meta particle flow for sequential bayesian inference." *In International Conference on Machine Learning.* 2019.

[2] Grathwohl W, Chen RT, Betterncourt J, Sutskever I, Duvenaud D. "FFJORD: Free-form continuous dynamics for scalable reversible generative models." *arXiv preprint arXiv:1810.01367.* 2018.

[3] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. "Neural Ordinary Differential Equations." *Advances in Neural Processing Information Systems.* 2018.

# Citation
```
@article{chen2019meta,
  title={Meta Particle Flow for Sequential Bayesian Inference},
  author={Chen, Xinshi and Dai, Hanjun and Song, Le},
  journal={arXiv preprint arXiv:1902.00640},
  year={2019}
}
```
