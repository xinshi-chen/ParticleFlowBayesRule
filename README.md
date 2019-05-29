# Particle Flow Bayes' Rule

Implementation of [Particle Flow Bayes’ Rule](http://proceedings.mlr.press/v97/chen19c.html)

Stepwise posterior estimation for 'two-gaussian'.

From left to right: 
1. smc (Sequential Monte Carlo), 
2. pfbayes (our method), 
3. True posterior

![demo](video/two_gaussian.gif)


# Setup

## Install the package

This package requires the dependency of torch==1.0.0 and torchdiffeq[3].
Our implementation is based on [ffjord](https://github.com/rtqichen/ffjord) [2].

```
pip install torch==1.0.0
git clone https://github.com/rtqichen/torchdiffeq
cd torchdiffeq
pip install -e .
```

After that, clone and install the current package.

```
pip install -e .
```

## data and results

The data and pretrained model dumps can be obtained via the [shared dropbox folder](https://www.dropbox.com/sh/a6bbockuft7bilb/AAAbH_AE1DHkcBkuVNio5IFla?dl=0)

After downloading the shared folder, put it under the root of the project (or create a symbolic link) and rename it as 'dropbox', so that the default bash script can automatically find them.

Finally the project has the following folder structure:
```
pfbayes
|___pfbayes  # source code
|   |___common # common implementations
|   |___experiments # code for each experiment
|
|___dropbox  # data, trained model dumps and result statistics
    |___data  
    |___results 
        |____hmm_lds # for hmm lds
        |......
...
```


# Reproducing the experiments

In general, the scripts come with the experiment have the default configurations. 
You can also tune the hyperparameters like number of particles, solver types, etc. 

## Multivariate Gaussian Model

## data
The data used in the paper is included, and you can find them here:
```
cd pfbayes/experiments/mvn_unimodal/data
```
You can also generate new data using the script `run_create_test_data.sh`

### train/evaluate
First navigate to the experiment folder, and you can use the pretrained model directly:
```
cd pfbayes/experiments/mvn_unimodal
./run_main.sh
```
This will generate evaluation results under current `scratch/` folder.
To train from scratch, simply set `phase=seg_train` in above script. 

## Gaussian Mixture Model
The data is generated on the fly. This experiment is mainly used for qualitative evaluation. 

### train/evaluate
```
cd pfbayes/experiments/two_gaussian
./run_main.sh
```
You will get videos under current `scratch/` folder. You can also try different seeds to see how our pfbayes can estimate the posterior for new sequences. 
To train from scratch, simply set `phase=train` in above script. 


## HMM-LDS

### data
We've included the LDS model and sampled traces under `pfbayes/experiments/hmm_lds/data`. 

You can also create new data and samples using the scripts under that folder, e.g.
```
cd pfbayes/experiments/hmm_lds/data
python saved_lds.py  # this will create a new LDS model
./run_create_test_data.sh  # this will load the model and generate samples
```

### train/evaluate
First navigate to the experiment folder, and you can use
the pretrained model directly:

```
cd pfbayes/experiments/hmm_lds
./run_main.sh
```

This will generate results under current `scratch/` folder

To train from scratch, simply set `phase=train` in above script. 


## Bayesian Logistic Regression

### data
The dataset is preprocessed into numpy array, you can find more details in the paper.

### pfbayes as variational inference

This experiment simply performs variational inference for posterior of entire training set. 
We perform 10 rounds of training/test, with different random splits per each. 

```
cd pfbayes/experiments/logistic_regression
./run_vi.sh
```

### generalize learned bayes rules

Similar to meta-learning, here each 'task' corresponds to a fixed angle of data rotation.
The pretrained model can be generalized to different random angles. Try different random seeds to evaluate against different angles:
```
./eval_angle_meta.sh
```
You can also train from scratch with random angles:
```
cd pfbayes/experiments/logistic_regression
./train_angle_meta.sh
```



# References
[1] Xinshi Chen, Hanjun Dai, Le Song. "Particle Flow Bayes' Rule." *In International Conference on Machine Learning.* 2019.

[2] Grathwohl W, Chen RT, Betterncourt J, Sutskever I, Duvenaud D. "FFJORD: Free-form continuous dynamics for scalable reversible generative models." *arXiv preprint arXiv:1810.01367.* 2018.

[3] Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David Duvenaud. "Neural Ordinary Differential Equations." *Advances in Neural Processing Information Systems.* 2018.

