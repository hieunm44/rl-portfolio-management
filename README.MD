# Actor-Critic Reinforcement Learning Models for Portfolio Management

<div align="center">
<a href="https://www.math.unipd.it/en/">
   <img src="https://www.math.unipd.it/~tommasi/dueloghi.png" height=100"/>
</a>
</div>

## Overview
This repo is my thesis "Actor-Critic Reinforcement Learning Models for Portfolio Management" of the [Master's in Data Science](https://apply.unipd.it/courses/course/33-data-science) at Università degli Studi di Padova (UniPd). This is also the thesis of the [Erasmus Mundus Joint Master Degree Programme in Big Data Management and Analytics (BDMA)](https://bdma.ulb.ac.be/).

## Built With
<div align="center">
    <a href="https://finance.yahoo.com/">
        <img src="https://upload.wikimedia.org/wikipedia/commons/8/8f/Yahoo%21_Finance_logo_2021.png" height=40 hspace=10/>
    </a>
    <a href="https://vnstocks.com/">
        <img src="https://vnstocks.com/img/vnstock_logo_trans_rec_hoz.png" height=40 hspace=10/>
    </a>
    <a href="https://gymnasium.farama.org/">
        <img src="https://gymnasium.farama.org/_static/img/gymnasium_black.svg" height=40 hspace=10/>
    </a>
    <a href="https://stable-baselines3.readthedocs.io/">
        <img src="https://stable-baselines3.readthedocs.io/en/master/_static/logo.png" height=40/>
    </a>
</div>


## Usage
1. Clone the repo
   ```sh
   https://github.com/hieunm44/rl-portfolio-management.git
   cd rl-portfolio-management
   ```
2. Install necessary packages
   ```sh
   pip install -r requirements.txt
   ```
3. Check and run the notebook `Preprocess.ipynb` to collect and preprocess stock data. This will create `.csv` data files in folder `data`.
4. Check and run the notebook `TrainTest.ipynb` to train and test RL models. This will create `.zip` model files in folder `trained_models` and `.csv` result files in folder `results`.
5. Check and run the notebook `Plotting.ipynb` to create visualization figures. This will create `.pdf` figure files in folder `figures`.

The thesis file is published on [this site](https://thesis.unipd.it/handle/20.500.12608/90414).
