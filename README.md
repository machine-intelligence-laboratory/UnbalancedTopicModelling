# UnbalancedTopicModelling

During training topic models on the unbalanced collections, one can face the fact, that topic capacities tend to become similar/have small dispersity, and small topics are included into the big one, while big topics are splitted into approximately equal parts.
This library proposes different methods for training unbalanced topic models and for estimating models quality in terms of different statistical functions.

## Topic Prior Regularizer

One of the possible imbalance solution is to construct a priori probabilities of words in topics proportionally to the topics capacities in the collection. 
Regularizer formula is the following:

<img src="https://render.githubusercontent.com/render/math?math=R_{TopicPrior}(\Phi,\Theta) = \sum_t\sum_w\beta_t\log\phi_{wt}">

with partial derivative

<img src="https://render.githubusercontent.com/render/math?math=\frac{\partial R}{\partial\Phi_{wt}} = \frac{\beta_t}{\varphi_{wt}}">

## Semantic Heterogeneity Regularizer

Second possible approach considers minimization of topic semantic heterogeneity S<sub>t</sub> for all topics as an additional regularization part in the model loss. 
Definition of topic semantic heterogeneity is presented below:

<img src="https://render.githubusercontent.com/render/math?math=S_t=\sum_{d\in D}\sum_{w\in d}\frac{n_{tdw}}{n_t}\ln{\frac{\hat{p}(w|d)}{p(w|d)}}=\avg{d,w}{n_{tdw}}{\ln{\frac{\hat{p}(w|d)}{p(w|d)}}}">

## Requirements

For the modelling text collection as a vowpal wabbit file is required (example of the data can be found in [data.vw](./data/lenta_1000_100.vw)):
```
228 |text новый альбом <person> выложить сеть сервис яндекс ... |ngramms новый_альбом_<person> официальный_релиз певица_<person>_<person> ...
...
```

## Structure

    .
    └── library                                       # Core
        ├── regularization.py                         # TopicPriorRegularizer and SemanticHeterogenityRegularizer
        ├── statistics.py                             # Statistic tests for topic modelling and SemanticHeterogenityRegularizer
        └── top_tokens.py                             # Top tokens visualizer for topic model
    └── notebooks                                     # Application examples
        ├── regularized_balance.ipynb                 # Notebook with application of regularizers from regularization.py
        └── conditional_independence_statistics.ipynb # Additional notebook with calculation of conditional independence statistics
    └── data                                          # Examples of training data
