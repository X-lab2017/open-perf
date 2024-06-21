# Open Source Behavioral Data Completion and Prediction

### Research Background
In the field of open source software, high-quality data governance has become a crucial factor in driving the development of open source initiatives. Especially in today's era of accelerated digital processes, data has transformed into a vital resource. High-quality data helps to accurately grasp the current state of a project, while poor or missing data can lead to errors in research conclusions, thereby affecting decision-making for open source projects. When collecting developer behavior data from open source platforms like GitHub, incomplete data collection due to platform internal mechanisms, API limitations, collection techniques, and fluctuations in related services can result in missing behavior data for some projects, significantly hindering subsequent granular research.

### Task Description
How can a model be designed that not only completes missing values in open source behavior data but also predicts future trends? This model needs to effectively mine the cyclicality and correlations within open source behavior data, preserve timestamp information, and not rely on prior knowledge or probability distributions of the data. Additionally, the model must effectively handle situations of data missing due to reasons such as GitHub's collection limitations and fluctuations in network services.

### Task Challenges
Uncertainty of missing data: Missing behavioral data can be caused by many factors, such as platform API limitations and fluctuations in network services. This uncertainty makes it difficult to apply a unified method to handle all missing data, requiring a flexible approach to address different scenarios.
Handling time-series data: Open source behavior data is a type of time-series data that requires preserving its timestamp information when dealing with missing data, maintaining the temporality of the data. Handling time-series data, especially with missing values, is more challenging than dealing with static data.
Unified handling of missing data completion and trend prediction: The model needs to address data missing issues and perform trend prediction. It must consider various characteristics of the data, such as cyclicality, correlations, and temporality.

### Dataset
To validate the generality and predictive accuracy of the baseline models, this task used a dataset of the top 10 most active open source projects in 2020 from OpenLeaderboard, including PyTorch, SkyWalking, TensorFlow, TiDB, VSCode, Flutter, Kibana, Kubernetes, Nixpkgs, and Rust. Some projects in this dataset contain missing behavioral data, providing a comprehensive test scenario for evaluating the baseline models.

### Evaluation Metrics
To validate the effectiveness of various models in predicting missing and future values, NMAE, NRMSE, and NMSEA were selected as evaluation standards. These metrics collectively assess the model's predictive performance by focusing on the differences and relative differences between the predicted and actual values.

### Model Experiments

Using OSS behavioral data with missing values from the dataset as a test set, the following comparison algorithms were selected: TRMF, Regularized MF (RMF), MF, Non-negative MF (NMF), Probabilistic MF (PMF), Basic-SVD, and BSMF. To obtain more accurate comparative effects, the following experimental scheme was designed: setting the iteration count of the seven algorithms to a fixed value of 1000 (adjustable), then calculating the prediction error of all other normal values excluding missing values. The table below and the accompanying figure illustrate the NMSE and NMAE values of each algorithm group across five datasets.

| Dataset      | Metric | TAMF  | TRMF  | L-SVR | L-R   | KNN   | iForest | K-fold | R-chain |
|--------------|--------|-------|-------|-------|-------|-------|---------|--------|---------|
| Pytorch      | NMSE   | 0.041 | 0.987 | 1.236 | 3.326 | 2.28  | 0.161   | 5.457  | 1.388   |
| Pytorch      | NRMSE  | 0.203 | 0.993 | 1.112 | 1.823 | 1.51  | 0.401   | 2.336  | 1.178   |
| Pytorch      | NMAE   | 0.282 | 1.132 | 1.371 | 2.133 | 1.80  | 0.518   | 2.660  | 1.356   |
| Skywalking   | NMSE   | 0.235 | 0.264 | 0.886 | 0.202 | 0.21  | 0.143   | 0.353  | 0.875   |
| Skywalking   | NRMSE  | 0.484 | 0.514 | 0.941 | 0.449 | 0.46  | 0.379   | 0.594  | 0.935   |
| Skywalking   | NMAE   | 0.471 | 0.671 | 1.059 | 0.482 | 0.49  | 0.448   | 0.636  | 1.042   |
| Tensorflow   | NMSE   | 0.024 | 0.031 | 0.051 | 0.182 | 0.04  | 0.049   | 0.094  | 0.262   |
| Tensorflow   | NRMSE  | 0.157 | 0.176 | 0.225 | 0.426 | 0.22  | 0.223   | 0.307  | 0.512   |
| Tensorflow   | NMAE   | 0.210 | 0.250 | 0.253 | 0.429 | 0.23  | 0.250   | 0.318  | 0.526   |
| Tidb         | NMSE   | 0.043 | 0.124 | 6.462 | 0.23  | 0.18  | 0.546   | 0.272  | 2.858   |
| Tidb         | NRMSE  | 0.208 | 0.352 | 2.542 | 0.479 | 0.43  | 0.739   | 0.521  | 1.691   |
| Tidb         | NMAE   | 0.227 | 0.435 | 2.261 | 0.449 | 0.37  | 0.640   | 0.586  | 1.682   |
| Vscode       | NMSE   | 0.234 | 0.326 | 0.482 | 1.528 | 0.61  | 1.753   | 2.709  | 0.516   |
| Vscode       | NRMSE  | 0.484 | 0.571 | 0.694 | 1.236 | 0.78  | 1.324   | 1.646  | 0.718   |
| Vscode       | NMAE   | 0.435 | 0.535 | 0.559 | 1.259 | 0.75  | 1.207   | 1.516  | 0.677   |

![Alt text](result.png)

#### References
1. Chen L, Yang Y, Wang W. Temporal Autoregressive Matrix Factorization for High-Dimensional Time Series Prediction of OSS[J]. IEEE Transactions on Neural Networks and Learning Systems, 2023.
