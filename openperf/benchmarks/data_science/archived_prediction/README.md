# Open Source Archival Project Prediction

### Research Background
The sustainability of the open source software ecosystem has become a key topic. Unlike traditional software development lifecycle models with clear delivery targets, responsible teams, and milestones, open source software development in the early stages heavily relies on self-organizing contributors and voluntary work, which also leads to the archiving of open source software projects. In the world of open source software, numerous projects are archived for various reasons, meaning they are marked by developers as read-only, no longer accepting new issues, pull requests, or comments. For developers active on GitHub and others who follow specific open source projects, predicting whether a project is likely to be archived is very important.

### Task Description
How can one predict whether an open source project is likely to be archived? This task requires analysis of a large amount of data, including but not limited to the project's commit history, project activity, maintainer information, and project maturity, among others. This data must be used to identify key factors that may lead to the archiving of a project, and then a model must be developed based on these factors to predict whether an open source project will be archived.

### Task Challenges
Diversity of influencing factors: Factors influencing whether an open source project is archived include, but are not limited to, the project's lifecycle, activity level, developer engagement, number of contributors, code complexity, and project dependencies. These factors may interact with each other, making the prediction results more complex.
Time dynamics: The status of open source projects is constantly changing over time, so the prediction model needs to be able to process time series data and capture the dynamic changes in project status.

### Reference Code
https://doi.org/10.5281/zenodo.7230174

#### References
1. Xia X, Zhao S, Zhang X, et al. Understanding the Archived Projects on GitHub[C]//2023 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER). IEEE, 2023: 13-24.
