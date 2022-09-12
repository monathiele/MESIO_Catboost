# MESIO_catboost
## MESIO Summer School 2021 competition by Jordi Moragas

https://mesioupcub.masters.upc.edu/en/xv-summer-school-2022/courses/applied-machine-learning-to-solve-real-life-problems

This contest was be used to evaluate the performance of the models developed by the students of the course "Applied Machine Learning to solve Real Life Problems".

A rating (classification) model should be constructed on the variable "TARGET" that takes values 0 (not default) or 1 (default, which is bankruptcy status after 1 year). Each column represents a characteristic (ratio) of the company and can contain missing values. There are both numerical and categorical variables.

The output will be a default probability (it does not need to be calibrated, only the sorting matters).

## Provided Data

Train and test files and an example of submission with random default probabilities ("Pred" column).

To train the model, use the train file: The variable to predict ("TARGET" column) is the 12-month default from the observation point of the data. 0 is healthy, 1 is default. Each line represents a company. The predictors have not been previously chosen or treated and there can be missing values.

To make predictions, take the test data, which no longer has the "TARGET" column, but the names of the predictors are exactly the same.

Finally, for each "ID" of the submission there is a corresponding "ID" in the test set. You must put the default probability in the "Pred" column (a real number between 0 and 1). It is not necessary the calibration of the probability, only the sorting matters

## Some Output Examples:

Plotting SHAP values for Feature Importance:

![image](https://user-images.githubusercontent.com/97023507/189594814-03fa3f38-3ebd-4e77-91c2-225902c48ce7.png)
![image](https://user-images.githubusercontent.com/97023507/189595229-b124a01a-2961-4cda-b8c1-f45f712e6115.png)

