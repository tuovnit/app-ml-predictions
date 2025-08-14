# Final Project - SVM & Logistic Regression

```sh

# CD into the svm_logreg sub-folder inside the project folder.

# Run Cross Validation to find better parameters to run for SVM.
python cross_validation.py -m svm --lr0_values 1 0.1 0.01 0.001 0.0001 --reg_tradeoff_values 10 1 0.1 0.01 0.001 --epochs 5

# Run the actual training with the parameters found for SVM.
python train.py -m svm --lr0 0.0001 --reg_tradeoff 10.0 --epochs 20

```

```sh

# Run Cross Validation to find better parameters to run for Logistic Regression.
python cross_validation.py -m logistic_regression --lr0_values 1 0.1 0.01 0.001 0.0001 --reg_tradeoff_values 1 10 100 1000 --epochs 5

# Run the actual training with the parameters found for Logistic Regression.
python train.py -m logistic_regression --lr0 0.01 --reg_tradeoff 1000.0 --epochs 20

```