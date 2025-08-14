# Final Project - Perceptron

```sh

# CD into the Perceptron sub-folder inside the project folder.

# Run Cross Validation to find better parameters to run.
python cross_validation.py -m margin --lr_values 0.1 0.01 0.001 1 --mu_values 0.1 0.3 0.5 0.7 1 5 10 --epochs 15

# Run Epoch Training to find the optimal number of epochs after finding the optimal parameters.
python epochs.py -m margin --lr 0.1 --mu 10 --epochs 50

# Run the actual training with the parameters found.
python train.py -m margin --lr 0.1 --mu 10 --epochs 9


```

```sh

# Run this for Margin Perceptron with non-trivial pre-processing.
python cross_validation.py -m margin --lr_values 0.1 0.01 0.001 1 --mu_values 0.1 0.3 0.5 0.7 1 5 10 --epochs 15

python epochs.py -m margin --lr 0.01 --mu 0.1 --epochs 50

python train.py -m margin --lr 0.01 --mu 0.1 --epochs 4

```