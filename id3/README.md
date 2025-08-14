# Final Project - ID3

```sh

# CD into the ID3 sub-folder inside the project folder.

# Run Cross Validation to find the best depth for the tree.
python cross_validation.py -d 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 -i collision

# Run the actual training with the parameters found.
python train.py -m decision_tree -d 13 -i collision


```