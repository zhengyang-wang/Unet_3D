# Data Preprocessing

1. Cut unnecessary zero edge. For simplicity, let the cutted data still have the same size for each subject. You need to set how many zero margins to keep.

2. Calculate means for mean-substraction based on cutted data.

3. Covert 0/10/150/250 labels into 0/1/2/3.

3. Do data augmentations: flip on 3 dims and rotation on 3 planes & 3 angles. In total, get 3+9=12 extra datasets.

# How to use

1. Properly set arguments and run
```
python calculate_mean.py
```

2. Copy the results to generate_h5.py

3. Properly set arguments and run
```
python generate_h5.py
```
