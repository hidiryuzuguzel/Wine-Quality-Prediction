# Wine-Quality-Prediction
Cortez et al. (2009) studied the relation of various chemical properties, such as acidity, on perceived quality of Porteguese vinho verde wines as evaluated by a panel of experts. 

## Data: 
The datasets are  *(../code/white_wine.mat)* and  *(../code/red_wine.mat)*. Each dataset contains a 1-by-12 cell variable **headers** which indicates the meaning of the columns of N-by-12 data matrix **data**, where *N* is the number data points. 

|Variable number  | Variable info  | 
|---|---|
| 1  | fixed acidity |  
| 2  | volatile acidity |
| 3  | citric acid |  
| 4  | residual sugar  | 
| 5  | chlorides | 
| 6  | free sulfur dioxide | 
| 7  | total sulfur dioxide |  
| 8  | density |
| 9  | pH |  
| 10  | sulphates  | 
| 11 | alcohol | 
| 12 | quality (score between 0 and 10)  | 

Further information may be found at https://archive.ics.uci.edu/ml/datasets/wine+quality

## Model:

Multi layer perceptron is used. As a comparison, linear regression is used. 
