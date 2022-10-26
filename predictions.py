#Line below allows you to use RandomForestRegressor for predictions.
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#The X and y lines below read that data you will use to train your machine learning algorithm.  Your paths will be different though.  For example, I doubt yours will have “genwal” in it.  You can find the path on Kaggle.  You must make this change on your own.
X = pd.read_csv('Xdata.csv')
y = pd.read_csv('../input/d/genwal/Data/Ydata.csv')
#The bottom two lines drop the column of names from the dataset.
Xdrop = X.drop('Names',1)
ydrop = y.drop('Names',1)
#The next two lines read the input data you will use for predictions, and drop the Names column from that dataset.
PredictX = pd.read_csv('../input/d/genwal/Data/PredictX.csv')
PredictXdrop = PredictX.drop('Names',1)
#The next two lines train your machine learning algorithm on the inputs where you know the resulting outputs.
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(Xdrop,ydrop)
#The next three lines use your trained machine learning algorithm to make the predictions you want and to convert them to a downloadable .csv file (spreadsheet file).
test_preds = rf_model_on_full_data.predict(PredictXdrop)
output = pd.DataFrame({'Names': PredictX.Names, 'Predictions': test_preds})
output.to_csv('Predictions.csv', index=False)
