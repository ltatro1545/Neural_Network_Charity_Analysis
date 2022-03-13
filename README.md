# Neural_Network_Charity_Analysis

## Background
Alphabet Soup, a philanthropic foundation, has raised and donated roughly $10B over the past 20 years to various organizations to promote welfare, unity, and environmental sanctity. Unfortunately, they have disovered that multiple organizations have been unsuccessful in their endeavors, thus rendering the funding fruitless. Alphabet Soup determined, based on their data of 34,000 groups that have received funding, that a neural network model could be used to predict whether a company would be successful if provided funding.

## Baseline Method
#### Pre-Processing
After loading in the provided dataset ("Resources/charity_data.csv") to a Jupyter Notebook, exploratory analysis was performed. Initially, columns such as "EIN" (a form of ID number) and the "NAME" column were dropped, as they did not appear to be relevant features in predicting organization success. Various code was used to examine the number of unique values in each column and then to plot the density of those values. This was an important step in determining if features should be binned or not, and, if so, to what degree. After binning "CLASSIFICATION" and "APPLICATION_TYPE", sklearn's OneHotEncoder was used to encode the categorical data. 

#### Neural Network Model
For the baseline neural network model, a sequential model with two hidden layers were used. The first hidden layer multiplied the number of input features by three to determine how many neurons would be present; the second layer multiplied the inputs by 1.75x, then rounded if needed. Both hidden layers and the output layers utilized a sigmoid activation, as we are predicting binary results with it. In total, there were 16,259 parameters within the neural network.

The model was then compiled as such:
  - loss: "binary_crossentropy"
  - optimizer: "adam"
  - metrics: ["accuracy"]
 
 A callback was created to save checkpoints during the model training phase using Tensorflow's ModelCheckpoint(), which can be found in the folder labeled "checkpoints". The optimized model weights are labeled as such: "opt_weights.05.hdf5", whereas the bsae model do not have "opt" in the name. The save_freq parameter was not functioning properly and convoluted the results, so the deprecated parameter "period" was used instead. The model was then trained with 150 epochs. The historic loss and accuracy was plotted afterward to determine if the metrics were still improving, and possibly requiring more epochs; 150 was sufficient.
 
 The model was then saved to the "Trained_Models" folder:
  - Pre-Optimized Model: "Trained_Models/AlphabetSoupCharity_D2"
  - Optimized Model: "Trained_Models/AlphabetSoupCharity_D3
 
      ^The models were both re-imported to test their functionality, with success.

#### Summary of the steps taken for the baseline:
 - Exploratory analysis conducted
 - Dropped irrelevant columns
 - Plotted density of values to determine binning parameters
 - Encoded categorical features then dropped original columns
 - Split the data
 - Normalized the split data
 - Created sequential neural network model
 - Determined adequate model parameters
 - Fit and evaluated the model
 
 ### Baseline Results
