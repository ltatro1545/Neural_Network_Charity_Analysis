# Neural_Network_Charity_Analysis

## Background
Alphabet Soup, a philanthropic foundation, has raised and donated roughly $10B over the past 20 years to various organizations to promote welfare, unity, and environmental sanctity. Unfortunately, they have disovered that multiple organizations have been unsuccessful in their endeavors, thus rendering the funding fruitless. Alphabet Soup determined, based on their data of 34,000 groups that have received funding, that a neural network model could be used to predict whether a company would be successful if provided funding.

## Method
#### Pre-Processing
After loading in the provided dataset ("Resources/charity_data.csv") to a Jupyter Notebook, exploratory analysis was performed. Initially, columns such as "EIN" (a form of ID number) and the "NAME" column were dropped, as they did not appear to be relevant features in predicting organization success. Various code was used to examine the number of unique values in each column and then to plot the density of those values. This was an important step in determining if features should be binned or not, and, if so, to what degree. After binning "CLASSIFICATION" and "APPLICATION_TYPE", sklearn's OneHotEncoder was used to encode the categorical data. The encoded dataframe was merged with the original, and then the original categorical columns were dropped. Next, the data was split into features, target, training, and testing batches. The target feature, "IS_SUCCESSFUL", was denoted as variable "y". The data was stratified upon splitting to ensure the balance of the training and testing groups remained in tact.

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
 
 ----------------------------
 
## Results
### Baseline Results
With 44 features, 16,000 trainable parameters across two hidden layers, and 150 epochs to iterate, the results are as follows:

![Base_NN_Eval](https://user-images.githubusercontent.com/92493572/158045738-1c3fb070-6910-4928-8ef9-810089485bda.PNG)

**Test Results**
-- Loss: 56.82% --
Accuracy: 72.56%

#### Historic Loss Across Epochs
![Base_Loss](https://user-images.githubusercontent.com/92493572/158045907-c74ed12a-4f87-4bcd-8b3e-03be77c39cf5.PNG)

#### Historic Accuracy Across Epochs
![Base_Acc](https://user-images.githubusercontent.com/92493572/158045914-94752ade-e900-4d0f-9243-360301f098a3.PNG)

------------------------------

### Optimized Results
Utilized 394 features, 1,100,000 trainable parameters, three hidden layers, and also 150 epochs, the results are as follows:

![Opt_NN_Eval](https://user-images.githubusercontent.com/92493572/158046052-9ff4c0fe-9558-4620-80b3-ba5996c8dec3.PNG)

**Test Results**
-- Loss: 47.82% --
Accuracy: 78.62%

#### Historic Loss Across Epochs
![Opt_Loss](https://user-images.githubusercontent.com/92493572/158046058-f062c77f-c7fe-4c06-b106-c0e37c25e117.PNG)

#### Historic Accuracy Across Epochs
![Opt_Acc](https://user-images.githubusercontent.com/92493572/158046066-10a71de7-2156-4c5b-bc7b-62eacd5ad297.PNG)

### Changes During Optimization
  - "NAME" column was reconsidered, as many organizations have received funding multiple times, thus have a record of success. Groups with five instances of funding or less were binned into an "Other" group.
  - Columns "STATUS" and "SPECIAL_CONSIDERATIONS" were dropped due to having vastly imbalanced binary information, providing little information.
  - Third hidden layer added with around 525 neurons (all three layers were of sigmoid activation type)
  - Bin for application type was increased from under 100 value counts, to under 700 value counts.
  - Bin for classification type was increased from under 1,250 value counts, to under 1,500 value counts.

-------------------------------

## Secondary Random Forest Analysis
Sklearn was again used to create a random forest model to test another supervised machine learning method against the data. The random forest used the optimized data, which meant the encoded and binned "NAME" column was included in calculation. The main benefit of using the random forest is that the user can extract feature importances from the model; it is also significantly faster to execute than the neural network. The number of estimators used was 128.

### Results:

![RF_cm_class](https://user-images.githubusercontent.com/92493572/158046595-cd3a901a-3b63-4230-b392-d06bfc9b93ab.PNG)

**Test Results**
-- Accuracy: 77.2% --

### Top 10 Feature Importances

![RF_Feature_Imp](https://user-images.githubusercontent.com/92493572/158046696-a5a4955c-34ef-4530-b3f4-a19d4167711d.PNG)

### Summary of Random Forest Result
The random forest model performed about the same as the neural network with a lot less code and time involved in the process. Additionally, it provided feature importances - surprisingly, the ask amount was the weighted much more heavily than any other feature. The next step would be to investigate that more thoroughly. Was it lower or higher ask amounts where the organization failed, or had taken the money and fled? Was there any correlation at all?

--------------------------------

## Summary
