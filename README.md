# Project 4 - Poker Hand Machine Learning
### Hunter Becker, Kat Chu, Oumar Diakite, Mysee Lee

# Project Goal
For this project we will be training a model to classify what hand is present in a set of five playing cards. The goal is to have a model with at least 75% accuarcy.

---
# Dataset
The dataset can be found here:
```
https://archive.ics.uci.edu/dataset/158/poker+hand
```
The dataset contains 11 columns representing a five card hand and the classification of that hand. 
* ['S1','S2','S3','S4','S5'], Ordinal (1-4) representing {Hearts, Spades, Diamonds, Clubs}.
* ['C1','C2','C3','C4','C5'], Numerical (1-13) representing (Ace, 2, 3, ... , Queen, King). 
* ['CLASS'] tells us what Poker hand we have. Ordinal (0-9)
  ```
     0: Nothing in hand; not a recognized poker hand 
     1: One pair; one pair of equal ranks within five cards
     2: Two pairs; two pairs of equal ranks within five cards
     3: Three of a kind; three equal ranks within five cards
     4: Straight; five cards, sequentially ranked with no gaps
     5: Flush; five cards with the same suit
     6: Full house; pair + different rank three of a kind
     7: Four of a kind; four equal ranks within five cards
     8: Straight flush; straight + flush
     9: Royal flush; {Ace, King, Queen, Jack, Ten} + flush
   ```
The ['CLASS'] column will be the target variable and all remaining columns will be features. 

The data is already clean.

---
# Import data into Postgres and retrieve back to Python
The dataset provides an option to import the data from a library. 

* Install ucimlrepo using terminal or bash
```
  pip install ucimlrepo
```
* Fetch the dataset using the following code
```
     poker_hand = fetch_ucirepo(id=158)
     X = poker_hand.data.features
     y = poker_hand.data.targets
```
* Concat the X and y dataframes and export as a CSV ('poker_data.csv')
* Create a database in Postgres called 'poker'
* Using the tool upload the 'poker_table_schema.sql' script found in the Resources folder create the poker_data table
* Import the 'poker_data.csv' found in the Resources folder
* Edit the config.py file to allow access to the SQL database
* Query all records from the 'poker' database into a pandas dataframe 

---
# Find a model to optimize
Before training any models split the data into X feature variables and y target variable. 

## Random Forest
* Split X and y into training and testing variables. 
* One hot encode the feature variables
* Fit the sklearn model RandomForestClassifier(n_estimators=100, random_state=42)
* Use .ravel on the y_train when fitting the model
* Use the X and y test variables to make predictions with the model
* View accuarcy and efficiency using sklearns classification_report

  ## Results 
  * The ClassificationReport has an accuarcy of 92% but only predicted 2 of the 10 hands well
  * RandomForest can easily overfit, especially with complex datasets like poker hands, leading to lower accuracy

  ![Random Forest](https://github.com/user-attachments/assets/6c6af811-ce7d-4f01-ae29-88a7b94f3530)

---
## Linear Regression
* Encode the Target Variable
* Split X and y into training and testing variables. 
* Fit the sklearn model LinearRegression
* Use y_pred to make predicitons with the trained model
* View accuarcy and efficiency using mean_squared_error and r2_score

  ## Results 

    * Mean Squared Error (MSE): 1.47  
      This suggests that, on average, the squared difference between the predicted and actual values is 1.47. While this is not extremely high, it doesn't provide meaningful insights because the problem involves categorical classes rather than continuous numerical values.
    
    * R² Score: 0.00  
      An R² score of `0.00` means the model does not explain any of the variance in the target variable. Essentially, the linear regression model performs no better than predicting the mean value of the target.
    
    * Linear regression is not suitable for this problem because the target variable represents categories (e.g., poker hand types). Linear regression is designed for continuous numerical targets, not categorical ones.

  ![Linear Regression](https://github.com/user-attachments/assets/ee09cf25-b4f2-4f83-a678-0ed415c259c8)

---
## K-Nearest Neighbor
* Split X and y into training and testing variables. 
* Apply MinMaxScaler() to the X training and test variables.
* Use .ravel on the y training and test variables
* fit the sklearn model KNeighborsClassifier(n_neighbors = 9)
* Use y_pred to make predicitons with the trained model
* View accuarcy and efficiency using sklearns classification_report

  ## Results 
  * The ClassificationReport has an accuarcy of 100% yet four of the hands are at or very near 0% so this accuarcy number is skewed.
  * There arent enough instances of the four hands predcited poorly for the model learn them
  * K-Nearest Neighbor struggles with large datasets, slower with a high number of features (curse of dimensionality), and doesn't handle complex patterns as well as other models.
 
  ![K-Nearest Neighbor](https://github.com/user-attachments/assets/146279b5-225c-4f89-b6c0-7d87a07dc5bb)

---
## Neural Network
* Encode the y varibale with LabelEncoder()
* One hot encode the y variable
* Split X and y into training and testing variables.
* Define the number of layers, neurons, and activation types
```
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(y_onehot.shape[1], activation='softmax')  # Output layer with softmax for multiclass classification
])
```
* Compile the model
```
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```
* Train the model
```
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2, verbose=1)
```
* Use y_pred to make predicitons with the trained model
* View accuarcy and efficiency using sklearns classification_report

  ## Results 
  * The ClassificationReport has and accuarcy of 100%
  * The model was able to predict all but two hands very well giving the best results compared to the other models

  ![Neural Network 1](https://github.com/user-attachments/assets/b353aebf-5f13-4bc1-9f21-21a6c818d205)

---
# Optimizing our model

## First Attempt
* Adding a third hidden layer with a ReLU activation
```
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(16, activation='relu'), # Hidden layer
    Dense(y_onehot.shape[1], activation='softmax')  # Output layer with softmax for multiclass classification
])
```

  ## Results 
  * The ClassificationReport still has an accuarchy of 100% but we see a drop in accuarcy for some hands

  ![Neural Network 2](https://github.com/user-attachments/assets/f1bff0f7-9d0a-4fe1-b471-fd61570c4c8f)

## Second Attempt
* Return to only two layers and changing activation to Tanh 
```
model = Sequential([
    Dense(64, input_dim=X_train.shape[1], activation='tanh'),  # Input layer
    Dense(32, activation='tanh'),  # Hidden layer
    Dense(y_onehot.shape[1], activation='softmax')  # Output layer with softmax for multiclass classification
])
```

  ## Results 
  * The ClassificationReport still has an accuarchy of 100% but once again we see a drop in accuarcy for some hands

  ![Neural Network 3](https://github.com/user-attachments/assets/315b10bb-59ba-4e9c-9230-63607467357b)

---
# Conclusion 

Our neural network model with two relu layers one of 64 neurons and the second of 32 neurons with a softmax output layer provided the optimal results for our goal. With an accuarcy of 100% this model does well at predicting most hands yet has some room for improvement when predicting a Flush or Royal Flush.
---
# References
 * Chatgpt and prior assignments were referenced for creating models
