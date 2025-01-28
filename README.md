# Poker Hand Classification Project

## Overview

This project tackles the challenge of classifying poker hands using machine learning models. The dataset used is sourced from the UCI Machine Learning Repository and consists of 1,025,010 samples. Each sample represents a hand of poker and includes information about the hand’s five card values and their suits. The objective is to predict the correct poker hand class, such as "One Pair," "Straight," or "Royal Flush."

The project’s goal was to explore various machine learning algorithms and determine which model could most effectively predict poker hands based on the dataset. We explored different approaches, including K-Nearest Neighbors (KNN), Linear Regression, Random Forest, and Neural Networks, aiming to understand which method could best handle the intricacies of poker hand classification.

The work was a collaborative effort among our team members: Hunter Becker, Kat Chu, Oumar Diakite, and Mysee Lee. Each member contributed to data preprocessing, model evaluation, and the overall presentation of results. The final solution presented a comprehensive comparison of different models, highlighting the Neural Network as the most effective approach for achieving high accuracy in predicting poker hands.

## Dataset Information

The dataset used in this project is the **Poker Hand Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/poker+hand). It contains 1,025,010 samples of poker hands with 10 classes, each representing a different type of poker hand. The features consist of:

- **5 card values**: C1, C2, C3, C4, C5
- **5 card suits**: S1, S2, S3, S4, S5

Each row in the dataset represents a poker hand with five cards, where each card is described by two features: its value (ranging from 2 to 14) and its suit (represented by integers 1 to 4). The target variable is the poker hand class, which includes 10 possible categories, such as "One Pair," "Straight," "Flush," and so on.

The class distribution in the dataset is balanced, with each class containing a roughly equal number of instances, ensuring that no particular class is underrepresented.

## Group Members

- **Hunter Becker** – Data Preprocessing, Model Evaluation, Performance Tuning, README
- **Kat Chu** – Data Preprocessing, Model Evaluation, Presentation, README
- **Oumar Diakite** – Model Implementation
- **Mysee Lee** – Model Implementation

## Objective

The primary objective of this project is to explore various machine learning algorithms for classifying poker hands. Our aim is to:

- Evaluate and compare multiple models to determine which performs best in predicting poker hand types.
- Understand how different algorithms handle categorical classification problems with high-dimensional data.
- Optimize the models to achieve the highest possible accuracy in predicting poker hands.

We will experiment with traditional machine learning models like K-Nearest Neighbors and Random Forest, as well as deep learning models like Neural Networks. Each model's performance will be assessed using classification accuracy and other relevant metrics.

## Key Technologies and Libraries

In this project, we leveraged several key technologies and libraries:

- **Python**: Programming language used to implement models and data analysis.
- **Scikit-learn**: For model building, evaluation, and preprocessing (e.g., K-Nearest Neighbors, Linear Regression, Random Forest).
- **TensorFlow/Keras**: For implementing the Neural Network model.
- **Pandas**: For data manipulation and preprocessing.
- **NumPy**: For numerical operations.
- **Matplotlib/Seaborn**: For visualizing data and model performance.

## Model Evaluation

In this project, we experimented with several machine learning models to classify poker hands. The goal was to predict the correct poker hand category, such as "One Pair," "Straight," or "Royal Flush." Below is a detailed overview of the models we tested and how each performed:

### 1. **K-Nearest Neighbors (KNN)**
   - **Steps**:
     - Split data into training and test sets.
     - Applied MinMaxScaler to normalize the features.
     - Trained the KNN classifier using `KNeighborsClassifier(n_neighbors=9)`.
   - **Performance**:
     - **Accuracy**: 50%
     - While KNN was able to classify some hands correctly, it struggled to predict certain categories due to the high dimensionality of the data and the curse of dimensionality.
     - Despite experimenting with different values of `k` for neighbors, the accuracy remained around 50%.
   - **Challenges**:
     - KNN doesn't handle complex relationships well and performs slower as the dataset grows larger.

   ![KNN Performance](https://github.com/user-attachments/assets/d8b94fb2-7893-407f-ac36-1d1f24c22cf4)

### 2. **Linear Regression**
   - **Steps**:
     - Encoded the target variable.
     - Split data into training and test sets.
     - Trained a linear regression model using `LinearRegression` from scikit-learn.
   - **Performance**:
     - **R-squared**: 0.00
     - Linear regression models are not suited for classification problems like poker hand classification because the target variable is categorical. The low R-squared value (0) indicated that the model didn’t capture any of the variance in the target.
   - **Challenges**:
     - Linear regression assumes a linear relationship between input and output, which doesn't apply to poker hand classification.

### 3. **Random Forest**
   - **Steps**:
     - Randomly sampled data and trained a Random Forest classifier.
     - The Random Forest model was optimized by adjusting hyperparameters.
   - **Performance**:
     - **Accuracy**: 92%
     - Random Forest significantly outperformed KNN and Linear Regression, achieving 92% accuracy. It did a much better job of handling the complexity and variability in the dataset.
   - **Challenges**:
     - While Random Forest performed well, it could still be improved in terms of fine-tuning hyperparameters and optimizing for better generalization.

   ![Random Forest Performance](https://github.com/user-attachments/assets/random-forest-placeholder)

### 4. **Neural Network**
   - **Steps**:
     - Encoded the target variable using `LabelEncoder()`.
     - One-hot encoded the target variable.
     - Split data into training and test sets.
     - Defined the neural network model with two hidden layers and softmax for multiclass classification:
       ```python
       model = Sequential([
           Dense(64, input_dim=X_train.shape[1], activation='relu'),  # Input layer
           Dense(32, activation='relu'),  # Hidden layer
           Dense(y_onehot.shape[1], activation='softmax')  # Output layer
       ])
       ```
     - Trained the model using `adam` optimizer and `categorical_crossentropy` loss.
   - **Performance**:
     - **Accuracy**: 100% for most hands
     - The Neural Network model provided the best results by far, achieving 100% accuracy for most poker hand categories.
     - The neural network was able to detect intricate patterns in the data and generalize well across the different categories.
   - **Challenges**:
     - Some categories, like "Royal Flush" and "Flush," still had minor prediction issues, though the accuracy for these categories was much higher than any of the other models tested.

   ![Neural Network Performance](https://github.com/user-attachments/assets/89943c55-99a8-40bd-bdfc-e11add1a43c4)

### Conclusion
- **Best Model**: The Neural Network was the standout model, achieving 100% accuracy for most poker hands.
- **Runner-up**: Random Forest showed strong performance, with 92% accuracy, but still had room for improvement in certain areas.
- **Limitations of Other Models**: Both KNN and Linear Regression were not suitable for this classification problem. KNN's performance was limited by the high dimensionality of the data, and Linear Regression failed to capture any meaningful variance in the target variable.

By experimenting with these models and analyzing their performance, we were able to identify the strengths and weaknesses of each approach. The Neural Network model, while still needing fine-tuning for specific categories, gave the best overall performance for this task.

