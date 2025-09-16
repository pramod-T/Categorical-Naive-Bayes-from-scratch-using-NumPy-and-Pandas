import pandas as pd
import numpy as np
import math


def split_dataset(features, labels, test_size=0.2, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - features (pandas.DataFrame): The feature matrix.
    - labels (pandas.Series): The labels corresponding to each row in the feature matrix.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int): Seed for random number generation.

    Returns:
    - Tuple containing the following:
        - X_train (pandas.DataFrame): Training data.
        - X_test (pandas.DataFrame): Testing data.
        - y_train (pandas.Series): Labels for training data.
        - y_test (pandas.Series): Labels for testing data.
    """
    np.random.seed(random_state)
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    split_idx = int(len(features) * (1 - test_size))
    train_indices, test_indices = indices[:split_idx], indices[split_idx:]

    X_train, X_test = features.iloc[train_indices], features.iloc[test_indices]
    y_train, y_test = labels.iloc[train_indices], labels.iloc[test_indices]

    return X_train, X_test, y_train, y_test

def evaluate_classifier(model, X_test, y_test):
    '''
    Evaluates the performance of the Naive Bayes classifier on the test dataset.

    Parameters:
    - model: The trained Naive Bayes classifier model.
    - X_test (pandas.DataFrame): Testing data.
    - y_test (pandas.Series): True labels for the testing data.

    Returns:
    - Tuple containing the following:
        - accuracy (float): Accuracy of the model.
        - precision (float): Precision of the model.
        - recall (float): Recall of the model.
        - f1 (float): F1-score of the model.
    '''
    """ for _, sample in X_test.iterrows():
        print(sample) """
    y_pred = [model(sample) for _, sample in X_test.iterrows()]
    #print(y_pred)
    correct_predictions = np.sum(np.array(y_pred) == np.array(y_test))
    total_samples = len(y_test)

    accuracy = correct_predictions / total_samples

    # Calculate precision, recall, and f1 for each class
    unique_classes = np.unique(y_test)
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for cls in unique_classes:
        true_positive = np.sum((np.array(y_pred) == cls) & (np.array(y_test) == cls))
        false_positive = np.sum((np.array(y_pred) == cls) & (np.array(y_test) != cls))
        false_negative = np.sum((np.array(y_pred) != cls) & (np.array(y_test) == cls))

        #print(true_positive)
        #print(false_positive)

        precision = true_positive / (true_positive + false_positive) 
        recall = true_positive / (true_positive + false_negative) 
        f1 = 2 * (precision * recall) / (precision + recall) 

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)

    precision = np.mean(precision_per_class)
    recall = np.mean(recall_per_class)
    f1 = np.mean(f1_per_class)

    return accuracy, precision, recall, f1 

def fit_naive_bayes_classifier(features, labels):
    """
    Fits a Naive Bayes classifier to the given features and labels.

    Parameters:
    - features (pandas.DataFrame): The feature matrix.
    - labels (pandas.Series): lables.

    Returns:
    - A function that predicts the class label for a given sample.
    """
    
    #calculating prior probabilities
    total_samples = len(labels)
    class_counts = labels.value_counts().to_dict()
    #print(class_counts)

    prior_probabilities = {}
    for label, count in class_counts.items():
        prior_probabilities[label] = count / total_samples

    #calculating likelihood probabilities
    likelihood_probabilities = {}
    alpha = 1.0

    for feature_column in features.columns:
    # counts of (feature_value, class)
        cross_counts = (
            features
            .groupby([labels, feature_column])      # <— class first, then feature
            .size()
            .unstack(fill_value=0)                  # rows = class, cols = feature_value
        )

        # cross_counts looks like:
        #           Red   Green  Blue
        # Yes         2       1     0
        # No          1       1     1

        # number of samples in each class (row sum)
        class_totals = cross_counts.sum(axis=1)    # Series: index=class, value=count
        
        V = cross_counts.shape[1]  # number of unique feature values
        # convert counts to P(value|class)
        feature_likelihoods = {}
        for cls in cross_counts.index:              # each class
            for val in cross_counts.columns:        # each feature value
                count = cross_counts.loc[cls, val]
                total = class_totals[cls]
                prob = (count + alpha) / (total + alpha * V) # applying laplace smoothing as well
                # store: for this feature column, this class, this value → probability
                feature_likelihoods.setdefault(cls, {})[val] = math.log(prob) # log propabilities to avoid underflow

        likelihood_probabilities[feature_column] = feature_likelihoods
    """ print(likelihood_probabilities)
    print("_--------------")
    print(likelihood_probabilities[feature_column].keys())
    print(likelihood_probabilities[feature_column][label].keys())
    print(likelihood_probabilities[feature_column][label]['high']) """

    def predict(sample):
        '''
        Predict each sample given

        Parameters:
            sample (np.series): each row from the X_test.

        return:
            prediction (str): predicted class of the input sample.
        '''
        #print(sample)
        #print("predict function ececution start here")
        class_log_probs = {}

        log_prob = {}

        for lable in prior_probabilities:
            log_prob[lable] = math.log(prior_probabilities[lable])

        for label, prior in log_prob.items():
            log_p = prior

            for feature_column, feature_value in sample.items():
                # add log-likelihood if exists, else add log of a tiny value (or use Laplace smoothing)
                if feature_value in likelihood_probabilities[feature_column][label]:
                    log_p += likelihood_probabilities[feature_column][label][feature_value]
                else:
                    # unseen value: pretend count=0+alpha
                    V = len(likelihood_probabilities[feature_column][label])
                    total = class_counts[label]
                    smoothed = (0 + alpha) / (total + alpha * V)
                    log_p += math.log(smoothed)
            
            class_log_probs[label] = log_p
        prediction = max(class_log_probs, key=class_log_probs.get)
        return prediction
        
    return predict 



# load dataset
df=pd.read_csv('car.data',names=['col1', 'col2','col3','col4','col5', 'col6','labels'])
#print(df.info())
#print(df.head())

labels = df['labels']
features = df.drop('labels',axis=1)

#split the data
X_train, X_test, y_train, y_test = split_dataset(features, labels)
#print(X_train[:10])
#print(y_train[:10])
#print(y_test[:10])

nb_model = fit_naive_bayes_classifier(X_train, y_train)
accuracy, precision, recall, f1 = evaluate_classifier(nb_model, X_test, y_test)

# You can print or store these metrics for your report.
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
