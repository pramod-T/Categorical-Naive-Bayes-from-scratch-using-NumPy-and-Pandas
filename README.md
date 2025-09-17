# Categorical-Naive-Bayes-from-scratch-using-NumPy-and-Pandas

## Naive Bayes Classifier

Naive Bayes is a simple yet powerful probabilistic classifier based on Bayes’ Theorem.

We want to find the class $y$ that maximizes the posterior probability:

$​P(y | x1, x2, ..., xn)$ ∝ $P(y)$ * $Π_{j=1}^n$ $P(x_j \mid y)$

Where:

$P(y)$ is the prior probability — how common class $y$ is

$P(x_j \mid y)$ is the likelihood — how likely we see feature value $x_j$ if the sample belongs to class $y$

The "Naive" part assumes all features are conditionally independent given the class.

This project implements a Categorical Naive Bayes Classifier from scratch in Python.

## Step 1 — Computing Prior Probabilities

```
    #calculating prior probabilities
    total_samples = len(labels)
    class_counts = labels.value_counts().to_dict()
    #print(class_counts)

    prior_probabilities = {}
    for label, count in class_counts.items():
        prior_probabilities[label] = math.log(count / total_samples)
```
Explanation: 

- Count how many training samples belong to each class

- Divide by total samples

This gives $P(y)$ — the base probability of each class.

Example:

If you have 10 samples: 6 "Yes" and 4 "No":

- P(Yes) = 6/10 = 0.6

- P(No) = 4/10 = 0.4$

## Computing Likelihoods $P(x \mid y)$ with Laplace Smoothing
```
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
```
What happens:

- Groups data by (class, feature_value) and counts how many times each combination appears

- For each class and feature value, compute the conditional probability

## Laplace Smoothing

- If a feature value never appeared with a class, its probability would be 0.
- We fix this with Laplace smoothing adding a Constant: A smoothing parameter, denoted as α (alpha) , is added to the observed counts. For standard Laplace smoothing, α is set to 1. 
- Adjust the Denominator: The denominator of the probability calculation is also adjusted by adding α times the number of possible values for that feature that is 'V' in our code. This keeps all probabilities valid and sums to 1 across the 
𝑉 Values.

  $P(x_j \mid y)$ = $count(x_j​,y)+α​$ / $count(y)+α⋅V$

Where:

- $count(x_j​,y)​$ = number of samples of class $𝑦$ having feature value $𝑥_𝑗$
- ​count($𝑦$) = total number of samples of class $𝑦$
- α = smoothing constant (usually 1)
- V = number of possible distinct feature values for this feature

## Log Probabilities

- We store math.log(prob) instead of prob directly.
- multiplying many small numbers quickly makes the result very tiny, which can cause underflow errors in computers.
- log turn multiplication into addition which is faster and stable.
- This avoids numerical underflow when multiplying many small probabilities

  $P(Y \mid X)$ = $log P(y)$ + $∑_{j=1}^n$ $log P(x_j | y)$

where X = { x1,x2,x3,.....}

## Step 3 — Making Predictions

```
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

        for label, prior in prior_probabilities.items():
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
```

How it works: 

For each possible class:

- Start with $\log P(y)$

- Add $\log P(x_j \mid y)$ for each feature value in the sample

- If a value never appeared in training, we do on-the-fly Laplace smoothing

- Choose the class with the highest total log-probability

Mathematically

Y = argymax $log P(y)$ + $∑_{j}$ $log P(x_j | y)$

## Limitations of Naive Bayes

While Naive Bayes is simple and fast, it has some important limitations:

- Strong independence assumption : Assumes all features are conditionally independent given the class — which is rarely true in real-world data. Correlated features can reduce accuracy.

- Zero-frequency problem : If a feature value never appears with a class in training, the probability becomes zero and wipes out the whole prediction.
(We fix this with Laplace smoothing.)

- The categorical version of Naive Bayes works best on discrete/categorical data. For continuous features, Gaussian Naive Bayes is used instead.

## Common Use Cases of Naive Bayes

Despite its limitations, Naive Bayes works surprisingly well in many situations, especially with text and categorical data:

- Text classification / spam filtering (e.g. Spam vs. Not Spam emails — each word is a feature)

- Sentiment analysis (e.g. Positive vs. Negative movie reviews)

- Document/topic classification

- Medical diagnosis (with categorical features)

- Real-time prediction tasks (Because it is fast and needs little memory)

# Note 
- This is a full Categorical Naive Bayes implementation from scratch 
- It builds probability tables from training data, then scores each class for a new sample using Bayes’ rule.

