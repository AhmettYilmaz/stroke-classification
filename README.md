# Stroke Classification

![Brain Image](/brain_cell.jpg 'Neuron Activity')


What is **stroke** ❓

A stroke, sometimes called a brain :brain: attack, occurs when something blocks blood supply to part of the brain or when a blood vessel in the brain bursts. In either case, parts of the brain become damaged or die. A stroke can cause lasting brain damage, long-term disability, or even death.

Learn about the health conditions and lifestyle habits that can increase your risk for stroke. How about you?
> More information about of [stroke](https://www.cdc.gov/stroke/about.htm#:~:text=A%20stroke%2C%20sometimes%20called%20a,term%20disability%2C%20or%20even%20death.).

## Import Libraries 
```py
# Loading Fundamental Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
```

## Load the Data
```py
# Read the data using pandas
data = pd.read_csv("stroke_prediction_data.csv")
data.head()
```
**Output:**
|index|id|gender|age|hypertension|heart\_disease|ever\_married|work\_type|Residence\_type|avg\_glucose\_level|bmi|smoking\_status|stroke|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|0|9046|Male|67\.0|0|1|Yes|Private|Urban|228\.69|36\.6|formerly smoked|1|
|1|51676|Female|61\.0|0|0|Yes|Self-employed|Rural|202\.21|NaN|never smoked|1|
|2|31112|Male|80\.0|0|1|Yes|Private|Rural|105\.92|32\.5|never smoked|1|
|3|60182|Female|49\.0|0|0|Yes|Private|Urban|171\.23|34\.4|smokes|1|
|4|1665|Female|79\.0|1|0|Yes|Self-employed|Rural|174\.12|24\.0|never smoked|1|

> **Attribute Information:**
>>1. **id:** unique identifier
>>2. **gender:** Male :man:, Female :woman: or Other
>>3. **age:** age of the patient
>>4. **hypertension:** 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
>>5. **heart_disease:** 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
>>6. **ever_married:** No or Yes
>>7. **work_type:** children, govt_job, Never worked, private or self-employed
>>8. **Residence_type:** Rural or Urban
>>9. **avg_glucose_level:** average glucose level in blood
>>10. **bmi:** body mass index
>>11. **smoking_status:** formerly smoked, never smoked, smokes or Unknown*
>>12. **stroke:** 1 if the patient had a stroke or 0 if not

>❗ **Note**: "*Unknown*" in smoking_status means that the information is unavailable for this patient

---

## Descriptive Statistics :chart:
```py
# attribute types
data.info()
```
**Output:**
```
RangeIndex: 5110 entries, 0 to 5109
Data columns (total 12 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   id                 5110 non-null   int64  
 1   gender             5110 non-null   object 
 2   age                5110 non-null   float64
 3   hypertension       5110 non-null   int64  
 4   heart_disease      5110 non-null   int64  
 5   ever_married       5110 non-null   object 
 6   work_type          5110 non-null   object 
 7   Residence_type     5110 non-null   object 
 8   avg_glucose_level  5110 non-null   float64
 9   bmi                4909 non-null   float64
 10  smoking_status     5110 non-null   object 
 11  stroke             5110 non-null   int64  
dtypes: float64(3), int64(4), object(5)
memory usage: 479.2+ KB
```
We have **5110** instances and **12** attributes. Also we have some missing values in `bmi` attribute. But how we can handle missing values :question:

### Handling Missing Data

1. Deleting Row
2. Deleting Column
3. Filling with a Constant
4. Filling with Mean, Median, or Mode

When we working with data, we should always check the missing values. We do this using pandas library.

```py
# Check the missing value
for key, value in data.isnull().sum().items():
  if value > 0:
    percent = np.round(value / len(data) *100, decimals=2)
    print(f"'{key}' attribute doesn't have {percent}% of its data.")
```
`data.isnul().sum()` Return a Pandas Series for each column with missing values, but I would like to see the percentage of the missing values. This approach is more useful. 

>**Output: :arrow_down:**
```py
'bmi' attribute doesn't have 3.93% of its data.
```
##
#### Deleting Row 
```py
# Delete entire row
delete_rows_df = data.dropna(axis=0)

# Compare shape of the new data
data.shape, delete_rows_df.shape
```
>**Output: :arrow_down:**
```py
((5110, 12), (4909, 12))
```
Original data has **5110** instances, after deleting missing values it has **4909** instances.

#### Deleting Column
```py
# Delete column
delete_cols_df = data.dropna(axis=1)

# Check the shape of the data
data.shape, delete_cols_df.shape
```
>**Output: :arrow_down:**
```py
((5110, 12), (5110, 11))
```
Original data has **12** attributes, after deleting missing value column, now it has **11** attributes.

#### Filling with a Constant
```py
# Fill the missing data using pandas
data_fill_constant = data.fillna(value=0)
```
Be carefull when you are using this method because its return a new data frame or series.

#### Filling with Mean
```py
# Fill the missing data using pandas
mean_value = np.mean(data["bmi"])
data_fill_mean = data.fillna(value=np.round(mean_value, decimals=1))
```
The other methods (median and mode) work as this.

After all these method I want to delete all rows which has missing value.

```py
data = data.dropna(axis=0)
```
### Basic Statistic Information
If we want to see some statistical information like mean, std, etc. we can use `describe` method.

```py
# Basic statistic
data.describe().apply(lambda x: x.apply('{0:.2f}'.format)).T
```
This `apply` method with `lambda` function doing as a formatter.


>**Output: :arrow_down:**

|index|count|mean|std|min|25%|50%|75%|max|
|---|---|---|---|---|---|---|---|---|
|id|4909\.00|37064\.31|20995\.10|77\.00|18605\.00|37608\.00|55220\.00|72940\.00|
|age|4909\.00|42\.87|22\.56|0\.08|25\.00|44\.00|60\.00|82\.00|
|hypertension|4909\.00|0\.09|0\.29|0\.00|0\.00|0\.00|0\.00|1\.00|
|heart\_disease|4909\.00|0\.05|0\.22|0\.00|0\.00|0\.00|0\.00|1\.00|
|avg\_glucose\_level|4909\.00|105\.31|44\.42|55\.12|77\.07|91\.68|113\.57|271\.74|
|bmi|4909\.00|28\.89|7\.85|10\.30|23\.50|28\.10|33\.10|97\.60|
|stroke|4909\.00|0\.04|0\.20|0\.00|0\.00|0\.00|0\.00|1\.00|

Some data columns don't have any meaningful information because they seem integer but accualy they are categorical attributes. (id, hypertension, heart disease and stroke).

## Data Visualization :sunrise:

We already know three numerical attributes what we can use for visualization. WE can plot *histograms* and *scatter* plots. Start with scatter plots and this shown us how looks like our data.

```py
# Scatter Plots
sns.set_palette("bright")
fig, [ax1, ax2] = plt.subplots(ncols=2, nrows=1, figsize=(15,5),  
                               dpi=120, sharex=False, sharey=False)

sns.scatterplot(x=data.index, y=data["avg_glucose_level"], alpha=0.6, ax=ax1)
sns.scatterplot(x=data.index, y=data["bmi"], alpha=0.6, ax=ax2)

ax1.set_title("Glucose Level")
ax1.set_xlabel("Index")
ax2.set_title("BMI")
ax2.set_xlabel("Index")

plt.tight_layout()
plt.savefig("scatter_plot_with_outlier.png", dpi=200, bbox_inches="tight")
plt.show()
```
> **Output**:arrow_down:

![Scatter Plot](/scatter_plot_with_outlier.png "Scatter plot with outliers")

We can see the some `bmi`(body mass index) values very high and its create some difficulties wheen model predicting target variable. Let's look athe histogram for these attributes.

```py
# Histograms
sns.set_style("whitegrid"), sns.set_palette("bright")

fig, [ax1, ax2] = plt.subplots(ncols=2, nrows=1, figsize=(15,5),  
                               dpi=120, sharex=False, sharey=False)

sns.histplot(data["avg_glucose_level"], ax=ax1, kde=True)
sns.histplot(data["bmi"], ax=ax2, kde=True)

ax1.set_title("Glucose Level")
ax2.set_title("BMI")

plt.tight_layout()
plt.savefig("histogram_with_outlier.png", dpi=200, bbox_inches="tight")

plt.show()
```
> **Output**:arrow_down:

![Histogram](/histogram_with_outlier.png "Distributions with outliers")

`avg_glucose_level` attribute fit the [Poission distribution](https://en.wikipedia.org/wiki/Poisson_distribution) . On the other hand, `bmi` attribute looks like [Normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) but nor perfectly normal. We call this `right-skewed distribution`, outliers might be the reason for this. 

### Detect and Remove Outliers

I want to use `zscore` method from `scipy` library. There is different ways to detect outliers. 


```python
from scipy import stats

for col in ["avg_glucose_level", "bmi"]:
  # Calculate the z-score each patient's body mass index
  z_scores = np.abs(stats.zscore(data[col]))

  # Identify outliers as patients with a z-score greater than 3
  threshold = 3
  outliers = data[z_scores > threshold]

  # Remove outliers from data
  data.drop(outliers.index, inplace=True)
```
Data has **4909** instance before, but now its **4697**. 
:exclamation: If you select bigger or smaller treshold value then number of instance change.


After this process, how looks like the histogram for each value :question:

![Histogram Without Outliers](/histogram_without_outlier.png "Histogram Without Outliers")

We see the difference especialy on the `bmi` attribute. It is look normal distribution without any skew.

---
## Split Data 

Let's seperate the data to features and target variable. The target variable is `stroke` column. Also we don't need `id` column so we seperate it.
```python
# Seperate features and target variable
X = data.drop(["stroke", "id"], axis=1)
y = data["stroke"]
```
`X` refers to input variables, `y` refers to output variable.

After that, we should encode the categorical attributes. I want to use `pd.get_dummies` for encoding. 

```python
# Encode the categorical features
X = pd.get_dummies(X)
```

We need to check target attribute is balanced or not. We can use `value_counts` method for this purpose.
```python
# Check target class
for key, value in y.value_counts().items():
  print(f"Target class `{key}` represented {(value/len(y)*100):.2f}% of dataset.")
```
> **Output :arrow_down:**

```python
Target class `0` represented 95.91% of dataset.
Target class `1` represented 4.09% of dataset.
```
:eyes: Look at that :exclamation:
Almost every instance belongs to a class.

How can we balance this dataset :question:

We should balance the data using `imblearn` library. 

* *SMOTE*
* *ADASYN*
* *RandomOverSampling*
* *BorderlineSMOTE*
* *KMeansSMOTE*
* *SVMSMOTE*

More information about resample the data, visit the 
[`imblearn website`](https://imbalanced-learn.org/stable/references/over_sampling.html "visit imblearn site").

### SMOTE
SMOTE, which stands for Synthetic Minority Over-sampling 
Technique, is a method used in machine learning and data mining 
to address the class imbalance problem in classification. The 
class imbalance problem occurs when one class in the dataset has 
significantly fewer instances than the other, leading to a 
biased model that may perform poorly on the minority class.

```python
from imblearn.over_sampling import SMOTE

# Define Estimator
smt = SMOTE(random_state=42)

# SMOTE
X_smt, y_smt = smt.fit_resample(X, y)
```

```python
# SMOTE result
for key, value in y_smt.value_counts().items():
  print(f"Target class `{key}` represented {(value/len(y_smt)*100):.2f}% of dataset.")
```
> **Output** :arrow_down:
```python
Target class `1` represented 50.00% of dataset.
Target class `0` represented 50.00% of dataset.
```
:white_check_mark: Now we have balanced dataset.

---

## Define Model
I want to use `Logistic Regression` model from `scikit-learn` 
library. 
Logistic Regression is a statistical method used for binary 
classification, which is the task of categorizing an input into 
one of two possible classes. Despite its name, logistic 
regression is used for classification rather than regression 
problems.

The logistic regression model is a type of generalized linear 
model, specifically designed for predicting the probability of 
an instance belonging to a particular class. The output of 
logistic regression is transformed using the logistic function 
(also called the sigmoid function), which ensures that the 
predicted values fall between 0 and 1. 

The logistic function is defined as:

$\sigma(x) = \dfrac{1}{1-e^{-x}}$

* $\sigma(x)$ is the sigmoid function.

* $e$ represents the mathematical constant Euler's 
number, approximately equal to 2.71828.

* $x$ is the linear combination of input features and 
their associated weights, as explained in the logistic 
regression context.

1. Split the data to train and test set
```python
from sklearn.model_selection import train_test_split

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_smt, y_smt, test_size=0.2, random_state=42)
```
2. Scale the data
```python
from sklearn.preprocessing import StandardScaler

# Define a scaler
scaler = StandardScaler()

# Fit and transform train data
X_train = scaler.fit_transform(X_train)

# Transform the test data
X_test = scaler.transform(X_test)
```
❗only **transform** the **test** data

3. Fit the model
```python
from sklearn.linear_model import LogisticRegression

# Define a model
model = LogisticRegression()

# Fit the model
model.fit(X_train, y_train)
```
4. Cross-Validation
```python
from sklearn.model_selection import cross_val_score

# Cross-validation the model
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")

# Result for cross validation
avg_acc = np.round(np.mean(scores), decimals=2)
std_acc = np.round(np.std(scores), decimals=5)
```
Model works 5 times, `cv`, and collect the accuracy for each epoch. After cross validation average accuracy is **96%** and standard deviation of accuracy is **0.204%**, which is very good for the prediction.

5. Plot Confusion Matrix
```python
from sklearn.metrics import confusion_matrix

# Make prediction for train data
preds = model.predict(X_test)

# Create confusion matrix
cm = confusion_matrix(y_test, preds)

# Plot the confusion matrix
plt.figure(figsize=(5,5), dpi=150)
sns.heatmap(cm, annot=True, cmap="Greens", fmt="d", cbar=False,
            lw=0.5)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix")
plt.savefig("lg_confusion_matrix.png", bbox_inches="tight", dpi=400)
plt.show()
```
> **Output :arrow_down:**

![Confusion Matrice](/lg_confusion_matrix.png "Confusion Matrix")
 
Model's predictions looks very well. 
How about `precision`, `recall` or `F1` scores?

We can use classification report from `sklearn.metrics`

```python
 from sklearn.metrics import classification_report

# Classification Report
cr = classification_report(y_test, preds)
```
> **Output :arrow_down:** 

|       |precision |   recall | f1-score |  support|
|-------|----------|----------|----------|---------|
| 0     |   0.93   |   1.00   |   0.96   |    896  |
| 1     |   1.00   |   0.93   |   0.96   |    942  |
|       |          |          |          |         |
|accuracy |                 | | 0.96     |  1838   |
|macro avg| 0.96   |  0.96    |    0.96  |  1838   |
|weighted avg| 0.96|   0.96   |   0.96   |     1838|

:white_check_mark: Model predicted good. :smile: