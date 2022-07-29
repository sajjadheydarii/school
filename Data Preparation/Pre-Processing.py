# **Import the Libraries**

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import category_encoders as ce

# **Import the Dataset**

data_cat = pd.read_csv("international-migration-June-2019.csv")
data_cat

"""## **Data Preparation**

### **Pre-Processing**

**Handle Categorical Data**
"""
data_cat1 = data_cat.drop(["year_month", "month_of_release"], axis = 1)
data_cat2 = data_cat1.drop("passenger_type", axis = 1)
data_cat3 = data_cat2[data_cat2["sex"] != "TOTAL"].reset_index().drop("index", axis = 1)
data_cat4_1 = data_cat3.replace({"sex" : "Female"}, 0).replace({"sex" : "Male"}, 1)

encoder = ce.OneHotEncoder(cols = "sex", use_cat_names = True)
data_cat4_2 = encoder.fit_transform(data_cat3)

data_cat5 = data_cat4_1[(data_cat4_1["age"] == "0-4 years") | (data_cat4_1["age"] == "5-9 years") | (data_cat4_1["age"] == "10-14 years") | (data_cat4_1["age"] == "15-19 years")]

data_cat6 = data_cat5.replace({"age" : "0-4 years"}, 1).replace({"age" : "5-9 years"}, 2).replace({"age" : "10-14 years"}, 3).replace({"age" : "15-19 years"}, 4)

data_cat7 = data_cat6.replace({"status" : "Final"}, 0).replace({"status" : "Provisional"}, 1)

data_cat8 = pd.get_dummies(data_cat7)


"""**Handle Missing Values**"""

data_mv = pd.read_csv("diabetes.csv")

data_mv.isnull().sum()                            #isna()

#Filling Missing values
data_mv1 = data_mv.fillna({"BloodPressure" : 70,
                           "SkinThickness" : 0,
                           "Insulin" : 0, "BMI" : 32.1,
                           "DiabetesPedigreeFunction" : 0.473})

#Removing Missing Values
data_mv2 = data_mv1.dropna().reset_index().drop("index", axis = 1)

"""**Handle Outliers**"""

fig = plt.figure(figsize = (13,6))
data_mv2.boxplot()

#Insulin
Q1 = data_mv2.iloc[:,4].quantile(0.25)
Q3 = data_mv2.iloc[:,4].quantile(0.75)

LB = Q1 - 1.5 * (Q3 - Q1)
UB = Q3 + 1.5 * (Q3 - Q1)

data_mv2[data_mv2["Insulin"] > 305].index
data_mv2[data_mv2["Insulin"] <= 305].reset_index().drop("index", axis = 1)

#BloodPressure (Box plot)
Q1 = data_mv2.iloc[:,2].quantile(0.25)
Q3 = data_mv2.iloc[:,2].quantile(0.75)

LB = Q1 - 1.5 * (Q3 - Q1)
UB = Q3 + 1.5 * (Q3 - Q1)

data_mv2[(data_mv2["BloodPressure"] > 104) | (data_mv2["BloodPressure"] < 40)].index

#BloodPressure (Statistical method)
Xbar = 69.217450
Sigma = 18.958386	
n = 745
Zhalfa = 1.96

LB = Xbar - ((Zhalfa * Sigma) / np.sqrt(n))
UB = Xbar + ((Zhalfa * Sigma) / np.sqrt(n))

"""**Handle Duplicate Data**"""

data_mv2.duplicated().sum()
data_mv3 = data_mv2.drop_duplicates()