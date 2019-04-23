import pandas as pd
import numpy as np


df = pd.read_csv("train_file.csv")
'''
print(len(df["Greater_Risk_Question"].unique()))
print((df["Race"].unique()))
print(len(df["GeoLocation"].unique()))
print(len(df["QuestionCode"].unique()))
print(len(df["LocationDesc"].unique()))
print(df.shape)

#print(df.describe())

'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
encoder = LabelEncoder()
encoder.fit(df['Sex'].drop_duplicates())
df['Sex'] = encoder.transform(df['Sex'])
encoder.fit(df['LocationDesc'].drop_duplicates())
df['LocationDesc'] = encoder.transform(df['LocationDesc'])
#data1 = pd.get_dummies(df['StratificationType'])
encoder.fit(df['StratificationType'].drop_duplicates())
df['StratificationType'] = encoder.transform(df['StratificationType'])
encoder.fit(df['Race'].drop_duplicates())
df['Race'] = encoder.transform(df['Race'])

#print(df["LocationDesc"][:21])

#print(df.info())
#df = df.values
#print(df)
#X = df[:, [0,1,2,3,6,7,8,9,12,13,14,15]]
X = df[["Patient_ID","LocationDesc","Subtopic", "Sex",	"Race",	"Grade"]].copy()

#X = df1[:, [0,1,2,3,6,7,8,9]]

Y = df[["Greater_Risk_Probability"]].copy()

#print(X,Y)

import statsmodels.formula.api as sm

#X_opt =  X
#regressor_OLS = sm.OLS (endog = Y, exog = X_opt.values).fit()
#print(regressor_OLS.summary())


def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(Y, x).fit()
        maxVar = max(regressor_OLS.pvalues)

        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    print(regressor_OLS.summary())

    return x


SL = 0.05
X_opt = X
X_modeled = backwardElimination(X_opt.values, SL)
print(X_modeled)


#testing data
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()

lin_reg.fit(X, Y)
print("DOne")

df_test = pd.read_csv("test_file.csv")


encoder.fit(df_test['Sex'].drop_duplicates())
df_test['Sex'] = encoder.transform(df_test['Sex'])
encoder.fit(df_test['LocationDesc'].drop_duplicates())
df_test['LocationDesc'] = encoder.transform(df_test['LocationDesc'])
#data1 = pd.get_dummies(df['StratificationType'])
encoder.fit(df_test['StratificationType'].drop_duplicates())
df_test['StratificationType'] = encoder.transform(df_test['StratificationType'])
encoder.fit(df_test['Race'].drop_duplicates())
df_test['Race'] = encoder.transform(df_test['Race'])


#df_test = df_test.values
#print(df)
#X_test = df_test[["Patient_ID", "YEAR","LocationDesc","Subtopic", "Sex",	"Race",	"Grade", "StratID1", "StratID2", "StratID3", "StratificationType"]].copy()
X_test = df_test[["Patient_ID","LocationDesc","Subtopic", "Sex",	"Race",	"Grade"]].copy()

#print(X_test)

#df_test["LocationDesc"] = val
df_test["Greater_Risk_Probability"]=lin_reg.predict(X_test)

#print(pred_y)
#print(df_test["LocationDesc"])
print(df_test["Greater_Risk_Probability"])
#df_test["LocationDesc"] = encoder.inverse_transform(df_test["LocationDesc"])
#L = list(encoder.inverse_transform(df_test["LocationDesc"]))
#df_result = pd.concat([X_test,pred_y])
#print(df_test["LocationDesc"])
#print(df_result)
df_test.to_csv('output4.csv', index= False)

#import statsmodels.formula.api as sm

#X_opt =  X.drop(["LocationDesc"], axis=1)
#regressor_OLS = sm.OLS (endog = Y, exog = X_opt.values).fit()
#print(regressor_OLS.summary())
