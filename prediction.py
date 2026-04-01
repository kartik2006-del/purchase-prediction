import pandas as pd
import numpy as np

np.random.seed(42)

n = 100

age = np.random.randint(20, 60, n)
salary = np.random.randint(20000, 100000, n)
experience = np.random.randint(1, 20, n)

buy = []
for i in range(n):
    if salary[i] > 50000 and experience[i] > 5:
        buy.append(1)
    else:
        buy.append(0)

df = pd.DataFrame({
    'age': age,
    'salary': salary,
    'experience': experience,
    'buy': buy
})


x=df[['age','salary','experience']]
y=df['buy']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix


print('_____MODEL-1(LOGISTIC REGRESSION)______'.center(60))

# model-1
model=LogisticRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print('actual :',y_test.values)
print('predict :',y_pred)
print('Logistic_accuracy :',accuracy_score(y_test,y_pred))
print('confusion_matrix :',confusion_matrix(y_test,y_pred))

print('_____MODEL-2(RANDOM FOREST)_____'.center(60))

#model-2
model2=RandomForestClassifier(n_estimators=100,max_depth=3)
model2.fit(x_train,y_train)
y_pred2=model2.predict(x_test)
print('actual :',y_test.values)
print('predict :',y_pred2)
print('new_pred :',model2.predict(pd.DataFrame({'age':[30],'salary':[40000],'experience':[3]})))
print('Random_Forest_accuracy :',accuracy_score(y_test,y_pred2))
print('confusion_matrix :',confusion_matrix(y_test,y_pred2))

print('_____MODEL-3(KNN)_____'.center(60))

#model-3
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
model3=KNeighborsClassifier(n_neighbors=3)
model3.fit(x_train,y_train)
y_pred3=model3.predict(x_test)
print('actual :',y_test.values)
print('predict :',y_pred3)
print('new_pred :',model3.predict(scaler.transform(pd.DataFrame({'age':[30],'salary':[40000],'experience':[3]}))))
print('KNN_accuracy :',accuracy_score(y_test,y_pred3))
print('confusion_matrix :',confusion_matrix(y_test,y_pred3))
