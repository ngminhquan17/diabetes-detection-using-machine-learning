import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import pickle
from lazypredict.Supervised import LazyClassifier

data = pd.read_csv("diabetes.csv")

print(data.info())
result = data.describe()

plt.figure(figsize=(10, 10))
sn.histplot(data["Outcome"])
plt.title("Lable distribution")
plt.show()

sn.heatmap(data.corr(), annot=True)
plt.show() 

#data visualization

#chia theo chieu doc
target = "Outcome"
x = data.drop(target, axis=1)
y = data[target]

#chia theo chieu ngang
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

scaler = StandardScaler()
#fit: đưa dữ liệu vào tính toán để tìm ra cách transform
#transform: apply cách đó để biến đổi dữ liệu
#fit_transform: làm cả 2

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

cls = SVC()
cls.fit(X_train, y_train)

y_predict = cls.predict(X_test)

for i, j in zip(y_test, y_predict):
  print("Actual: {}. Predict: {}".format(i,j))

print(classification_report(y_test, y_predict))

cm = np.array(confusion_matrix(y_test, y_predict, labels=[0,1]))
confusion = pd.DataFrame(cm, index=["Khỏe", "Bệnh"], columns=["Khỏe", "Bệnh"])
sn.heatmap(confusion, annot=True, fmt="g")
plt.savefig("confusion_matrix.png")

params = {
    "n_estimators": [50, 100, 200],
    "criterion": ["gini", "entropy", "log_loss"],
    "max_features": ["sqrt", "log2", None]
}
cls = GridSearchCV(RandomForestClassifier(), param_grid=params, cv=6, verbose=1, scoring="f1", n_jobs=-1)
cls.fit(X_train, y_train)

# load the model from disk
cls = pickle.load(open("finalized_model.pickle", 'rb'))

print(cls.best_estimator_)
print(cls.best_score_)
print(cls.best_params_)
y_predict = cls.predict(X_test)
print(classification_report(y_test, y_predict))

# save the model to disk
filename = 'finalized_model.pickle'
pickle.dump(cls, open(filename, 'wb'))

# Run multiple model at the same time
clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
models,predictions = clf.fit(X_train, X_test, y_train, y_test)
print(models)