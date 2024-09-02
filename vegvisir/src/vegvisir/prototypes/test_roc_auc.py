from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from itertools import cycle
import matplotlib.pyplot as plt


iris_data = datasets.load_iris()
features = iris_data.data
target = iris_data.target


target = label_binarize(target,
                        classes=[0, 1, 2])


train_X, test_X,\
    train_y, test_y = train_test_split(features,
                                       target,
                                       test_size=0.25,
                                       random_state=42)

model_1 = LogisticRegression(random_state=0)\
    .fit(train_X, train_y[:, 0])
model_2 = LogisticRegression(random_state=0)\
    .fit(train_X, train_y[:, 1])
model_3 = LogisticRegression(random_state=0)\
    .fit(train_X, train_y[:, 2])

model = OneVsRestClassifier(LogisticRegression(random_state=0)) \
    .fit(train_X, train_y)
prob_test_vec = model.predict_proba(test_X)


n_classes = 3
fpr = [0] * 3
tpr = [0] * 3
thresholds = [0] * 3
auc_score = [0] * 3

for i in range(n_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(test_y[:, i],
                                              prob_test_vec[:, i])
    print(test_y[:,i])
    print(prob_test_vec[:,i])
    exit()
    auc_score[i] = auc(fpr[i], tpr[i])

