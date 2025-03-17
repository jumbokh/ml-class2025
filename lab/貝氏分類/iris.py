from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from Gaussian_NB import Gaussian_NB

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
data = np.concatenate([X_train,y_train.reshape(-1,1)],axis = 1)

nb = Gaussian_NB(data)
nb.fit()
print(sum(nb.predict(X_test)==y_test)/len(y_test))