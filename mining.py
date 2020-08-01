import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fbprophet import Prophet
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import math
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV 
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.neighbors import KNeighborsClassifier

def read_csv(path):
	return pd.read_csv(path)
def stastic_hotel(df):
	plt.figure(figsize=(6,6))
	plt.title(label='Cancellations by Hotel Types')
	sns.countplot(x='hotel',hue='is_canceled',data=df)
	plt.show()
def stastic_isCancel_leadTime(df):
	plt.figure(figsize=(12,6))
	plt.title(label='Cancellation by Lead Time')
	sns.barplot(x='hotel',y='lead_time',hue='is_canceled',data=df)
	plt.show()
def stastic_deposit(df):
	plt.figure(figsize=(6,6))
	plt.title(label='sự hủy bỏ theo chính sách có  đặc cọc trước')
	sns.countplot(x='deposit_type',hue='is_canceled',data=df)
	plt.show()

def static_non_value(df):
	print( df.isna().sum(), sep='\n')

def choose_feature(df):
	y = df['is_canceled'].values
	df['reservation_status_date'] = 0
	df = df.drop(['is_canceled','meal','stays_in_week_nights','stays_in_weekend_nights','adr','reservation_status','company','reservation_status_date'],axis=1)
	return df,y

def data_tranform(datas):
	datas['arrival_date_month'] = datas['arrival_date_month'].map({'January':1, 'February': 2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7,
                                                            'August':8, 'September':9, 'October':10, 'November':11, 'December':12})
	
	datas['hotel'] = datas['hotel'].map({'Resort Hotel':0, 'City Hotel': 1})
	datas = pd.concat([datas,pd.get_dummies(datas['country'], prefix='country')],axis=1)
	datas = pd.concat([datas,pd.get_dummies(datas['market_segment'], prefix='market_segment')],axis=1)
	datas = pd.concat([datas,pd.get_dummies(datas['distribution_channel'], prefix='distribution_channel')],axis=1)
	datas = pd.concat([datas,pd.get_dummies(datas['reserved_room_type'], prefix='reserved_room_type')],axis=1)
	datas = pd.concat([datas,pd.get_dummies(datas['assigned_room_type'], prefix='assigned_room_type')],axis=1)
	datas = pd.concat([datas,pd.get_dummies(datas['deposit_type'], prefix='customer_type')],axis=1)
	datas = pd.concat([datas,pd.get_dummies(datas['customer_type'], prefix='customer_type')],axis=1)
	datas = datas.drop(['deposit_type','country','market_segment','distribution_channel','reserved_room_type','assigned_room_type','customer_type'],axis=1)
	return datas.values	
	

def nan_procesing(X):
	(n,m) = X.shape
	for i in range(n):
		for j in range(m):
			if  math.isnan(X[i][j])  :
				X[i][j] = np.nanmean(X[i])
	return X
def clasifycation_use_SVM(X_train, X_test, y_train, y_test ):
	param_grid = {'C': [0.1, 1, 10, 100, 1000]}
	grid = GridSearchCV(LinearSVC(), param_grid, refit = True, verbose = 1) 

	grid.fit(X_train, y_train)
	print(grid.best_params_) 

	print(grid.best_estimator_) 
	svm = LinearSVC(C = 10000)
	y_pred = grid.predict(X_test)
	target_names = ['is_canceled = 0', 'is_canceled = 1']
	print(classification_report(y_test, y_pred, target_names=target_names))
	print(f1_score(y_test, y_pred, average='micro'))
	print(confusion_matrix(y_test, y_pred) )

def clasifycation_use_DT(X_train, X_test, y_train, y_test ):

	decisiontree = tree.DecisionTreeClassifier(criterion = 'entropy')
	param_grid = [{'min_samples_split':range(10,500,20),
              'max_depth': range(1,20,2)}]
	grid = GridSearchCV(decisiontree, param_grid) 

	grid.fit(X_train, y_train)
	print(grid.best_params_) 

	print(grid.best_estimator_) 

	y_pred = grid.predict(X_test)

	target_names = ['is_canceled = 0', 'is_canceled = 1']
	print(classification_report(y_test, y_pred, target_names=target_names))
	print(f1_score(y_test, y_pred, average='micro'))
	print(confusion_matrix(y_test, y_pred) )

def clasifycation_use_logistic_regresion(X_train, X_test, y_train, y_test ):
	param_grid = {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}# l1 lasso l2 ridge
	model = LogisticRegression()
	grid = GridSearchCV(model,param_grid,cv=10)	
	grid.fit(X_train, y_train)
	print(grid.best_params_) 

	print(grid.best_estimator_) 

	y_pred = grid.predict(X_test)

	target_names = ['is_canceled = 0', 'is_canceled = 1']
	print(classification_report(y_test, y_pred, target_names=target_names))
	print(f1_score(y_test, y_pred, average='micro'))
	print(confusion_matrix(y_test, y_pred) )

def classifycation_use_naivebyes(X_train, X_test, y_train, y_test ):
	clf = MultinomialNB()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print('MultinomialNB:')
	target_names = ['is_canceled = 0', 'is_canceled = 1']
	print(classification_report(y_test, y_pred, target_names=target_names))
	print(f1_score(y_test, y_pred, average='micro'))
	print(confusion_matrix(y_test, y_pred) )
	
	clf = GaussianNB()
	clf.fit(X_train, y_train)
	y_pred = clf.predict(X_test)
	print('GaussianNB:')
	target_names = ['is_canceled = 0', 'is_canceled = 1']
	print(classification_report(y_test, y_pred, target_names=target_names))
	print(f1_score(y_test, y_pred, average='micro'))
	print(confusion_matrix(y_test, y_pred) )	
def classifycation_use_KNN(X_train, X_test, y_train, y_test ):
	print("KNN ")
	K = int( math.sqrt(X_train.shape[0]) )
	if K % 2 == 0:
		K += 1
	neigh = KNeighborsClassifier(n_neighbors=K,algorithm = 'ball_tree')
	neigh.fit(X_train, y_train)
	y_pred = neigh.predict(X_test)
	target_names = ['is_canceled = 0', 'is_canceled = 1']
	print(classification_report(y_test, y_pred, target_names=target_names))
	print(f1_score(y_test, y_pred, average='micro'))
	print(confusion_matrix(y_test, y_pred) )




def main():
	path = 'hotel_bookings.csv'
	hotel_bookings = read_csv(path)
	print(len(hotel_bookings.columns ))
	# print(hotel_bookings.describe().T)
	#static_non_value(hotel_bookings)
	X,y = choose_feature(hotel_bookings)
	X = data_tranform(X)
	
	X = nan_procesing(X)
	print(X.shape)
	min_max_scaler = preprocessing.MinMaxScaler()
	X = min_max_scaler.fit_transform(X)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
	clasifycation_use_logistic_regresion(X_train, X_test, y_train, y_test)
	# clasifycation_use_DT(X_train, X_test, y_train, y_test )
	classifycation_use_naivebyes(X_train, X_test, y_train, y_test)
	# classifycation_use_KNN(X_train, X_test, y_train, y_test)
	
if __name__ == '__main__':
	main()