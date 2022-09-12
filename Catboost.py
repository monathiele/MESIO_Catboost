################################################################################
############################### INITIALIZE #####################################
################################################################################
import numpy as np
import pandas as pd
import os

# Read data
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

# Warning "A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead" removing:
pd.options.mode.chained_assignment = None
pd.set_option("display.max_rows", 50, "display.max_columns", None)

################################################################################
################################# FEATURE ######################################
############################### ENGINEERING ####################################
################################################################################
# Feature types
Features=train.dtypes.reset_index()
Categorical=Features.loc[Features[0]=='object','index']

# Categorical to the begining
cols = train.columns.tolist()
pos=0
for col in Categorical:
	cols.insert(pos, cols.pop(cols.index(col)))
	pos+=1
train = train[cols]
cols.remove('TARGET')
test = test[cols]

# 1) Missings
################################################################################
# Function to print columns with at least n_miss missings
def miss(ds,n_miss):
	miss_list=list()
	for col in list(ds):
		if ds[col].isna().sum()>=n_miss:
			print(col,ds[col].isna().sum())
			miss_list.append(col)
	return miss_list
# Which columns have 1 missing at least...
print('\n################## TRAIN ##################')
m_tr=miss(train,1)
print('\n################## TEST ##################')
m_te=miss(test,1)

# Missings in categorical features (fix it with an 'NA' string)
################################################################################
for col in Categorical:
	train.loc[train[col].isna(),col]='NA'
	test.loc[test[col].isna(),col]='NA'

# Missings -> Drop some rows
################################################################################
# We can see a lot of colummns with 3 missings in train, look the data and...
# there are 4 observations that have many columns with missing values:
# A1039
# A2983
# A3055
# A4665
train = train[train['ID']!='A1039']
train = train[train['ID']!='A2983']
train = train[train['ID']!='A3055']
train = train[train['ID']!='A4665']

train.reset_index(drop=True,inplace=True)

# 2) Correlations
################################################################################
# Let's see if certain columns are correlated
# or even that are the same with a "shift"
thresholdCorrelation = 0.99
def InspectCorrelated(df):
	corrMatrix = df.corr().abs() # Correlation Matrix
	upperMatrix = corrMatrix.where(np.triu(np.ones(corrMatrix.shape),k=1).astype(np.bool))
	correlColumns=[]
	for col in upperMatrix.columns:
		correls=upperMatrix.loc[upperMatrix[col]>thresholdCorrelation,col].keys()
		if (len(correls)>=1):
			correlColumns.append(col)
			print("\n",col,'->', end=" ")
			for i in correls:
				print(i, end=" ")
	print('\nSelected columns to drop:\n',correlColumns)
	return(correlColumns)

# Look at correlations in the original features
correlColumns=InspectCorrelated(train.iloc[:,len(Categorical):-1])

# If we are ok, throw them:
train=train.drop(correlColumns,axis=1)
test=test.drop(correlColumns,axis=1)

# 3) Constants
################################################################################
# Let's see if there is some constant column:
def InspectConstant(df):
	consColumns=[]
	for col in list(df):
		if len(df[col].unique())<2:
			print(df[col].dtypes,'\t',col,len(df[col].unique()))
			consColumns.append(col)
	print('\nSelected columns to drop:\n',consColumns)
	return(consColumns)

consColumns=InspectConstant(train.iloc[:,len(Categorical):-1])

# If we are ok, throw them:
train=train.drop(consColumns,axis=1)
test=test.drop(consColumns,axis=1)

################################################################################
################################ MODEL CATBOOST ################################
################################# TRAIN / TEST #################################
################################################################################
pred=list(train)[1:-1]
X_train=train[pred].reset_index(drop=True)
Y_train=train['TARGET'].reset_index(drop=True)
X_test=test[pred].reset_index(drop=True)

# 1) For expensive models (catboost) we first try with validation set (no cv)
################################################################################
from catboost import CatBoostClassifier
from catboost import Pool

# train / test partition
RS=1234 # Seed for partitions (train/test) and model random part
TS=0.3 # Validation size
esr=100 # Early stopping rounds (when validation does not improve in these rounds, stops)

from sklearn.model_selection import train_test_split
x_tr, x_val, y_tr, y_val = train_test_split(X_train, Y_train, test_size=TS, random_state=RS)

# Categorical positions for catboost
Pos=list()
As_Categorical=Categorical.tolist()
As_Categorical.remove('ID')
for col in As_Categorical:
    Pos.append((X_train.columns.get_loc(col)))

# To Pool Class (for catboost only)
pool_tr=Pool(x_tr, y_tr,cat_features=Pos)
pool_val=Pool(x_val, y_val,cat_features=Pos)

# By-hand paramter tuning. A grid-search is expensive
# We test different combinations
# See parameter options here:
# "https://catboost.ai/en/docs/references/training-parameters/"
model_catboost_val = CatBoostClassifier(
          eval_metric='AUC',
          iterations=3000, # Very high value, to find the optimum
          od_type='Iter', # Overfitting detector set to "iterations" or number of trees
          random_seed=RS, # Random seed for reproducibility
          verbose=100) # Shows train/test metric every "verbose" trees

# "Technical" parameters of the model:
params = {'objective': 'Logloss',
		  'learning_rate': 0.01, # learning rate, lower -> slower but better prediction
		  'depth': 5, # Depth of the trees (values betwwen 5 and 10, higher -> more overfitting)
		  'min_data_in_leaf': 50,
		  'l2_leaf_reg': 15, # L2 regularization (between 3 and 20, higher -> less overfitting)
		  'rsm': 0.5, # % of features to consider in each split (lower -> faster and reduces overfitting)
		  'subsample': 0.5, # Sample rate for bagging
		  'random_seed': RS}

model_catboost_val.set_params(**params)

print('\nCatboost Fit (Validation)...\n')
model_catboost_val.fit(X=pool_tr,
                       eval_set=pool_val,
                       early_stopping_rounds=esr)

################################################################################
################################ MODEL CATBOOST ################################
########################### k-Fold Cross-Validation ############################
################################################################################

# 1) k-Fold Cross-Validation Function
################################################################################
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def Model_cv(MODEL, k, X_train, X_test, y, RS, makepred=True, CatPos=None):
	# Create the k folds
	kf=StratifiedKFold(n_splits=k, shuffle=True, random_state=RS)

	# first level train and test
	Level_1_train = pd.DataFrame(np.zeros((X_train.shape[0],1)), columns=['train_yhat'])
	if makepred==True:
		Level_1_test = pd.DataFrame()

	# Main loop for each fold. Initialize counter
	count=0
	for train_index, test_index in kf.split(X_train, Y_train):
		count+=1
		# Define train and test depending in which fold are we
		fold_train= X_train.loc[train_index.tolist(), :]
		fold_test=X_train.loc[test_index.tolist(), :]
		fold_ytrain=y[train_index.tolist()]
		fold_ytest=y[test_index.tolist()]

		# (k-1)-folds model adjusting
		if CatPos:
			# Prepare Pool
			pool_train=Pool(fold_train, fold_ytrain,cat_features=Pos)
			# (k-1)-folds model adjusting
			model_fit=MODEL.fit(X=pool_train)

		else:
			# (k-1)-folds model adjusting
			model_fit=MODEL.fit(fold_train, fold_ytrain)

		# Predict on the free fold to evaluate metric
		# and on train to have an overfitting-free prediction for the next level
		p_fold=MODEL.predict_proba(fold_test)[:,1]
		p_fold_train=MODEL.predict_proba(fold_train)[:,1]

		# Score in the free fold
		score=roc_auc_score(fold_ytest,p_fold)
		score_train=roc_auc_score(fold_ytrain,p_fold_train)
		print(k, '-cv, Fold ', count, '\t --test AUC: ', round(score,4), '\t--train AUC: ', round(score_train,4),sep='')
		# Save in Level_1_train the "free" predictions concatenated
		Level_1_train.loc[test_index.tolist(),'train_yhat'] = p_fold

		# Predict in test to make the k model mean
		# Define name of the prediction (p_"iteration number")
		if makepred==True:
			name = 'p_' + str(count)
			# Predictin to real test
			real_pred = MODEL.predict_proba(X_test)[:,1]
			# Name
			real_pred = pd.DataFrame({name:real_pred}, columns=[name])
			# Add to Level_1_test
			Level_1_test=pd.concat((Level_1_test,real_pred),axis=1)

	# Compute the metric of the total concatenated prediction (and free of overfitting) in train
	score_total=roc_auc_score(y,Level_1_train['train_yhat'])
	print('\n',k, '- cv, TOTAL AUC:', round((score_total)*100,4),'%')

	# mean of the k predictions in test
	if makepred==True:
		Level_1_test['model']=Level_1_test.mean(axis=1)

	# Return train and test sets with predictions and the performance
	if makepred==True:
		return Level_1_train, pd.DataFrame({'test_yhat':Level_1_test['model']}), score_total
	else:
		return score_total

# 2) k-Fold Cross-Validation Implementattion
################################################################################
# Parameters of the CV
RS=2305 # Seed for k-fold partition and model random part
n_folds=5 # Number of folds (depends on the sample size, the proportion of 1's over 0's,...)

# Put in the "iter" list various values around the discovered in the previous step:
# (The number of iterations is altered proportionaly in function of the
# datasets sizes (where has been obtained and where has to be applied))
nrounds_cv=round(model_catboost_val.best_iteration_/(1-TS)*(1-1/n_folds))
iter=[round(nrounds_cv*0.9),nrounds_cv,round(nrounds_cv*1.1)]

print('\nCatboost CV...')
print('########################################################')
scores=[]
for nrounds in iter:

	print('\nn rounds: ',nrounds)

	# Define the model
	model_catboost_cv=CatBoostClassifier()
	model_catboost_cv.set_params(**params)
	model_catboost_cv.set_params(n_estimators=nrounds)
	model_catboost_cv.set_params(verbose=False)

	Pred_train, Pred_test, s = Model_cv(model_catboost_cv,n_folds,X_train,X_test,Y_train,RS,makepred=True,CatPos=Pos)

	# Look if we are in the first test:
	if len(scores)==0:
		max_score=float('-inf')
	else:
		max_score=max(scores)

	# If the score improves, we keep this one:
	if s>=max_score:
		print('BEST')
		Catboost_train=Pred_train.copy()
		Catboost_test=Pred_test.copy()

	# Append score
	scores.append(s)

# The best cross-validated score has been found in:
print('\n###########################################')
print('Catboost optimal rounds: ',iter[scores.index(max(scores))])
print('Catboost optimal GINI: ',round((max(scores)*2-1)*100,4),'%')
print('Catboost optimal AUC: ',round(max(scores)*100,4),'%')
print('###########################################')


# 3) Train a model on whole train with the optimal parameters:
################################################################################

# Adjust optimal CV number of rounds to whole sample size:
nrounds=int(iter[scores.index(max(scores))]/(1-1/n_folds))

# Define the optimal model
model_catboost=CatBoostClassifier(n_estimators=nrounds,
								  random_seed=RS,
								  verbose=100)
model_catboost.set_params(**params)

# To Pool Class (for catboost only)
pool_train=Pool(X_train, Y_train,cat_features=Pos)

# Fit the model
print('\nCatboost Optimal Fit with %d rounds...\n' % nrounds)
model_catboost.fit(X=pool_train)


# 4) Shap Importance for the features of the final model
################################################################################
# Shap methodology:
# "https://medium.com/@gabrieltseng/interpreting-complex-models-with-shap-values-1c187db6ec83"
# Catboost has already SHAP integrated
# Comes as variation with respect to LOG Odds
ShapImportance=model_catboost.get_feature_importance(data=pool_train,
												     type='ShapValues',
													 prettified=True,
													 verbose=False)
ShapValues = ShapImportance.iloc[:, :-1]
ShapValues.columns=list(X_train)

# Picture in Logodds
################################################################################
from shap import summary_plot
num_features=15
summary_plot(ShapValues.values,X_train,max_display=num_features,plot_type='dot')


# Variable Importance Recap
################################################################################
import matplotlib.pyplot as plt
import seaborn as sns

Importance=ShapValues.abs().mean(axis=0)
Importance=pd.DataFrame({'Feature':Importance.index.tolist(),'Importance':Importance}).sort_values(by=['Importance'],ascending=False).reset_index(drop=True)
# Top features:
plt.figure(figsize=(15, 15))
sns.barplot(x="Importance",
			y="Feature",
			data=Importance[:num_features],
			palette=sns.color_palette("Blues_d",
			n_colors=Importance[:num_features].shape[0]))

################################################################################
################################### RESULTS ####################################
################################################################################

# Prediction (All train model)
test['Pred']=model_catboost.predict_proba(X_test)[:,1]
catboost_submission=pd.DataFrame(test[['ID','Pred']])

# Cv predictions
catboost_cv_train=train[['ID']]
catboost_cv_train['catboost_pred']=Catboost_train['train_yhat']
catboost_cv_test=test[['ID']]
catboost_cv_test['catboost_pred']=Catboost_test['test_yhat']

# Outputs to .csv
catboost_submission.to_csv("catboost_submission.csv", index = False)
catboost_cv_train.to_csv("catboost_cv_train.csv", index = False)
catboost_cv_test.to_csv("catboost_cv_test.csv", index = False)
Importance.to_csv("catboost_importance.csv", index = False)


# If we want to save/load an "expensive" to compute
################################################################################
# How to save it:
# from sklearn.externals import joblib
# joblib.dump(model_catboost,'./BBDD Output/model_catboost.sav')

# How to load it and make predictions:
# loaded_model=joblib.load('./BBDD Output/model_catboost.sav')
# loaded_model.predict_proba(X1_test)[:,1]
################################################################################
