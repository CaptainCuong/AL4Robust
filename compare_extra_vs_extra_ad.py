import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.pylab as pylab
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import lightgbm as lgb
from argparse import ArgumentParser
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error,explained_variance_score,\
                            mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
import itertools
from sklearn.inspection import permutation_importance
from utils.PyALE import ale
import statsmodels.api as sm
import random
from statistics import mean 

parser = ArgumentParser()
parser.add_argument('--verify_random_forest_file', type=str, choices=['verify_data_bert.csv','verify_data_roberta.csv'],
                                    default='verify_data_roberta.csv', help='Folder in which to save model and logs')
parser.add_argument('--sample_data_file', type=str, choices=['data_roberta.csv','data_bert.csv'], 
                                          help='Sampled data file', default='data_roberta.csv')

parser.add_argument('--r2_threshold_verify', type=float, default=-2.0, help='Folder in which to save model and logs')
parser.add_argument('--min_iters_verify', type=int, default=200, help='Folder in which to save model and logs')

args = parser.parse_args()
if not (args.sample_data_file == 'data_bert.csv' and args.verify_random_forest_file == 'verify_data_bert.csv'
    or  args.sample_data_file == 'data_roberta.csv' and args.verify_random_forest_file == 'verify_data_roberta.csv'):
    raise Exception('Verification Data and Sampled Data do not match.')

pylab.rcParams['font.size'] = 30
file_test = pd.read_csv(args.verify_random_forest_file,sep=',')
file_test = file_test.drop(columns='ASR_BERT')
file_test = file_test[(file_test!='Nan').all(1)]
file_test = file_test.astype({'ASR_DeepWordBug': 'float64','ASR_PWWS': 'float64','ASR_TextFooler': 'float64'})
file_test['ASR']=(file_test['ASR_TextFooler']+file_test['ASR_PWWS']+file_test['ASR_DeepWordBug'])/3
file_test['Fisher ratio'] = file_test['Fisher ratio'].apply(lambda x:1/x)
file_test.rename(columns = {'Fisher ratio':'FR', 'CalHara Index':'CHI',
                       'DaBou Index':'DBI', 'Pearson Med':'PMS',
                       'Mean distance':'MD', 'Minimum number of tokens': 'Min # tokens',
                       'Maximum number of tokens': 'Max # tokens', 'Number of cluster': '# clusters', 'Kurtosis': 'KTS',
                       'Average number of tokens': 'Avg. # tokens', 'Number of unique tokens': '# unique tokens',
                       'Misclassification rate': 'MR', 'Number of classes': '# classes'}, inplace = True)

file_test.drop(columns=['ASR_TextFooler','ASR_PWWS','ASR_DeepWordBug'],inplace=True)

# args.sample_data_file: ['data_roberta.csv','data_bert.csv']
file = pd.read_csv(args.sample_data_file,sep=',')
model = '_bert' if args.sample_data_file == 'data_bert.csv' else '_distil_roberta'
file = file[file.notnull().all(1)].drop(columns='ASR_BERT')
file = file[(file!='Nan').all(1)]
file = file.astype({'ASR_DeepWordBug': 'float64','ASR_PWWS': 'float64','ASR_TextFooler': 'float64'})
file['ASR']=(file['ASR_TextFooler']+file['ASR_PWWS']+file['ASR_DeepWordBug'])/3
file['Fisher ratio'] = file['Fisher ratio'].apply(lambda x:1/x)
file.rename(columns = {'Fisher ratio':'FR', 'CalHara Index':'CHI',
                       'DaBou Index':'DBI', 'Pearson Med':'PMS',
                       'Mean distance':'MD', 'Minimum number of tokens': 'Min # tokens',
                       'Maximum number of tokens': 'Max # tokens', 'Number of cluster': '# clusters', 'Kurtosis': 'KTS',
                       'Average number of tokens': 'Avg. # tokens', 'Number of unique tokens': '# unique tokens',
                       'Misclassification rate': 'MR', 'Number of classes': '# classes'}, inplace = True)
file.drop(columns=['ASR_TextFooler','ASR_PWWS','ASR_DeepWordBug'],inplace=True)

file_test = file_test.drop_duplicates(subset=['Index'])
ind_test = pd.Series(list(set(file_test['Index']).intersection(set(file['Index']))))

data_train = file[~file['Index'].isin(ind_test)]
data_test_adv = file_test[file_test['Index'].isin(ind_test)]
data_test_norm = file[file['Index'].isin(ind_test)]

data_train.drop(columns=['Index'],inplace=True)
data_test_adv.drop(columns=['Index'],inplace=True)
data_test_norm.drop(columns=['Index'],inplace=True)

print('*-'*100)
print('Interpolation test')

############### Random Forest Verification Experiment ###############
rmse_gb,rmse_mlp,rmse_lr,rmse_rf = [],[],[],[]
r2_gb,r2_mlp,r2_lr,r2_rf = [],[],[],[]
mae_gb,mae_mlp,mae_lr,mae_rf = [],[],[],[]
evs_gb,evs_mlp,evs_lr,evs_rf = [],[],[],[]
mape_gb,mape_mlp,mape_lr,mape_rf = [],[],[],[]
ale_func_extra = None
base_r2 = -1000
ale_extra_x_test, ale_extra_y_test = None, None
for t in itertools.count():
    x_train, y_train = data_train.drop(columns=['Dataset','ASR']), np.array(data_train['ASR'])
    x_test_adv, y_test_adv = data_test_adv.drop(columns=['Dataset','ASR']), np.array(data_test_adv['ASR'])
    x_test_norm, y_test_norm = data_test_norm.drop(columns=['Dataset','ASR']), np.array(data_test_norm['ASR'])
    
    # Random Forest
    rdfr_rgs = RandomForestRegressor(max_depth=20, random_state=0).fit(x_train,y_train)
    predicted_y_adv = rdfr_rgs.predict(x_test_adv)
    r2_adv = r2_score(y_test_adv, predicted_y_adv)
    predicted_y_norm = rdfr_rgs.predict(x_test_norm)
    r2_norm = r2_score(y_test_norm, predicted_y_norm)
    print(np.mean(predicted_y_adv))
    print(np.mean(predicted_y_norm))
    # print(predicted_y_adv)
    # print(predicted_y_norm)