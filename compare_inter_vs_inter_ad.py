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
    raise
    if r2_rdfr > base_r2:
        ale_func_extra = rdfr_rgs
        base_r2 = r2_rdfr
        ale_extra_x_test, ale_extra_y_test = x_test, y_test
    summary = {'Predicted':[],'Groundtruth':[]}
    for i in range(predicted_y.shape[0]):
        summary['Predicted'].append(predicted_y[i])
        summary['Groundtruth'].append(y_test[i])
    summary = pd.DataFrame(summary)
    print('RMSE: ',mean_squared_error(y_test, predicted_y,squared=False))
    print('R2: ',r2_score(y_test, predicted_y))
    print('MAE: ',mean_absolute_error(y_test, predicted_y))
    print('Explained_variance_score: ',explained_variance_score(y_test, predicted_y))
    print('MAPE: ',mean_absolute_percentage_error(y_test, predicted_y))
    print('-'*30)
    rmse_rf.append(mean_squared_error(y_test, predicted_y,squared=False))
    r2_rf.append(r2_score(y_test, predicted_y))
    mae_rf.append(mean_absolute_error(y_test, predicted_y))
    evs_rf.append(explained_variance_score(y_test, predicted_y))
    mape_rf.append(mean_absolute_percentage_error(y_test, predicted_y))

    if (max(r2_rf) > args.r2_threshold_verify and t > args.min_iters_verify):
        print('Feature Importance')
        print('*'*10)
        
        # Gradient Boosting FI
        print('Gradient Boosting FI')
        r = permutation_importance(ale_func_extra, ale_extra_x_test, ale_extra_y_test,
                                    n_repeats=100,
                                    random_state=0)

        important_ind = []
        for i in r.importances_mean.argsort()[::-1]:
            if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
                important_ind.append(i)
                print(f"{ale_extra_x_test.columns[i]:<8}: "
                      f"{r.importances_mean[i]:.3f}"
                     f" +/- {r.importances_std[i]:.3f}")
        importances = pd.Series(r.importances_mean[important_ind], index=ale_extra_x_test.columns[important_ind])
        fig, ax = plt.subplots(figsize=(10,15))
        importances.plot.bar(yerr=r.importances_std[important_ind], ax=ax)
        ax.set_title("Feature importances\nusing permutation\non the Random Forest Model")
        ax.set_ylabel("Mean accuracy decrease")
        ax.tick_params(axis='x', rotation=45)
        fig.tight_layout()
        fig.savefig(f'image/interpret/permutation/random_forest_permute_extra{model}.png')
        plt.show()
        print('*'*10)
        break
    print('*'*100)

global_report = pd.DataFrame([[mean(rmse_gb),max(rmse_gb),min(rmse_gb),np.var(rmse_gb),
                               mean(r2_gb),max(r2_gb),min(r2_gb),np.var(r2_gb),
                               mean(mae_gb),max(mae_gb),min(mae_gb),np.var(mae_gb),
                               mean(evs_gb),max(evs_gb),min(evs_gb),np.var(evs_gb),
                               mean(mape_gb),max(mape_gb),min(mape_gb),np.var(mape_gb)], 
                              [mean(rmse_lr),max(rmse_lr),min(rmse_lr),np.var(rmse_lr),
                               mean(r2_lr),max(r2_lr),min(r2_lr),np.var(r2_lr),
                               mean(mae_lr),max(mae_lr),min(mae_lr),np.var(mae_lr),
                               mean(evs_lr),max(evs_lr),min(evs_lr),np.var(evs_lr),
                               mean(mape_lr),max(mape_lr),min(mape_lr),np.var(mape_lr)],
                              [mean(rmse_mlp),max(rmse_mlp),min(rmse_mlp),np.var(rmse_mlp),
                               mean(r2_mlp),max(r2_mlp),min(r2_mlp),np.var(r2_mlp),
                               mean(mae_mlp),max(mae_mlp),min(mae_mlp),np.var(mae_mlp),
                               mean(evs_mlp),max(evs_mlp),min(evs_mlp),np.var(evs_mlp),
                               mean(mape_mlp),max(mape_mlp),min(mape_mlp),np.var(mape_mlp)],
                              [mean(rmse_rf),max(rmse_rf),min(rmse_rf),np.var(rmse_rf),
                               mean(r2_rf),max(r2_rf),min(r2_rf),np.var(r2_rf),
                               mean(mae_rf),max(mae_rf),min(mae_rf),np.var(mae_rf),
                               mean(evs_rf),max(evs_rf),min(evs_rf),np.var(evs_rf),
                               mean(mape_rf),max(mape_rf),min(mape_rf),np.var(mape_rf)]], 
                                columns=[   'RMSE_MEAN','RMSE_MAX','RMSE_MIN','RMSE_VAR',
                                            'R2_MEAN','R2_MAX','R2_MIN','R2_VAR',
                                            'MAE_MEAN','MAE_MAX','MAE_MIN','MAE_VAR',
                                            'EVS_MEAN','EVS_MAX','EVS_MIN','EVS_VAR',
                                            'MAPE_MEAN','MAPE_MAX','MAPE_MIN','MAPE_VAR'], 
                                index=['Gradient Boosting', 'Linear Regression', 'MLP', 'Random Forest'])

with open(f'rmse_rf_verify{model}.npy', 'wb') as f:
    np.save(f, rmse_rf)
with open(f'r2_rf_verify{model}.npy', 'wb') as f:
    np.save(f, r2_rf)
with open(f'mae_rf_verify{model}.npy', 'wb') as f:
    np.save(f, mae_rf)
with open(f'evs_rf_verify{model}.npy', 'wb') as f:
    np.save(f, evs_rf)
with open(f'mape_rf_verify{model}.npy', 'wb') as f:
    np.save(f, mape_rf)

(global_report.T).to_csv(f'result_summary_random_forest_verification{model}.csv')
summary.to_csv(f'predict_summary_random_forest_verification{model}.csv')

print('Finish')