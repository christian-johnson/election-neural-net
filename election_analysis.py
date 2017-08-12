print "Loading required packages..."
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import matplotlib.ticker as mtick

import hashlib

import load_data as ld
import plotter
tf.logging.set_verbosity(tf.logging.INFO)

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import ElasticNet
from sklearn.svm import LinearSVR
from sklearn.preprocessing import PolynomialFeatures, RobustScaler, StandardScaler, MinMaxScaler
print "Successfully loaded packages!"

#TensorFlow functions
#Define input function to DNNRegressor
def get_input_fn(data_set, num_epochs=None, shuffle=True):
  return tf.estimator.inputs.pandas_input_fn(
      x=pd.DataFrame(data_set[FEATURES]),
      y=pd.Series(data_set[LABELS]),
      num_epochs=num_epochs,
      shuffle=shuffle)

def prep_neural_net(data):

    #Prepare the neural net Feature Columns
    COLUMNS = [
        'GOP2016',
        'GOP2012',
        
        'Turnout2012',
        'Turnout2016',
        
        'Population2016',
        'Population2009',
    
        'White2015',
        'Black2015',
        'Hispanic2015',
        'Asian2015',
        'Indian2015',
        'Youth2015',
        'White2009',
        'Black2009',
        'Hispanic2009',
        'Asian2009',
        'Indian2009',
        'Youth2009',
    
        'Unemployment2016',
        'Unemployment2007',
        'Unemployment2004',
    
        'MedianAge',
        'Bachelors',
        'CommuteTime',
        'FoodStamps',
        'Homeownership',
        'IncomeInequality',
    
        'Businesses2016',
        'Businesses2009',
    
        'RentBurdened2015',
        'RentBurdened2010',
    
        'EvangelicalProtestant',
        'MainlineProtestant',
        'BlackProtestant',
        'Catholic',
        'Jewish',
        'Muslim',
        'Mormon'
    ]

    FEATURES = [
        'GOP2012',
        
        'Turnout2012',
        
        'Population2016',
        'Population2009',
    
        'White2015',
        'Black2015',
        'Hispanic2015',
        'Asian2015',
        'Indian2015',
        'Youth2015',
        'White2009',
        'Black2009',
        'Hispanic2009',
        'Asian2009',
        'Indian2009',
        'Youth2009',
    
        'Unemployment2016',
        'Unemployment2007',
        'Unemployment2004',
    
        'MedianAge',
        'Bachelors',
        'CommuteTime',
        'FoodStamps',
        'Homeownership',
        'IncomeInequality',
    
        'Businesses2016',
        'Businesses2009',
    
        'RentBurdened2015',
        'RentBurdened2010',
    
        'EvangelicalProtestant',
        'MainlineProtestant',
        'BlackProtestant',
        'Catholic',
        'Jewish',
        'Muslim',
        'Mormon'
    ]

    LABELS = 'GOP2016'

    # Feature columns
    feature_cols = [tf.feature_column.numeric_column(k) for k in FEATURES]

    # Initialize fully connected DNN
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols,
          optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
             hidden_units=[100, 100, 100])
             
    return regressor, FEATURES, LABELS

def run_neural_net(regressor, test_state, FEATURES, LABELS):
    #Split up data into training set, test set
    states_codes={'AL':'01','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','FL':'12','GA':'13','HI':'15','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}
    test_indices = []

    for state in [test_state]:
        test_indices.append(np.where((data.index>int(states_codes[state])*1000) & (data.index<(int(states_codes[state])+1)*1000))[0])

    test_data = data.loc[data.index[np.concatenate(test_indices)],:]
    training_data = data.loc[data.index[np.where([data.index[k] not in data.index[np.concatenate(test_indices)] for k in range(len(data.index))])],:]
    
    # Train neural net
    regressor.fit(input_fn=get_input_fn(training_data), steps=3000)

    #Predict outputs
    results = regressor.predict(input_fn=get_input_fn(test_data, num_epochs=1, shuffle=False))
    predictions = list(itertools.islice(results, 500))

    q = np.zeros((len(test_data)))
    w = np.zeros((len(test_data)))
    for i in range(len(test_data)):
        q[i] = test_data[LABELS].values[i]
        w[i] = predictions[i]
        
    #Return residuals
    return pd.Series(q-w, index=test_data.index)

#regressor, FEATURES, LABELS = prep_neural_net(data)
#results = run_neural_net(regressor, test_state, FEATURES, LABELS)


#Scikit-Learn functions

def test_set_check(fips, i, hash):
    return float(int(hash(str(fips)).hexdigest(), 16)%256.0)/256.0>float(i)/20.0 and float(int(hash(np.int64(fips)).hexdigest(), 16)%256.0)/256.0<float(i+1)/20.0
    
#split data by an index i- compute the hash of the FIPS code and check whether it is between i/20 and (i+1)/20
def split_test_train(data, i, num_test_sets=20, hash=hashlib.md5):
    states_codes={'AL':'01','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','FL':'12','GA':'13','HI':'15','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}
    hashes = map(float,map((lambda id_: int(hash(str(id_)).hexdigest(), 16)%256), data.index.map(int)))
    hashes = [value/256.0 for value in hashes]
    in_test_set = map((lambda the_hash: the_hash>float(i)/num_test_sets and the_hash<float(i+1.0)/num_test_sets), hashes)
    in_training_set = [not bool for bool in in_test_set]
    return data.loc[in_training_set], data.loc[in_test_set]
    
    """
    for state in [test_state]:
        test_indices.append(np.where((data.index>int(states_codes[state])*1000) & (data.index<(int(states_codes[state])+1)*1000))[0])
    test_data = data.loc[data.index[np.concatenate(test_indices)],:]
    training_data = data.loc[data.index[np.where([data.index[k] not in data.index[np.concatenate(test_indices)] for k in range(len(data.index))])],:]
    
    return test_data, training_data
    """

#Class to add combined features to the data (i.e. features that are combinations of existing features)
#Can be incorporated nicely into a data-preparation pipeline with this format
#Hyperparameters: demographics_changes controls whether changes from 2009-2015 in ethnic/racial makeup are included
#economic_changes controls whether changes in unemployment, businesses per person, population, and rent burdened are included
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, demographics_changes=False, economic_changes=False):
        self.demographics_changes = demographics_changes
        self.economic_changes = economic_changes
        
        self.white2015_index = 10
        self.white2009_index = 16
        self.black_2015_index = 11
        self.black_2009_index = 17
        self.hispanic_2015_index = 12
        self.hispanic_2009_index = 18
        self.asian_2015_index = 13
        self.asian_2009_index = 19
        self.indian_2015_index = 14
        self.indian_2009_index = 20
        self.youth_2015_index = 15
        self.youth_2009_index = 21
        
        self.unemployment_2007_index = 24
        self.unemployment_2011_index = 23
        self.unemployment_2016_index = 22
        self.businesses_2009_index = 32
        self.businesses_2016_index = 31
        self.population_2016_index = 7
        self.population_2009_index = 9
        self.rent_burdened_2010_index = 34
        self.rent_burdened_2015_index = 33
        
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X, y=None):
        #Changes in demographics
        if self.demographics_changes:
            white_change = X[:,self.white2015_index]-X[:,self.white2009_index]
            black_change = X[:,self.black_2015_index]-X[:,self.black_2009_index]
            hispanic_change = X[:,self.hispanic_2015_index]-X[:,self.hispanic_2009_index]
            asian_change = X[:,self.asian_2015_index]-X[:,self.asian_2009_index]
            indian_change = X[:,self.indian_2015_index]-X[:,self.indian_2009_index]
            youth_change = X[:,self.youth_2015_index]-X[:,self.youth_2009_index]
            
            #if we want the changes, we don't want the old data itself
            X = np.c_[X, white_change, black_change, hispanic_change, asian_change, indian_change, youth_change]
        
        #Changes in economic indicators
        if self.economic_changes:
            unemployment_2007_change = X[:,self.unemployment_2016_index]-X[:,self.unemployment_2007_index]
            unemployment_2011_change = X[:,self.unemployment_2016_index]-X[:,self.unemployment_2011_index]
            businesses_change = X[:,self.businesses_2016_index]-X[:,self.businesses_2009_index]
            population_change = X[:,self.population_2016_index]-X[:,self.population_2009_index]
            rent_burdened_change = X[:,self.rent_burdened_2015_index]-X[:,self.rent_burdened_2010_index]
            X = np.c_[X, unemployment_2007_change, unemployment_2011_change, businesses_change, population_change, rent_burdened_change]

        #Delete old data- don't want it cluttering the algorithm
        X= np.delete(X, [self.white2009_index, self.black_2009_index, self.hispanic_2009_index, self.asian_2009_index, self.indian_2009_index, self.youth_2009_index, self.unemployment_2007_index, self.unemployment_2011_index, self.businesses_2009_index, self.population_2009_index, self.rent_burdened_2010_index], axis=1)
        return X


        
    
#Class to translate Pandas DataFrames into Numpy Arrays
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
        

states = ['AL','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','NC','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','ND','OH','OK','OR','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY']
#corr_matrix = data.corr()
#print corr_matrix['GOP2016'].sort_values(ascending=False)


attribs = [
        'GOP2012',
        'Turnout2012',
        'Turnout2016',
        'VotesDem2016',
        'VotesGOP2016',
        'VotesDem2012',
        'VotesGOP2012',
        'Population2016',
        'Population2012',
        'Population2009',
        'White2015',
        'Black2015',
        'Hispanic2015',
        'Asian2015',
        'Indian2015',
        'Youth2015',
        'White2009',
        'Black2009',
        'Hispanic2009',
        'Asian2009',
        'Indian2009',
        'Youth2009',
        'Unemployment2016',
        'Unemployment2011',
        'Unemployment2007',
        'MedianAge',
        'Bachelors',
        'CommuteTime',
        'FoodStamps',
        'Homeownership',
        'IncomeInequality',
        'Businesses2016',
        'Businesses2009',
        'RentBurdened2015',
        'RentBurdened2010',
        'EvangelicalProtestant',
        'MainlineProtestant',
        'BlackProtestant',
        'Catholic',
        'Jewish',
        'Muslim',
        'Mormon',
        'ClimateChange',
        'Corn',
        'Cotton',
        'Soybeans',
        'WinterWheat'
    ]
        
        


#Inject extra votes- would they be detected?
def add_fake_votes(data, num_votes):
    #Miami-Dade county
    data.loc[12086,'GOP2016'] = (data.loc[12086,'VotesGOP2016']+num_votes)/(data.loc[12086,'VotesGOP2016']+data.loc[12086,'VotesDem2016'])
    #Wayne county (Detroit)
    data.loc[26163,'GOP2016'] = (data.loc[26163,'VotesGOP2016']+num_votes)/(data.loc[26163,'VotesGOP2016']+data.loc[26163,'VotesDem2016'])
    #Cuyahoga county (Cleveland)
    data.loc[39035,'GOP2016'] = (data.loc[39035,'VotesGOP2016']+num_votes)/(data.loc[39035,'VotesGOP2016']+data.loc[39035,'VotesDem2016'])
    return data

#Need to try several models:
#Neural Net (MLPRegressor)
#Random Forest/Decision Trees
#Linear Model/Polynomial Model



#Figuring out which hyperparameters are best
def run_hyperparameter_search():
    #Use GridSearchCV to figure out optimal hyperparameters
    #Hide swing states (FL, NC, PA, MI, WI, OH), train on all others
    swing_states= ['FL','NC','PA','WI','MI','OH']
    test_data, training_data = split_test_train(swing_states)

    #Split into labels, features
    training_features = training_data.drop('GOP2016', axis=1)
    training_labels = training_data['GOP2016'].copy()
    test_features = test_data.drop('GOP2016', axis=1)
    test_labels = test_data['GOP2016'].copy()
    training_data_prepped = data_pipeline.fit_transform(training_features)
    test_data_prepped = data_pipeline.transform(test_features)
    
    lin_reg = LinearRegression()
    mlp_reg = MLPRegressor()
    forest_reg = RandomForestRegressor()

    param_grid = [
        {'n_estimators':[50, 100, 150], 'max_features':[8, 12, 20, 30]}
    ]
    grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(training_data_prepped, training_labels)
    print "Random Forest best estimator:" + str(grid_search.best_estimator_)

    param_grid = [
        {'hidden_layer_sizes':[[100, 100], [100, 100, 100]],
        'activation':['tanh','relu'],
        'alpha':[1e-8, 1e-6, 1e-5],
        'learning_rate':['constant','adaptive', 'invscaling'],
        'max_iter':[100, 1000, 10000]}
    ]
    grid_search = GridSearchCV(mlp_reg, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(training_data_prepped, training_labels)
    print "MLP best estimator:" + str(grid_search.best_estimator_)
    
    #Results:
    #Random Forest: max_features=30, n_estimators=150
    #MLP: activation='tanh', alpha=1e-06, batch_size='auto', beta_1=0.9,
    #   beta_2=0.999, early_stopping=False, epsilon=1e-08,
    #   hidden_layer_sizes=[100, 100, 100], learning_rate='constant',
    #   learning_rate_init=0.001, max_iter=100, momentum=0.9,
    #   nesterovs_momentum=True, power_t=0.5, random_state=None,
    #   shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
    #   verbose=False, warm_start=False)


def calculate_nationwide_resids(data, scheme):
    results = pd.Series()
    for i in range(20):
        print "Test group " + str(i)
        #Prep data
        if scheme == 'Poly':
            data_pipeline = Pipeline([
                ('selector', DataFrameSelector(attribs)),
                ('attribs', CombinedAttributesAdder(demographics_changes=True, economic_changes=True)),
                ('poly_features', PolynomialFeatures(degree=2, include_bias=False)),
                ('std_scaler', StandardScaler()),
            ])#
        else:
            data_pipeline = Pipeline([
                ('selector', DataFrameSelector(attribs)),
                ('attribs', CombinedAttributesAdder(demographics_changes=True, economic_changes=True)),
                ('std_scaler', StandardScaler()),
            ])#
            
        #Split into training/test data
        training_data, test_data = split_test_train(data, i)

        #Split into labels, features
        training_features = training_data.drop('GOP2016', axis=1)
        training_labels = training_data['GOP2016'].copy()
        test_features = test_data.drop('GOP2016', axis=1)
        test_labels = test_data['GOP2016'].copy()
    
        #Prepare data
        training_data_prepped = data_pipeline.fit_transform(training_features)
        test_data_prepped = data_pipeline.transform(test_features)

        if scheme == 'Linear' or scheme == 'Poly':
            #Train
            elastic_reg = ElasticNet(alpha=0.001, l1_ratio=0.1)
            elastic_reg.fit(training_data_prepped, training_labels)
            
            #Predict
            predictions = elastic_reg.predict(test_data_prepped)
            
            
        if scheme == 'MLP':
            #Train
            mlp_reg = MLPRegressor(hidden_layer_sizes=[250, 250, 250],max_iter=1000, early_stopping=True, alpha=0.2)
            mlp_reg.fit(training_data_prepped, training_labels)
            #Predict
            predictions = mlp_reg.predict(test_data_prepped) 
            
            
        if scheme == 'RandomForest':
        
            #Train
            forest_reg = RandomForestRegressor(max_features=30, n_estimators=200)            
            forest_reg.fit(training_data_prepped, training_labels)
            
            #Predict
            predictions = forest_reg.predict(test_data_prepped) #Predictions for test state
        
        #scores = cross_val_score(forest_reg, training_data_prepped, training_labels, scoring='neg_mean_squared_error', cv=50)
        #mlp_scores = np.sqrt(-scores)
        #print "mean score on cross validation = "+ str(mlp_scores.mean())
        #mlp_rmse = mean_squared_error(training_labels, forest_reg.predict(training_data_prepped))
        #print "score on training data: " + str(np.sqrt(mlp_rmse))
        
        residuals = test_labels-predictions
        #Concatenate to all results
        results = pd.concat([results,residuals])
        
    return results
    
    
def main():
    #Prepare data
    data = ld.load_data()
    #plotter.national_plot(data['Corn']+data['Cotton']+data['Soybeans']+data['WinterWheat'], data_min=0.0, data_max=1.0,vmin=0.0, vmax=100.0, plt_title='crops_planted',colorbar_label='Land Devoted to Growing Crops', use_cmap = cm.Greens, AK_value = False)
    #plotter.national_plot(data['ClimateChange'], data_min=0.3, data_max=0.65, vmin=30.0, vmax=65.0, plt_title='climate_change', colorbar_label='Belief in Human-Caused Climate Change', use_cmap = cm.plasma, AK_value = False)
    plotter.national_plot(data['Unemployment2016']-data['Unemployment2007'], data_min=-0.08, data_max=0.08,vmin=-8.0, vmax=8.0, colorbar_label='Change in Unemployment Rate 2007-2016', plt_title='unemployment_change2007_2016', use_cmap = cm.RdYlGn_r, AK_value = False)
    data = ld.clean_data(data)
    #data = add_fake_votes(data, 10000)
    linear_results = calculate_nationwide_resids(data,scheme='Linear')
    mlp_results = calculate_nationwide_resids(data,scheme='MLP')
    randomforest_results = calculate_nationwide_resids(data,scheme='RandomForest')
    
    ensemble_data = pd.concat([linear_results, mlp_results, randomforest_results], axis=1)
    ensemble_mean = ensemble_data.mean(axis=1)
    print "RMS = " + str(100.0*np.sqrt(np.mean(ensemble_mean.values**2))) + "%"
    plotter.national_plot(ensemble_mean, data_min=-0.2, data_max=0.2,vmin=-20.0, vmax=20.0, plt_title='model_comparison/vote_residmap_ensemble',colorbar_label='GOP Vote Share Residual', use_cmap = cm.seismic, AK_value = False)

if __name__ == '__main__':
    main()
    

"""

file = open('models/turnout_model.pk1','rb')
turnout_resids = pickle.load(file)
file.close()
file = open('models/voting_model.pk1','rb')
resids = pickle.load(file)
file.close()
file = open('models/injected_voting_model.pk1','rb')
inj_resids = pickle.load(file)
file.close()

fl_data = data['GOP2016']
fl_model = data['GOP2016']-resids

#plotter.national_plot(data['GOP2016']-data['GOP2012'], data_min=-0.2, data_max=0.2,vmin=-20.0, vmax=20.0, plt_title='injected_vote_change',colorbar_label='GOP Vote Change', use_cmap = cm.seismic, AK_value = False)

plotter.state_plot_data_model(fl_data, fl_model, 'FL', data_min=0.0, data_max=1.0, vmin=0.0, vmax=100.0, plt_title='FL_data_model', cb2_label='GOP Vote Share', use_cmap = cm.seismic)

#plotter.national_plot(data['GOP2016'], data_min=0.0, data_max=1.0,vmin=0.0, vmax=100.0, plt_title='National_vote_fraction',colorbar_label='GOP Vote Share', use_cmap = cm.seismic, AK_value = 0.5288700180057424)

#plotter.national_plot(resids, data_min=-0.2, data_max=0.2,vmin=-20.0, vmax=20.0, plt_title='vote_residmap',colorbar_label='GOP Vote Share Residual', use_cmap = cm.seismic, AK_value = False)

#plotter.national_plot(turnout_resids, data_min=-0.1, data_max=0.1,vmin=-10.0, vmax=10.0, plt_title='turnout_residmap',colorbar_label='Turnout Residual', use_cmap = cm.PuOr, AK_value = False)

#plotter.national_plot(inj_resids, data_min=-0.2, data_max=0.2,vmin=-20.0, vmax=20.0, plt_title='injected_vote_residmap',colorbar_label='GOP Vote Share Residual', use_cmap = cm.seismic, AK_value = False)

#plotter.national_plot(data['Turnout2016'], data_min=0.2, data_max=0.8,vmin=20.0, vmax=80.0, plt_title='National_turnout',colorbar_label='Turnout', use_cmap = cm.plasma, AK_value = False)

#plotter.national_plot(turnout_resids, data_min=-0.1, data_max=0.1,vmin=-10.0, vmax=10.0, plt_title='turnout_residmap',colorbar_label='Turnout Residual [\%]', use_cmap = cm.PuOr, AK_value = False)


new_data = pd.concat([pd.DataFrame(resids,columns=['Resid']),pd.DataFrame(turnout_resids,columns=['TurnoutResids']),data.loc[:,'GOP2016'],data.loc[:,'Population2016']],axis=1)
fig = plt.figure(figsize=[10,8])
plt.scatter(100.0*new_data.loc[:,'Resid'],100.0*new_data.loc[:,'TurnoutResids'],s=new_data.loc[:,'Population2016']*150.0,c=cm.seismic(new_data.loc[:,'GOP2016']))
ax = plt.gca()
ax.set_facecolor('#202020')
plt.axhline(0.0, linewidth=0.5, color='white',linestyle='--')

fmtr = mtick.StrMethodFormatter('{x:,g}\%')
ax.xaxis.set_major_formatter(fmtr)
ax.yaxis.set_major_formatter(fmtr)

plt.ylabel('Turnout Residual')
plt.xlabel('GOP Vote Residual')
#plt.savefig('plots/residual_vs_vote.pdf',bbox_inches='tight')
plt.show()
"""