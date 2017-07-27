print "Loading required packages..."
import numpy as np
import pandas as pd
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import matplotlib.ticker as mtick

import load_data as ld
import plotter
tf.logging.set_verbosity(tf.logging.INFO)



print "Successfully loaded packages!"


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

states = ['AL','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','NC','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY','ND','OH','OK','OR','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ','NM','NY']
data = ld.load_data()

#Inject extra votes- would they be detected?
#Miami-Dade county
data.loc[12086,'GOP2016'] = (data.loc[12086,'VotesGOP2016']+10000.0)/(data.loc[12086,'VotesGOP2016']+data.loc[12086,'VotesDem2016'])
#Detroit
data.loc[26163,'GOP2016'] = (data.loc[26163,'VotesGOP2016']+10000.0)/(data.loc[26163,'VotesGOP2016']+data.loc[26163,'VotesDem2016'])
#Cleveland
data.loc[39035,'GOP2016'] = (data.loc[39035,'VotesGOP2016']+10000.0)/(data.loc[39035,'VotesGOP2016']+data.loc[39035,'VotesDem2016'])
"""
print data
resids = pd.Series()
#Here, use swing states as the test set and the rest of the country as training set
for test_state in states:
    regressor, FEATURES, LABELS = prep_neural_net(data)
    print test_state
    results = run_neural_net(regressor, test_state, FEATURES, LABELS)
    resids = pd.concat([resids,results])
file = open('data_turnout.pk1','wb')
pickle.dump(resids, file)
file.close()
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


"""
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