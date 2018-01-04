from election_analysis import *


#Function to find the slope of a given feature as a function of county
def find_slope(data, feature):
    data_pipeline = make_pipeline('Linear')
    #Split into training/test data (Train on entire country)
    training_data, test_data = split_test_train(data, 1, num_test_sets = 1)
    training_features = training_data.drop('GOP2016', axis=1)
    training_labels = training_data['GOP2016'].copy()
    min_feature_value = np.min(training_features[feature].values)
    max_feature_value = np.max(training_features[feature].values)
    #Prepare data
    training_data_prepped = data_pipeline.fit_transform(training_features)
    #Make two test sets- one with the feature of interest decreased by 1% and one with it increased by 1%
    #Then predict the outcome for each case, and subtract the two and divide by 2.0 to get the slope
    test_data_low = training_data.copy()
    test_data_low.loc[:,feature] = test_data_low.loc[:,feature]-0.01
    test_data_low_prepped = data_pipeline.transform(test_data_low.drop('GOP2016', axis=1))

    test_data_high = training_data.copy()
    test_data_high.loc[:,feature] = test_data_high.loc[:,feature]+0.01
    test_data_high_prepped = data_pipeline.transform(test_data_high.drop('GOP2016', axis=1))
    #Train
    mlp_reg = MLPRegressor(hidden_layer_sizes=[250, 250, 250],max_iter=40000, early_stopping=True, alpha=0.2)
    mlp_reg.fit(training_data_prepped, training_labels)
    #Predict

    prediction_low = pd.DataFrame(mlp_reg.predict(test_data_low_prepped), index=training_data.index)
    prediction_high = pd.DataFrame(mlp_reg.predict(test_data_high_prepped), index=training_data.index)
    slopes = (prediction_high-prediction_low)/2.0
    return slopes


def avg_county_plotter(results_list):
    fig = plt.figure(figsize=[11,8])
    ax = plt.axes([0.1, 0.1, 0.64, 0.8])
    colors=['#1f77b4', '#ff7f0e', '#2ca02c',
                  '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                  '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                  '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    for rank, result in enumerate(results_list):
        ax.plot(100.*(result['Data']-0.46582717), color=colors[rank], linewidth=2.0)
        y_pos = 100.*(result['Data'][-1]-0.46582717)
        plt.text(107.0, y_pos, result['Name'], fontsize=14, color=colors[rank])

    ax = plt.gca()
    ax.set_facecolor('#202020')
    plt.axhline(0.0, linewidth=0.5, color='white',linestyle='--')

    fmtr = mtick.StrMethodFormatter('{x:,g}%')
    plt.xticks([0, 100], ['Lower', 'Higher'])
    ax.yaxis.set_major_formatter(fmtr)
    plt.ylim([-7.5, 7.5])
    plt.xlabel('Relative Value')
    plt.ylabel('Change in Trump Support')
    plt.savefig('plots/2017-9-13-election-factors/avg_county_variations_linear.pdf',bbox_inches='tight')
    plt.show()



#Generate an average county, train ML on the entire country and use the trained model to predict the average county
#As one value varies from its minimum value to its maximum value
def calculate_avg_county_range(data, scheme, feature):
    data_pipeline = make_pipeline(scheme)
    #Split into training/test data
    training_data, test_data = split_test_train(data, 1, num_test_sets = 1)
    training_features = training_data.drop('GOP2016', axis=1)
    training_labels = training_data['GOP2016'].copy()
    min_feature_value = np.min(training_features[feature].values)
    max_feature_value = np.max(training_features[feature].values)
    #Prepare data
    training_data_prepped = data_pipeline.fit_transform(training_features)

    #Make an average county weighted by population
    avg_data = (np.dot(data.values.transpose(), data['Population2016'])/np.sum(data['Population2016']))
    avg_test_data = pd.DataFrame(avg_data,index=training_data.columns).transpose()
    avg_test_data = avg_test_data.append([avg_test_data.loc[0]]*99)
    avg_test_data.loc[:,feature] = np.linspace(min_feature_value, max_feature_value, 100)
    avg_test_data_prepped = data_pipeline.transform(avg_test_data.drop('GOP2016', axis=1))
    if scheme == 'Linear':
        #Train
        elastic_reg = get_elastic_model(training_data_prepped, training_labels)
        prediction = elastic_reg.predict(avg_test_data_prepped)

    if scheme == 'MLP':
        #Train
        mlp_reg = MLPRegressor(hidden_layer_sizes=[100, 100, 100],max_iter=10000, early_stopping=True, alpha=0.01)
        mlp_reg.fit(training_data_prepped, training_labels)
        #Predict
        prediction = mlp_reg.predict(avg_test_data_prepped)
    if scheme == 'RandomForest':
        #Train
        forest_reg = RandomForestRegressor(max_features=30, n_estimators=200)
        forest_reg.fit(training_data_prepped, training_labels)
        #Predict
        prediction = forest_reg.predict(avg_test_data_prepped) #Predictions for test state
    if scheme == 'Ensemble':
        elastic_reg = get_elastic_model(training_data_prepped, training_labels)
        prediction_elastic = elastic_reg.predict(avg_test_data_prepped)
        mlp_reg = MLPRegressor(hidden_layer_sizes=[250, 250, 250],max_iter=10000, early_stopping=True, alpha=0.01)
        mlp_reg.fit(training_data_prepped, training_labels)
        #Predict
        prediction_mlp = mlp_reg.predict(avg_test_data_prepped)
        forest_reg = RandomForestRegressor(max_features=30, n_estimators=200)
        forest_reg.fit(training_data_prepped, training_labels)
        #Predict
        prediction_rf = forest_reg.predict(avg_test_data_prepped) #Predictions for test state
        prediction = np.mean(np.concatenate([prediction_elastic.reshape(100,1), prediction_mlp.reshape(100,1), prediction_rf.reshape(100,1)], axis=1), axis=1)
    return prediction

national_plot(data['GOP2016']-data['GOP2012'], data_min=-0.2, data_max=0.2,vmin=-20.0, vmax=20.0, plt_title='2017-9-13-election-factors/GOP_vote_change',colorbar_label='GOP Vote Change', use_cmap = cm.seismic, AK_value = False)

fig = plt.figure(figsize=[10,8])
plt.scatter(100.0*(data['Population2016']-data['Population1980'])/data['Population1980'],100.0*(data['GOP2016']-data['GOP2012']),s=250.*data.loc[:,'Population2016']/np.max(data.loc[:,'Population2016'].values),c=cm.seismic(data.loc[:,'GOP2016']))
ax = plt.gca()
ax.set_facecolor('#202020')
plt.axhline(0.0, linewidth=0.5, color='white',linestyle='--')

fmtr = mtick.StrMethodFormatter('{x:,g}%')
ax.xaxis.set_major_formatter(fmtr)
ax.yaxis.set_major_formatter(fmtr)
plt.ylim([-20.0, 20.0])
plt.xlim([-100.0, 400.0])
plt.ylabel('GOP Vote Change 2012-2016')
plt.xlabel('Change in Population 1980-2016')
plt.savefig('plots/2017-9-13-election-factors/population_change_correlation.pdf',bbox_inches='tight')
plt.show()

#Print correlation data
correlations = data.corr()['GOP2016'].copy()
print(correlations.sort_values(ascending=False))
correlations = data.corr()['GOPChange'].copy()
print(correlations.sort_values(ascending=False))

white_change = {'Name': 'White 2015', 'Data': calculate_avg_county_range(data, 'Linear', 'White2015')}
black_change = {'Name': 'Black 2015', 'Data': calculate_avg_county_range(data, 'Linear', 'Black2015')}
unemployment_change = {'Name': 'Unemployment Rate', 'Data': calculate_avg_county_range(data, 'Linear', 'Unemployment2016')}
foodstamps_change = {'Name': 'Food Stamps', 'Data': calculate_avg_county_range(data, 'Linear', 'FoodStamps')}
median_income_change = {'Name': 'Median Income', 'Data': calculate_avg_county_range(data, 'Linear', 'MedianIncome')}

avg_county_plotter([white_change, black_change, unemployment_change, foodstamps_change, median_income_change])


slopes = find_slope(data, feature)
plotter.national_plot(slopes, data_min=-0.01, data_max=-1.0, vmin=-1.0, vmax=100.0, plt_title=feature, colorbar_label='Slope', use_cmap=cm.seismic, AK_value=False)
