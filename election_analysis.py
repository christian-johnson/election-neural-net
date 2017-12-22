import numpy as np
import pandas as pd
#import tensorflow as tf
import itertools
import matplotlib as mpl
from matplotlib import pyplot as plt

from matplotlib import cm
import pickle
import matplotlib.ticker as mtick

import hashlib

import load_data as ld
#tf.logging.set_verbosity(tf.logging.INFO)
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

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


def split_by_state(data, test_state):
    states_codes={'AL':'01','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','FL':'12','GA':'13','HI':'15','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}
    test_indices = []
    for state in [test_state]:
        test_indices.append(np.where((data.index>int(states_codes[state])*1000) & (data.index<(int(states_codes[state])+1)*1000))[0])
    test_data = data.loc[data.index[np.concatenate(test_indices)],:]
    training_data = data.loc[data.index[np.where([data.index[k] not in data.index[np.concatenate(test_indices)] for k in range(len(data.index))])],:]
    print(len(test_data))
    print(len(training_data))
    return training_data, test_data


#Class to add combined features to the data (i.e. features that are combinations of existing features)
#Can be incorporated nicely into a data-preparation pipeline with this format
#Hyperparameters: demographicsChanges controls whether changes from 2009-2015 in ethnic/racial makeup are included
#economicChanges controls whether changes in unemployment, businesses per person, population, and rent burdened are included
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, demographicsChanges=False, economicChanges=False):
        self.demographicsChanges = demographicsChanges
        self.economicChanges = economicChanges

        self.white2015_index = 6
        self.white2009_index = 12
        self.black_2015_index = 7
        self.black_2009_index = 13
        self.hispanic_2015_index = 8
        self.hispanic_2009_index = 14
        self.asian_2015_index = 9
        self.asian_2009_index = 15
        self.indian_2015_index = 10
        self.indian_2009_index = 16
        self.youth_2015_index = 11
        self.youth_2009_index = 17

        self.unemployment_2007_index = 20
        self.unemployment_2011_index = 19
        self.unemployment_2016_index = 18
        self.businesses_2009_index = 28
        self.businesses_2016_index = 27
        self.population_2016_index = 3
        self.population_2009_index = 5
        self.rent_burdened_2010_index = 30
        self.rent_burdened_2015_index = 29


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        #Changes in demographics
        if self.demographicsChanges:
            white_change = X[:,self.white2015_index]-X[:,self.white2009_index]
            black_change = X[:,self.black_2015_index]-X[:,self.black_2009_index]
            hispanic_change = X[:,self.hispanic_2015_index]-X[:,self.hispanic_2009_index]
            asian_change = X[:,self.asian_2015_index]-X[:,self.asian_2009_index]
            indian_change = X[:,self.indian_2015_index]-X[:,self.indian_2009_index]
            youth_change = X[:,self.youth_2015_index]-X[:,self.youth_2009_index]

            #if we want the changes, we don't want the old data itself
            X = np.c_[X, white_change, black_change, hispanic_change, asian_change, indian_change, youth_change]

        #Changes in economic indicators
        if self.economicChanges:
            unemployment_2007_change = X[:,self.unemployment_2016_index]-X[:,self.unemployment_2007_index]
            unemployment_2011_change = X[:,self.unemployment_2016_index]-X[:,self.unemployment_2011_index]
            businesses_change = X[:,self.businesses_2016_index]-X[:,self.businesses_2009_index]
            population_change = X[:,self.population_2016_index]-X[:,self.population_2009_index]
            rent_burdened_change = X[:,self.rent_burdened_2015_index]-X[:,self.rent_burdened_2010_index]
            X = np.c_[X, unemployment_2007_change, unemployment_2011_change, businesses_change, population_change, rent_burdened_change]

        #Delete old data- don't want it cluttering the algorithm
        X = np.delete(X, [self.white2009_index, self.black_2009_index, self.hispanic_2009_index, self.asian_2009_index, self.indian_2009_index, self.youth_2009_index, self.unemployment_2007_index, self.unemployment_2011_index, self.businesses_2009_index, self.population_2009_index, self.rent_burdened_2010_index], axis=1)
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

#Features which we will train on. Modified later in the pipeline by CombinedAttributesAdder
attribs = [
        'GOP2012',
        'Turnout2012',
        'Turnout2016',
        'Population2016',
        'Population2012',
        'Population2009',
        'PopulationDensity2016',
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
        'WinterWheat',
        'MedianIncome'
    ]

shortAttribs = [
        'PopulationDensity2016',
        'White2015',
        'Black2015',
        'Hispanic2015',
        'Asian2015',
        'Indian2015',
        'Youth2015',
        'MedianAge',
        'Homeownership',
    ]

class ElectionAnalysis():
    """
    A general class for analyzing election data with machine learning.
    Includes methods for training ML models and predicting/finding residuals
    Also includes various scrips for making pretty plots
    """
    def __init__(self, regressionType = 'Linear', numDivisions = 50, attributes = attribs):
        self.regressionType = regressionType
        self.numDivisions = numDivisions
        self.attributes = attributes
        self.results = pd.Series()

        if regressionType == 'Linear':
            self.regressor = ElasticNet(alpha=0.001, l1_ratio=0.1)
        elif regressionType == 'MLP':
            self.regressor = MLPRegressor(hidden_layer_sizes=[100, 100, 100],max_iter=25000, early_stopping=True, alpha=0.2)
        elif regressionType == 'RandomForest':
            self.regressor = RandomForestRegressor(max_features=30, n_estimators=200)

    def loadData(self):
        """
        Load data into memory from various spreadsheets & databases in the folder 'data_spreadsheets/'
        """
        self.data = ld.load_data()

    def cleanData(self):
        """
        Remove a handful of problem counties in South Dakota from the dataset
        """
        self.data = ld.clean_data(self.data)

    def editData(self, fipsId, attribute, newValue):
        """
        Edit a particular data point- whether to inject votes or to find slope of predicted vote change
        fipsId: the FIPS ID of the county to edit
        attribute: which parameter you want to change
        newValue: the new value of the parameter
        """
        self.data.loc[fipsId, attribute] = newValue

    def splitTestTrain(self, i=0, hash=hashlib.md5):
        """
        Split data by an index i- compute the hash of the FIPS code and check whether it is between i/numDivisions and (i+1)/numDivisions
        i: which group of counties do you want to split around?
        """
        if self.numDivisions>1:
            hashes = list(map(float,map((lambda id_: int(hash(str(id_).encode('utf-8')).hexdigest(), 16)%256), self.data.index.map(int))))
            hashes = [value/256.0 for value in hashes]

            testSet = list(map((lambda theHash: theHash>=float(i)/self.numDivisions and theHash<float(i+1.0)/self.numDivisions), hashes))
            trainingSet = [not bool for bool in testSet]
            #Return training data, test data
            self.trainingData = self.data.loc[trainingSet]
            self.testData = self.data.loc[testSet]
        else:
            self.trainingData = self.data

    def makePipeline(self, demographicsChanges=True, economicChanges=True):
        """
        Make a scikit-learn pipeline to do standard data processing tasks like:
        - Normalize the data to be between 0 and 1
        - Add combined attributes (i.e. changes to data over time)
        - Select which attributes to look at
        """
        self.dataPipeline = Pipeline([
            ('selector', DataFrameSelector(self.attributes)),
            ('attribs', CombinedAttributesAdder(demographicsChanges=demographicsChanges, economicChanges=economicChanges)),
            ('std_scaler', MinMaxScaler()),])#StandardScaler()),])

    def prepData(self, labelName):
        """
        Define the features & labels of the training and test sets
        Then transform the data via the pipeline
        After this is run, the data should be ready for training
        """
        self.trainingFeatures = self.trainingData.drop(labelName, axis=1)
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print(self.trainingFeatures.head())

        self.trainingFeatures = self.dataPipeline.fit_transform(self.trainingFeatures)
        self.trainingLabels = self.trainingData[labelName].copy()
        if labelName in self.testData.columns:
            self.testFeatures = self.testData.drop(labelName, axis=1)

            self.testLabels = self.testData[labelName].copy()
            self.testFeatures = self.dataPipeline.transform(self.testFeatures)
        else:
            self.testFeatures = self.dataPipeline.transform(self.testData)
            #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #    print(self.testData.head())



    def trainModel(self):
        """
        Train the regressor on self.trainingData with self.trainingLabels
        """
        self.regressor.fit(self.trainingFeatures, self.trainingLabels)

    def predictModel(self):
        """
        Use the trained regressor to predict the labels of the self.testData
        """
        self.predictions = self.regressor.predict(self.testFeatures)
        return self.predictions

    def residModel(self):
        """
        Calculate the difference between the prediction and the data
        """
        self.residuals = self.testLabels-self.predictions
        return self.residuals

    def concatenateResiduals(self):
        """
        """
        self.results = pd.concat([self.results, self.residuals])

    def rmsModel(self):
        """
        Get the RMS of the residuals
        """
        self.rms = np.sqrt(np.mean(self.results.values**2))
        return self.rms

    def bubblePlot(self, xValues, yValues, sValues, cValues, fileName, xLabel, yLabel, minXValue, minYValue, maxXValue, maxYValue, cMap = cm.seismic):
        """
        Create a nice-looking bubble plot where each bubble is a single county
        xValues: x positions of the bubbles (np array)
        yValues: y positions of the bubbles (np array)
        sValues: size of the bubbles (np array)
        cValues: color of the bubbles (np array of values preferably between 0 and 1)
        fileName: where to save the file (string)
        xLabel: Label of the x-axis (string)
        yLabel: Label of the y-axis (string)
        """
        fig = plt.figure(figsize=[10,8])
        ax = plt.gca()
        ax.set_facecolor('#202020')
        plt.scatter(xValues, yValues, s=sValues,c=cMap(cValues))
        plt.axhline(0.0, linewidth=0.5, color='white',linestyle='--')
        fmtr = mtick.StrMethodFormatter('{x:,g}%')
        fmtr2 = mtick.StrMethodFormatter('{x:,g}')

        ax.xaxis.set_major_formatter(fmtr2)
        ax.yaxis.set_major_formatter(fmtr)
        plt.ylim([minYValue, maxYValue])
        plt.xlim([minXValue, maxXValue])
        plt.ylabel(yLabel)
        plt.xlabel(xLabel)
        plt.savefig('plots/'+fileName+'.pdf',bbox_inches='tight')
        plt.show()

    def countyPlot(self, dataSeries, dataMin, dataMax, vMin, vMax, pltTitle, cLabel, cMap = cm.seismic, AK_value = False):
        """
        Plots a map of the US, with each county colored by a data series
        dataSeries: a single-column Pandas DataFrame with the indices given by integer FIPS codes
        vMin, vMax: minimum and maximum of the colorbar- should correspond to the minima and maxima of the data
        cLabel: Label of the colorbar
        """
        fig = plt.figure(figsize=(10.0,7.0))
        #Mainland
        ax = plt.axes([0.0,0.0,1.0,1.0],projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0, cutoff=-20), aspect=1.15, frameon=False)
        ax.set_extent([-120.0,-74., 21.,47.])
        #Alaska
        ax2 = plt.axes([0.0,0.0,0.26,0.26],projection=ccrs.LambertConformal(central_longitude=-156.0, central_latitude=53.5, cutoff=-20), aspect=1.3, frameon=False)
        ax2.set_extent([-180.0,-132., 45.,62.])
        #Hawaii
        ax3 = plt.axes([0.25, 0.00,0.2,0.2],projection=ccrs.LambertConformal(central_longitude=-157.0, central_latitude=20.5, cutoff=-20), aspect=1.15, frameon=False)
        ax3.set_extent([-162.0,-154., 18.,23.])
        fileName = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
        lineWidth = 0.0
        edgeColor = 'black'

        for state, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
            id = int(record.__dict__['attributes']['GEOID'])
            #Shannon County was renamed to Oglala Dakota county, and its FIPS code was changed
            if id == '46102':
                id = '46113'

            if id in dataSeries.index:
                #Normalize the value so it matches the color mapping
                faceColor = cMap((dataSeries.loc[id]-dataMin)/(dataMax-dataMin))
                #Is the county in Hawaii, Alaska, or the mainland?
                if int(record.__dict__['attributes']['GEOID'])<2991 and int(record.__dict__['attributes']['GEOID'])>2013:
                    ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=faceColor, edgecolor=edgeColor, linewidth=lineWidth)
                elif int(record.__dict__['attributes']['GEOID'])<15010 and int(record.__dict__['attributes']['GEOID'])>15000:
                    ax3.add_geometries(state, crs=ccrs.Miller(), facecolor=faceColor, edgecolor=edgeColor, linewidth=lineWidth)
                else:
                    ax.add_geometries(state, crs=ccrs.Miller(), facecolor=faceColor, edgecolor=edgeColor, linewidth=lineWidth)
            else:
                #Missing data goes in gray
                faceColor = 'gray'
                #Is the county in Hawaii, Alaska, or the mainland?
                if int(record.__dict__['attributes']['GEOID'])<2991 and int(record.__dict__['attributes']['GEOID'])>2013:
                    #Make Alaska a uniform color (because election results aren't available on county-by-county level in Alaska)
                    if AK_value:
                        ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=cm.seismic(AK_value), edgecolor=edgeColor, linewidth=lineWidth)
                    else:
                        ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=faceColor, edgecolor=edgeColor, linewidth=lineWidth)

                elif int(record.__dict__['attributes']['GEOID'])<15010 and int(record.__dict__['attributes']['GEOID'])>15000:
                    ax3.add_geometries(state, crs=ccrs.Miller(), facecolor=faceColor, edgecolor=edgeColor, linewidth=lineWidth)
                else:
                    ax.add_geometries(state, crs=ccrs.Miller(), facecolor=faceColor, edgecolor=edgeColor, linewidth=lineWidth)

        #Mark a black line around each state
        fileName = 'cb_2015_us_state_5m/cb_2015_us_state_5m.shp'
        lineWidth = 0.25
        for state, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
            id = int(record.__dict__['attributes']['GEOID'])
            facecolor = cMap(0.0)
            #Is the county in Hawaii, Alaska, or the mainland?
            if int(record.__dict__['attributes']['GEOID'])==2:
                ax2.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor=edgeColor, linewidth=lineWidth)
            elif int(record.__dict__['attributes']['GEOID'])==15:
                ax3.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor=edgeColor, linewidth=lineWidth)
            else:
                ax.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor=edgeColor, linewidth=lineWidth)

        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
        ax2.background_patch.set_visible(False)
        ax2.outline_patch.set_visible(False)
        ax3.background_patch.set_visible(False)
        ax3.outline_patch.set_visible(False)
        #Add colorbar & format ticks
        axc = plt.axes([0.25, 0.92, 0.5, 0.012], frameon=False)
        norm = mpl.colors.Normalize(vmin=vMin, vmax=vMax)
        numTicks = 9
        cbarStep=float((vMax-vMin)/(numTicks-1.0))
        cb = mpl.colorbar.ColorbarBase(axc, ticks=np.linspace(vMin, vMax, numTicks),cmap=cMap,norm=norm,orientation='horizontal')
        cb.set_ticklabels(['{:.0f}'.format(x) for x in np.arange(vMin, vMax+cbarStep, cbarStep)])
        cb.ax.xaxis.set_ticks_position('top')
        cb.set_label(cLabel, fontdict = {
            'horizontalalignment' : 'center'
            })
        plt.savefig('plots/'+str(pltTitle)+'.pdf',bbox_inches='tight')
        plt.show()

    def congressionalDistrictPlot(self, dataSeries, pltTitle, vMin, vMax, cLabel, cMap = cm.seismic):
        """
        Make a map of the Census tracts in a particular Congressional District
        The value to plot is contained in the dataSeries, and the index of the dataSeries is the Census tract number
        """

        fig = plt.figure(figsize=(10.0,7.0))
        #Mainland
        ax = plt.axes([0.0,0.0,1.0,1.0],projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0, cutoff=-20), aspect=1.15, frameon=False)
        ax.set_extent([-118.88,-117.606, 33.126,33.544])
        edgeColor = 'black'
        #Color in the census tracts
        fileName = 'cb_2016_06_tract_500k/cb_2016_06_tract_500k.shp'
        lineWidth = 0.0
        for tract, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
            if '0625'+record.__dict__['attributes']['COUNTYFP']+str(record.__dict__['attributes']['TRACTCE']) in dataSeries.index:
                faceColor = cMap(float(dataSeries.loc['0625'+record.__dict__['attributes']['COUNTYFP']+str(record.__dict__['attributes']['TRACTCE'])]))
                ax.add_geometries(tract, crs=ccrs.Miller(), facecolor=faceColor, alpha=1.0, edgecolor=edgeColor, linewidth=lineWidth)

        #Mark a black line around each congressional district
        fileName = 'cb_2016_us_cd115_500k/cb_2016_us_cd115_500k.shp'
        for district, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
            id = int(record.__dict__['attributes']['GEOID'])
            if int(id) != 625:
                ax.add_geometries(district, crs=ccrs.Miller(), facecolor='white', alpha=1.0, edgecolor='white', linewidth=1.0)

        for district, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
            id = int(record.__dict__['attributes']['GEOID'])
            if int(id) == 625:
                ax.add_geometries(district, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor='black', linewidth=2.0)

        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)

        plt.suptitle('Elastic Regression',y=0.05)

        #Add colorbar & format ticks
        axc = plt.axes([0.25, 0.92, 0.5, 0.012], frameon=False)
        norm = mpl.colors.Normalize(vmin=vMin, vmax=vMax)
        numTicks = 9
        cbarStep=float((vMax-vMin)/(numTicks-1.0))
        cb = mpl.colorbar.ColorbarBase(axc, ticks=np.linspace(vMin, vMax, numTicks),cmap=cMap,norm=norm,orientation='horizontal')
        cb.set_ticklabels(['{:.0f}%'.format(x) for x in np.arange(vMin, vMax+cbarStep, cbarStep)])
        cb.ax.xaxis.set_ticks_position('top')
        cb.set_label(cLabel, fontdict = {
            'horizontalalignment' : 'center'
            })
        plt.savefig('plots/'+str(pltTitle)+'.pdf',bbox_inches='tight')

        plt.show()

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
    print("Random Forest best estimator:" + str(grid_search.best_estimator_))

    param_grid = [
        {'hidden_layer_sizes':[[100, 100], [100, 100, 100]],
        'activation':['tanh','relu'],
        'alpha':[1e-8, 1e-6, 1e-5],
        'learning_rate':['constant','adaptive', 'invscaling'],
        'max_iter':[100, 1000, 10000]}
    ]
    grid_search = GridSearchCV(mlp_reg, param_grid, cv=10, scoring='neg_mean_squared_error')
    grid_search.fit(training_data_prepped, training_labels)
    print("MLP best estimator:" + str(grid_search.best_estimator_))

    #Results:
    #Random Forest: max_features=30, n_estimators=150
    #MLP: activation='tanh', alpha=1e-06, batch_size='auto', beta_1=0.9,
    #   beta_2=0.999, early_stopping=False, epsilon=1e-08,
    #   hidden_layer_sizes=[100, 100, 100], learning_rate='constant',
    #   learning_rate_init=0.001, max_iter=100, momentum=0.9,
    #   nesterovs_momentum=True, power_t=0.5, random_state=None,
    #   shuffle=True, solver='adam', tol=0.0001, validation_fraction=0.1,
    #   verbose=False, warm_start=False)




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


#Split all the counties into num_divisions "states" randomly
#Then for each "state", train a ML model on the remaining part of the country
#Use that trained model to predict the results in that "state". Concatenate the results
def calculate_nationwide_resids(data, scheme, num_divisions = 50):
    #This function splits the country up randomly into num_divisions groups of counties, trains an ML technique on the training set
    #and then applies it to the test set. Then returns the residuals
    results = pd.Series()
    for i in range(int(num_divisions)):
        data_pipeline = make_pipeline(scheme)
        #Split into training/test data
        training_data, test_data = split_test_train(data, i, num_test_sets = num_divisions)
        training_features = training_data.drop('GOP2016', axis=1)
        training_labels = training_data['GOP2016'].copy()
        test_features = test_data.drop('GOP2016', axis=1)
        test_labels = test_data['GOP2016'].copy()
        #Prepare data
        training_data_prepped = data_pipeline.fit_transform(training_features)
        test_data_prepped = data_pipeline.transform(test_features)

        if scheme == 'Linear':
            #Train
            elastic_reg = get_elastic_model(training_data_prepped, training_labels)
            #Predict
            predictions = elastic_reg.predict(test_data_prepped)

        if scheme == 'MLP':
            #Train
            mlp_reg = MLPRegressor(hidden_layer_sizes=[100, 100, 100],max_iter=40000, early_stopping=True, alpha=0.01)
            mlp_reg.fit(training_data_prepped, training_labels)
            #Predict
            predictions = mlp_reg.predict(test_data_prepped)

        if scheme == 'RandomForest':
            #Train
            forest_reg = RandomForestRegressor(max_features=30, n_estimators=200)
            forest_reg.fit(training_data_prepped, training_labels)
            #Predict
            predictions = forest_reg.predict(test_data_prepped) #Predictions for test state

        residuals = test_labels-predictions
        #Concatenate to all results
        results = pd.concat([results,residuals])
    return results

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

def plots_for_2017_8_12_election_update(data):
    linear_results = calculate_nationwide_resids(data,scheme='Linear', num_divisions=50.0)
    mlp_results = calculate_nationwide_resids(data,scheme='MLP', num_divisions=50.0)
    randomforest_results = calculate_nationwide_resids(data,scheme='RandomForest',num_divisions=50.0)

    ensemble_data = pd.concat([linear_results, mlp_results, randomforest_results], axis=1)
    ensemble_mean = ensemble_data.mean(axis=1)
    print("RMS MLP = " + str(100.0*np.sqrt(np.mean(mlp_results.values**2))) + "%")
    print("RMS Random Forest = " + str(100.0*np.sqrt(np.mean(randomforest_results.values**2))) + "%")
    print("RMS Elastic = " + str(100.0*np.sqrt(np.mean(linear_results.values**2))) + "%")

    plotter.national_plot(mlp_results, data_min=-0.2, data_max=0.2,vmin=-20.0, vmax=20.0, plt_title='model_comparison/vote_residmap_mlp',colorbar_label='GOP Vote Share Residual', use_cmap = cm.seismic, AK_value = False)
    plotter.national_plot(randomforest_results, data_min=-0.2, data_max=0.2,vmin=-20.0, vmax=20.0, plt_title='model_comparison/vote_residmap_random_forest',colorbar_label='GOP Vote Share Residual', use_cmap = cm.seismic, AK_value = False)
    plotter.national_plot(linear_results, data_min=-0.2, data_max=0.2,vmin=-20.0, vmax=20.0, plt_title='model_comparison/vote_residmap_linear',colorbar_label='GOP Vote Share Residual', use_cmap = cm.seismic, AK_value = False)

def plots_for_2017_9_13_election_factors(data):
    plotter.national_plot(data['GOP2016']-data['GOP2012'], data_min=-0.2, data_max=0.2,vmin=-20.0, vmax=20.0, plt_title='2017-9-13-election-factors/GOP_vote_change',colorbar_label='GOP Vote Change', use_cmap = cm.seismic, AK_value = False)

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

def plots_for_2017_12_25():
    model = ElectionAnalysis()
    model.loadData()
    model.cleanData()
    model.bubblePlot(xValues=100*model.data['Bachelors'], yValues=100*model.data['Unemployment2016'], sValues=250*model.data['Population2016']/max(model.data['Population2016']), cValues=model.data['GOP2016'], fileName='Unemployment_vs_Bachelors', xLabel='Bachelors Degrees', yLabel='Unemployment Rate', minXValue=0.0, minYValue=0.0, maxXValue=60.0, maxYValue=12.0, cMap = cm.seismic)
    model.bubblePlot(xValues=100*model.data['RentBurdened2015'], yValues=100*model.data['Homeownership'], sValues=250*model.data['Population2016']/max(model.data['Population2016']), cValues=model.data['GOP2016'], fileName='Homeownership_vs_rentBurdened', xLabel='Rent Burdened', yLabel='Homeownership Rate', minXValue=10.0, minYValue=18.0, maxXValue=60.0, maxYValue=100.0, cMap = cm.seismic)
    model.bubblePlot(xValues=100*(model.data['EvangelicalProtestant']), yValues=100*model.data['ClimateChange'], sValues=250*model.data['Population2016']/max(model.data['Population2016']), cValues=model.data['GOP2016'], fileName='Religion_vs_ClimateChange', xLabel='Evangelical Protestant Fraction', yLabel='Belief in Human-Caused Climate Change', minXValue=00.0, minYValue=35.0, maxXValue=100.0, maxYValue=70.0, cMap = cm.seismic)
    model.bubblePlot(xValues=(100*model.data['MedianAge']), yValues=100*model.data['Turnout2016'], sValues=250*model.data['Population2016']/max(model.data['Population2016']), cValues=model.data['GOP2016'], fileName='Age_vs_Turnout', xLabel='Median Age', yLabel='Vote Turnout 2016', minXValue=20.0, minYValue=15.0, maxXValue=65, maxYValue=70.0, cMap = cm.seismic)
    model.countyPlot(dataSeries=model.data['EvangelicalProtestant'], dataMin=0.0, dataMax=1.0, vMin=0.0, vMax=100.0, pltTitle='EvangelicalProtestant', cLabel='Evangelical Protestant', cMap = cm.Reds, AK_value = False)
    model.countyPlot(dataSeries=model.data['ClimateChange'], dataMin=0.4, dataMax=0.65, vMin=40.0, vMax=65.0, pltTitle='ClimateChange', cLabel='Belief in Human-Caused Climate Change', cMap = cm.viridis, AK_value = False)
    model.countyPlot(dataSeries=model.data['MedianAge'], dataMin=0.2, dataMax=0.7, vMin=20.0, vMax=70.0, pltTitle='MedianAge', cLabel='Median Age', cMap = cm.YlGnBu, AK_value = False)

    model.makePipeline()
    for i in range(50):
        #Split into training/test data
        model.splitTestTrain(i)
        model.prepData(labelName = 'GOP2016')
        model.trainModel()
        model.predictModel()
        model.residModel()
        model.concatenateResiduals()
    print(model.rmsModel())
    model.countyPlot(dataSeries=model.results, dataMin = -0.2, dataMax = 0.2, vMin = -20.0, vMax = 20.0, pltTitle = 'linearResiduals', cLabel = 'GOP Residual Vote', cMap = cm.seismic, AK_value = False)

def main():

    #Initialize the analysis object
    model = ElectionAnalysis(regressionType='Linear',numDivisions=1, attributes=shortAttribs)
    model.loadData()
    model.makePipeline(demographicsChanges=False, economicChanges=False)
    model.splitTestTrain()
    #Load up the Census data for the tracts as the testData
    df = pd.read_csv('data_spreadsheets/CA_25_Census/DEC_10_115_DP1_with_ann.csv',index_col=1)

    #Remove useless information (i.e. pointless CSV headers and percent information- we can calculate that ourselves!)
    df = df.drop('Id2')
    df = df.drop('GEO.id',1)
    df = df.drop('GEO.display-label',1)
    for columnName in df.columns:
        if columnName[:4] == 'HD02' or columnName == 'HD01_S179' or columnName == 'HD01_S178':
            df = df.drop(columnName, 1)
    for rowName in df.index:
        if float(df['HD01_S001'].loc[rowName])==0.0:
            df = df.drop(rowName)
    model.testData = pd.DataFrame(columns=shortAttribs, index=df.index)

    model.testData['PopulationDensity2016'] = np.array(list(map(float,df['HD01_S001'].values)))
    model.testData['White2015'] = np.array(list(map(float,df['HD01_S100'].values)))/np.array(list(map(float,df['HD01_S001'].values)))
    model.testData['Black2015'] = np.array(list(map(float,df['HD01_S101'].values)))/np.array(list(map(float,df['HD01_S001'].values)))
    model.testData['Hispanic2015'] = np.array(list(map(float,df['HD01_S107'].values)))/np.array(list(map(float,df['HD01_S001'].values)))
    model.testData['Asian2015'] = np.array(list(map(float,df['HD01_S103'].values)))/np.array(list(map(float,df['HD01_S001'].values)))
    model.testData['Indian2015'] = np.array(list(map(float,df['HD01_S102'].values)))/np.array(list(map(float,df['HD01_S001'].values)))
    model.testData['MedianAge'] = np.array(list(map(float,df['HD01_S020'].values)))/100.0
    model.testData['Youth2015'] = 1.0-np.array(list(map(float,df['HD01_S022'].values)))/np.array(list(map(float,df['HD01_S001'].values)))
    model.testData['Homeownership'] = np.array(list(map(float,df['HD01_S181'].values)))/np.array(list(map(float,df['HD01_S169'].values)))
    #Convert to population density (people per square kilometer)
    print('eligible voters:')
    print(np.sum(np.array(list(map(float,df['HD01_S022'].values)))))
    fileName = 'cb_2016_06_tract_500k/cb_2016_06_tract_500k.shp'
    for tract, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
        tractName = '0625'+record.__dict__['attributes']['COUNTYFP']+str(record.__dict__['attributes']['TRACTCE'])
        if tractName in model.testData.index:
            model.testData['PopulationDensity2016'].loc[tractName] = model.testData['PopulationDensity2016'].loc[tractName]/(record.__dict__['attributes']['ALAND']/1000**2)


    model.testData = model.testData.dropna(how='any')
    model.trainingData = model.trainingData.dropna(how='any')
    model.prepData(labelName = 'GOP2016')
    model.trainModel()
    model.predictModel()

    model.congressionalDistrictPlot(pd.Series(index=model.testData.index, data=model.predictions),vMin=0.0, vMax=100.0, cLabel = 'GOP Vote Share', pltTitle='2017-12-22-california25/elastic')
    model.bubblePlot(xValues=(100*model.data['MedianAge']), yValues=100*model.data['Turnout2016'], sValues=250*model.data['Population2016']/max(model.data['Population2016']), cValues=model.data['GOP2016'], fileName='2017-12-22-california25/Age_vs_Turnout', xLabel='Median Age', yLabel='Vote Turnout 2016', minXValue=20.0, minYValue=15.0, maxXValue=65, maxYValue=85.0, cMap = cm.seismic)


    input('wait for key')
    for feature in attribs:
        slopes = find_slope(data, feature)
        plotter.national_plot(slopes, data_min=-0.01, data_max=-1.0, vmin=-1.0, vmax=100.0, plt_title=feature, colorbar_label='Slope', use_cmap=cm.seismic, AK_value=False)

if __name__ == '__main__':
    main()
