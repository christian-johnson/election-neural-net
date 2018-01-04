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
        'MedianIncome'
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
        self.stateCodes={'AL':'01','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','FL':'12','GA':'13','HI':'15','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}
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

    def splitTestTrain(self, technique='hash', i=0, hash=hashlib.md5):
        """
        Split data by an index i
        If technique == 'hash' (default & recommended for a national analysis), compute the hash of the FIPS code and check whether it is between i/numDivisions and (i+1)/numDivisions
        If technique == 'statewise' (if you want to focus just on one state), consider the ith state (alphabetically) as the test data
        """
        if technique=='hash':
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
        if technique=='statewise':

            testIndices= np.where((self.data.index>int(self.stateCodes[i])*1000) & (self.data.index<(int(self.stateCodes[i])+1)*1000))
            #testIndices = np.where((self.data.index>int(stateCodes[i])*1000) & (self.data.index<(int(stateCodes[i])+1)*1000))[0])
            self.testData = self.data.loc[self.data.index[np.concatenate(testIndices)],:]
            self.trainingData = self.data.loc[self.data.index[np.where([self.data.index[k] not in self.data.index[np.concatenate(testIndices)] for k in range(len(self.data.index))])],:]



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
        fmtr2 = mtick.StrMethodFormatter('{x:,g}%')

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
        cb.set_ticklabels(['{:.0f}%'.format(x) for x in np.arange(vMin, vMax+cbarStep, cbarStep)])
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


def main():
    pass
if __name__ == '__main__':
    main()
