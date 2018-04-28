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
        #X = np.delete(X, [self.white2009_index, self.black_2009_index, self.hispanic_2009_index, self.asian_2009_index, self.indian_2009_index, self.youth_2009_index, self.unemployment_2007_index, self.unemployment_2011_index, self.businesses_2009_index, self.population_2009_index, self.rent_burdened_2010_index], axis=1)
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

class ElectionAnalysis():
    """
    A general class for analyzing election data with machine learning.
    Includes methods for training ML models and predicting/finding residuals
    Also includes various scrips for making pretty plots
    """
    def __init__(self, attribs, regressionType = 'Linear', numDivisions = 50, reloadData = False):
        self.regressionType = regressionType
        self.numDivisions = numDivisions
        self.attributes = attribs
        self.results = pd.Series()
        self.stateCodes={'AL':'01','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','FL':'12','GA':'13','HI':'15','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}
        self.acres_to_m2 = 4047.

        if regressionType == 'Linear':
            self.regressor = ElasticNet(alpha=0.001, l1_ratio=0.1)
        elif regressionType == 'MLP':
            self.regressor = MLPRegressor(hidden_layer_sizes=[100, 100, 100],max_iter=25000, early_stopping=True, alpha=0.2)
        elif regressionType == 'RandomForest':
            self.regressor = RandomForestRegressor(max_features=30, n_estimators=200)

        if reloadData:
            self.loadData()
        else:
            file = open('database.pk1','rb')
            self.data = pickle.load(file)
            file.close()

    def removeProblemCounties(self):
        if 46103 in self.data.index:
            self.data = self.data.drop(46103)
        if 46105 in self.data.index:
            self.data = self.data.drop(46105)
        if 46109 in self.data.index:
            self.data = self.data.drop(46109)
        if 46111 in self.data.index:
            self.data = self.data.drop(46111)
        if 46102 in self.data.index:
            self.data = self.data.drop(46102)

    def loadData(self):
        """
        Load data into memory from various spreadsheets & databases in the folder 'data_spreadsheets/'
        """

        #Election data
        election_results = pd.read_excel('data_spreadsheets/US_County_Level_Presidential_Results_08-16.xlsx', 'Sheet 1', header=1, usecols=list(range(0,100)), skiprows=range(2,31))
        election_results.index = election_results['combined_fips']
        alabama_special_election_results = pd.read_excel('data_spreadsheets/alabama_special_election2017.xlsx', 'Sheet 1', index_col=0, header=1, usecols=list(range(1,100)))

        #Data from St. Louis GeoFRED
        food_stamps = pd.read_excel('data_spreadsheets/food_stamps_1989_2013.xls','Sheet0', index_col=0, header=0, usecols=[2,3,7,9]+list(range(11,28)))
        population = pd.read_excel('data_spreadsheets/population_1970_2016.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,50)))
        #Population is in thousands in the spreadsheet, we need to multiply everything by 1000.0:
        population = pd.DataFrame(population.values*1000.0, index=population.index, columns=population.columns)

        youth_population = pd.read_excel('data_spreadsheets/youth_percent_1989_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        black_population = pd.read_excel('data_spreadsheets/black_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        white_population = pd.read_excel('data_spreadsheets/white_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        asian_population = pd.read_excel('data_spreadsheets/asian_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        indian_population = pd.read_excel('data_spreadsheets/indian_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        hispanic_population = pd.read_excel('data_spreadsheets/hispanic_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        commute_time = pd.read_excel('data_spreadsheets/commuting_time_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,50)))
        unemployment_rate_0 = pd.read_excel('data_spreadsheets/unemployment_rate_1970_2017.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,400)))
        unemployment_rate_1 = pd.read_excel('data_spreadsheets/unemployment_rate_1970_2017.xls','Sheet1', index_col=0, header=1, skiprows=0, usecols=list(range(2,400)))
        bachelors_degrees = pd.read_excel('data_spreadsheets/bachelors_2010_2012.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        median_age = pd.read_excel('data_spreadsheets/median_age_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        rent_burdened = pd.read_excel('data_spreadsheets/rent_burdened_2010_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        homeownership_rate = pd.read_excel('data_spreadsheets/homeownership_rate_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        income_inequality = pd.read_excel('data_spreadsheets/income_inequality_2010_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,100)))
        business_establishments = pd.read_excel('data_spreadsheets/business_establishments_1990_2016.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,200)))
        median_income = pd.read_excel('data_spreadsheets/median_income_1989_2014.xls','Sheet0', index_col=0, header=1, skiprows=0, usecols=list(range(2,200)))
        #Religion data from CSV files
        religion = pd.read_stata('data_spreadsheets/religion.dta')
        religion = religion.fillna(0.0)
        evangelicals = pd.DataFrame(religion.loc[:,'evanadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Evangelical'])

        protestant = pd.DataFrame(religion.loc[:,'mprtadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Protestant'])
        blackprotestant = pd.DataFrame(religion.loc[:,'bprtadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['BlackProtestant'])
        catholic = pd.DataFrame(religion.loc[:,'cathadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Catholic'])

        #Multiple types of Judaism- add them up into a single group
        jewish = pd.DataFrame(religion.loc[:,'cjudadh'].div( religion.loc[:,'POP2010']).values,index= religion.loc[:,'fips'],columns=['Jewish'])
        jewish.add(pd.DataFrame(religion.loc[:,'ojudadh'].div( religion.loc[:,'POP2010']).values,index= religion.loc[:,'fips'],columns=['Jewish']))
        jewish.add(pd.DataFrame(religion.loc[:,'rjudadh'].div( religion.loc[:,'POP2010']).values,index= religion.loc[:,'fips'],columns=['Jewish']))
        jewish.add(pd.DataFrame(religion.loc[:,'rfrmadh'].div(religion.loc[:,'POP2010']).values,index= religion.loc[:,'fips'],columns=['Jewish']))

        muslim = pd.DataFrame(religion.loc[:,'mslmadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Muslim'])
        mormon = pd.DataFrame(religion.loc[:,'ldsadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Mormon'])

        climatechange = pd.read_csv('data_spreadsheets/yale_climate_change.csv')

        #Drug poisonings per 100,000 people per year
        drugpoisoning = pd.read_csv('data_spreadsheets/drug_poisoning.csv').pivot(index='FIPS',columns='Year',values='Estimated Age-adjusted Death Rate, 16 Categories (in ranges)')
        for column in drugpoisoning.columns:
            drugpoisoning[column] = drugpoisoning[column].str.split('-')
        drugpoisoning = drugpoisoning.dropna()
        for fips in drugpoisoning.index:
            for column in drugpoisoning.columns:
                drugpoisoning[column][fips] = float(drugpoisoning[column][fips][0].replace('>',''))


        #Agricultural data
        #Get county areas, in m^2
        filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
        geometries = pd.DataFrame([y[0].__dict__['attributes'] for y in zip(shpreader.Reader(filename).records())])
        geometries.index = list(map(int,map(str,geometries['GEOID'])))

        #Corn planted
        corn_planted = pd.read_csv('data_spreadsheets/corn_planted_2016.csv')
        corn_planted = corn_planted[np.isfinite(corn_planted['County ANSI'])]
        corn_planted = corn_planted[np.isfinite(corn_planted['State ANSI'])]
        acres_planted = list(map(str,corn_planted['Value'].values))
        corn_indices = [None]*len(acres_planted)
        corn_planted.index = list(range(len(corn_planted)))
        for i in list(range(len(acres_planted))):
            if ',' in acres_planted[i]:
                acres_planted[i] = self.acres_to_m2*float(acres_planted[i].split(',')[0]+acres_planted[i].split(',')[1])
            else:
                acres_planted[i] = self.acres_to_m2*float(acres_planted[i])
            if len(str(int(float(corn_planted.loc[i,'County ANSI']))))>2:
                corn_indices[i] = int(float(str(corn_planted.loc[i,'State ANSI'])+str(corn_planted.loc[i,'County ANSI'])))
            elif len(str(int(float(corn_planted.loc[i,'County ANSI']))))>1:
                corn_indices[i] = int(float(str(corn_planted.loc[i,'State ANSI'])+'0'+str(corn_planted.loc[i,'County ANSI'])))
            else:
                corn_indices[i] = int(float(str(corn_planted.loc[i,'State ANSI'])+'00'+str(corn_planted.loc[i,'County ANSI'])))

        corn = pd.DataFrame(acres_planted, index=corn_indices, columns=['Corn'])

        for corn_index,corn_row in corn.iterrows():
            corn.loc[corn_index,'Corn'] = corn_row[0]/geometries.loc[corn_index]['ALAND']

        #Cotton planted
        cotton_planted = pd.read_csv('data_spreadsheets/cotton_planted_2016.csv')
        cotton_planted = cotton_planted[np.isfinite(cotton_planted['County ANSI'])]
        cotton_planted = cotton_planted[np.isfinite(cotton_planted['State ANSI'])]
        acres_planted = list(map(str,cotton_planted['Value'].values))
        cotton_indices = [None]*len(acres_planted)
        cotton_planted.index = list(range(len(cotton_planted)))
        for i in list(range(len(acres_planted))):
            if ',' in acres_planted[i]:
                acres_planted[i] = self.acres_to_m2*float(acres_planted[i].split(',')[0]+acres_planted[i].split(',')[1])
            else:
                acres_planted[i] = self.acres_to_m2*float(acres_planted[i])
            if len(str(int(float(cotton_planted.loc[i,'County ANSI']))))>2:
                cotton_indices[i] = int(float(str(cotton_planted.loc[i,'State ANSI'])+str(cotton_planted.loc[i,'County ANSI'])))
            elif len(str(int(float(cotton_planted.loc[i,'County ANSI']))))>1:
                cotton_indices[i] = int(float(str(cotton_planted.loc[i,'State ANSI'])+'0'+str(cotton_planted.loc[i,'County ANSI'])))
            else:
                cotton_indices[i] = int(float(str(cotton_planted.loc[i,'State ANSI'])+'00'+str(cotton_planted.loc[i,'County ANSI'])))

        cotton = pd.DataFrame(acres_planted, index=cotton_indices, columns=['Cotton'])

        for cotton_index,cotton_row in cotton.iterrows():
            cotton.loc[cotton_index,'Cotton'] = cotton_row[0]/geometries.loc[cotton_index]['ALAND']
        #Soybeans planted
        soybeans_planted = pd.read_csv('data_spreadsheets/soybeans_planted_2016.csv')
        soybeans_planted = soybeans_planted[np.isfinite(soybeans_planted['County ANSI'])]
        soybeans_planted = soybeans_planted[np.isfinite(soybeans_planted['State ANSI'])]
        acres_planted = list(map(str,soybeans_planted['Value'].values))
        soybeans_indices = [None]*len(acres_planted)
        soybeans_planted.index = list(range(len(soybeans_planted)))
        for i in list(range(len(acres_planted))):
            if ',' in acres_planted[i]:
                acres_planted[i] = self.acres_to_m2*float(acres_planted[i].split(',')[0]+acres_planted[i].split(',')[1])
            else:
                acres_planted[i] = self.acres_to_m2*float(acres_planted[i])
            if len(str(int(float(soybeans_planted.loc[i,'County ANSI']))))>2:
                soybeans_indices[i] = int(float(str(soybeans_planted.loc[i,'State ANSI'])+str(soybeans_planted.loc[i,'County ANSI'])))
            elif len(str(int(float(soybeans_planted.loc[i,'County ANSI']))))>1:
                soybeans_indices[i] = int(float(str(soybeans_planted.loc[i,'State ANSI'])+'0'+str(soybeans_planted.loc[i,'County ANSI'])))
            else:
                soybeans_indices[i] = int(float(str(soybeans_planted.loc[i,'State ANSI'])+'00'+str(soybeans_planted.loc[i,'County ANSI'])))

        soybeans = pd.DataFrame(acres_planted, index=soybeans_indices,columns=['Soybeans'])

        for soybeans_index,soybeans_row in soybeans.iterrows():
            soybeans.loc[soybeans_index,'Soybeans'] = soybeans_row[0]/geometries.loc[soybeans_index]['ALAND']

        #Winter Wheat planted
        winter_wheat_planted = pd.read_csv('data_spreadsheets/winter_wheat_planted_2016.csv')
        winter_wheat_planted = winter_wheat_planted[np.isfinite(winter_wheat_planted['County ANSI'])]
        winter_wheat_planted = winter_wheat_planted[np.isfinite(winter_wheat_planted['State ANSI'])]
        acres_planted = list(map(str,winter_wheat_planted['Value'].values))
        winter_wheat_indices = [None]*len(acres_planted)
        winter_wheat_planted.index = list(range(len(winter_wheat_planted)))
        for i in list(range(len(acres_planted))):
            if ',' in acres_planted[i]:
                acres_planted[i] = self.acres_to_m2*float(acres_planted[i].split(',')[0]+acres_planted[i].split(',')[1])
            else:
                acres_planted[i] = self.acres_to_m2*float(acres_planted[i])
            if len(str(int(float(winter_wheat_planted.loc[i,'County ANSI']))))>2:
                winter_wheat_indices[i] = int(float(str(winter_wheat_planted.loc[i,'State ANSI'])+str(winter_wheat_planted.loc[i,'County ANSI'])))
            elif len(str(int(float(winter_wheat_planted.loc[i,'County ANSI']))))>1:
                winter_wheat_indices[i] = int(float(str(winter_wheat_planted.loc[i,'State ANSI'])+'0'+str(winter_wheat_planted.loc[i,'County ANSI'])))
            else:
                winter_wheat_indices[i] = int(float(str(winter_wheat_planted.loc[i,'State ANSI'])+'00'+str(winter_wheat_planted.loc[i,'County ANSI'])))

        winter_wheat = pd.DataFrame(acres_planted, index=winter_wheat_indices,columns=['Winter_wheat'])

        for winter_wheat_index,winter_wheat_row in winter_wheat.iterrows():
            winter_wheat.loc[winter_wheat_index,'Winter_wheat'] = winter_wheat_row[0]/geometries.loc[winter_wheat_index]['ALAND']


        #Combine relevant data into a single DataFrame
        self.data= pd.concat([
            election_results.loc[:,'per_gop_2016']-election_results.loc[:,'per_gop_2012'],
            election_results.loc[:,'per_gop_2016'],
            election_results.loc[:,'per_gop_2012'],

            #Alabama special election results, if desired
            #alabama_special_election_results.loc[:,'per_gop_2017'],

            #Will become voter turnout
            youth_population.loc[:,'2012'],
            youth_population.loc[:,'2015'],
            election_results.loc[:, 'votes_dem_2016'],
            election_results.loc[:, 'votes_gop_2016'],
            election_results.loc[:, 'votes_dem_2012'],
            election_results.loc[:, 'votes_gop_2012'],

            population.loc[:,'2016'],
            population.loc[:,'2012'],
            population.loc[:,'2009'],
            population.loc[:,'1980'],
            #Will become population density
            population.loc[:,'2016'],

            white_population.loc[:,'2015'],
            black_population.loc[:,'2015'],
            hispanic_population.loc[:,'2015'],
            asian_population.loc[:,'2015'],
            indian_population.loc[:,'2015'],
            youth_population.loc[:,'2015'],

            white_population.loc[:,'2009'],
            black_population.loc[:,'2009'],
            hispanic_population.loc[:,'2009'],
            asian_population.loc[:,'2009'],
            indian_population.loc[:,'2009'],
            youth_population.loc[:,'2009'],
            youth_population.loc[:,'2012'],

            unemployment_rate_1.loc[:,'2016 November'],
            unemployment_rate_1.loc[:,'2011 November'],
            unemployment_rate_1.loc[:,'2007 November'],

            median_age.loc[:,'2015'],
            bachelors_degrees.loc[:,'2012'],
            commute_time.loc[:,'2015'],
            food_stamps.loc[:,'2013'],
            homeownership_rate.loc[:,'2015'],
            income_inequality.loc[:,'2015'],

            business_establishments.loc[:,'2016 Q3'],
            business_establishments.loc[:,'2009 Q3'],

            rent_burdened.loc[:,'2015'],
            rent_burdened.loc[:,'2010'],

            evangelicals['Evangelical'],
            protestant['Protestant'],
            blackprotestant['BlackProtestant'],
            catholic['Catholic'],
            jewish['Jewish'],
            muslim['Muslim'],
            mormon['Mormon'],

            pd.DataFrame(climatechange[climatechange['GeoType']=='County']['human'].values,index=climatechange[climatechange['GeoType']=='County']['GEOID']),

            corn['Corn'],
            cotton['Cotton'],
            soybeans['Soybeans'],
            winter_wheat['Winter_wheat'],
            median_income.loc[:,'2014'],
            drugpoisoning[2015]
        ],axis=1)


        self.data.columns=[
            'GOPChange',
            'GOP2016',
            'GOP2012',

            #'AL_GOP2017',

            'Turnout2012',
            'Turnout2016',
            'VotesDem2016',
            'VotesGOP2016',
            'VotesDem2012',
            'VotesGOP2012',

            'Population2016',
            'Population2012',
            'Population2009',
            'Population1980',
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
            'Youth2012',

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
            'MedianIncome',
            'DrugPoisoning'
        ]

        #Not all counties have crops grown in them
        self.data['Corn'].fillna(0.0,inplace=True)
        self.data['Cotton'].fillna(0.0,inplace=True)
        self.data['Soybeans'].fillna(0.0,inplace=True)
        self.data['WinterWheat'].fillna(0.0,inplace=True)
        self.removeProblemCounties()
        #First, normalize by converting everything to percents, etc. Don't have to worry about training vs test data here
        #Turnout
        self.data['Turnout2012'] = (self.data['VotesDem2012'].add(self.data['VotesGOP2012'])).div(self.data['Population2012'])#-(self.data['Turnout2012'].div(100.0)).mul(self.data['Population2012']))
        self.data['Turnout2016'] = (self.data['VotesDem2016'].add(self.data['VotesGOP2016'])).div(self.data['Population2016'])#-(self.data['Turnout2016'].div(100.0)).mul(self.data['Population2016']))

        self.data['White2015'] = self.data['White2015'].div(self.data['Population2016'])
        self.data['Black2015'] = self.data['Black2015'].div(self.data['Population2016'])
        self.data['Hispanic2015'] = self.data['Hispanic2015'].div(self.data['Population2016'])
        self.data['Asian2015'] = self.data['Asian2015'].div(self.data['Population2016'])
        self.data['Indian2015'] = self.data['Indian2015'].div(self.data['Population2016'])
        self.data['Youth2015'] = self.data['Youth2015'].div(100.0)

        self.data['White2009'] = self.data['White2009'].div(self.data['Population2009'])
        self.data['Black2009'] = self.data['Black2009'].div(self.data['Population2009'])
        self.data['Hispanic2009'] = self.data['Hispanic2009'].div(self.data['Population2009'])
        self.data['Asian2009'] = self.data['Asian2009'].div(self.data['Population2009'])
        self.data['Indian2009'] = self.data['Indian2009'].div(self.data['Population2009'])
        self.data['Youth2009'] =self. data['Youth2009'].div(100.0)

        self.data['Unemployment2016'] = self.data['Unemployment2016'].div(100.0)
        self.data['Unemployment2007'] = self.data['Unemployment2007'].div(100.0)
        self.data['Unemployment2011'] = self.data['Unemployment2011'].div(100.0)

        self.data['IncomeInequality'] = self.data['IncomeInequality'].div(100.0)
        self.data['MedianAge'] = self.data['MedianAge'].div(100.0)

        self.data['RentBurdened2015'] = self.data['RentBurdened2015'].div(100.0)
        self.data['RentBurdened2010'] = self.data['RentBurdened2010'].div(100.0)

        self.data['Homeownership'] = self.data['Homeownership'].div(100.0)
        self.data['FoodStamps'] = self.data['FoodStamps'].div(self.data['Population2016'])
        self.data['Bachelors'] = self.data['Bachelors'].div(100.0)
        self.data['CommuteTime'] = self.data['CommuteTime'].div(60.0)

        self.data['Businesses2016'] = self.data['Businesses2016'].div(self.data['Population2016'])
        self.data['Businesses2009'] = self.data['Businesses2009'].div(self.data['Population2009'])

        self.data['ClimateChange'] = self.data['ClimateChange'].div(100.0)

        #Get population density (people per square kilometer)
        fileName = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
        for county, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
            fipsID = int(record.__dict__['attributes']['GEOID'])
            if fipsID in self.data.index:
                self.data['PopulationDensity2016'].loc[fipsID] = self.data['Population2016'].loc[fipsID]/(record.__dict__['attributes']['ALAND']/1000**2)
        self.data = self.data.dropna()

        file = open('database.pk1', 'wb')
        pickle.dump(self.data, file)
        file.close()

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
        If technique == 'statewise' (if you want to split training & test by state), the state(s) abbreviation as the test data. Here i is a list of abbreviations
        If technique == 'individual' (split each county separately as its own test set)
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
            testIndices = np.array([],'i')
            for state in i:
                testIndices= np.concatenate([testIndices,np.where((self.data.index>int(self.stateCodes[state])*1000) & (self.data.index<(int(self.stateCodes[state])+1)*1000))[0]])
            self.testData = self.data.iloc[testIndices]

            trainingIndices = range(len(self.data))
            for k in testIndices:
                trainingIndices.remove(k)
            self.trainingData = self.data.iloc[trainingIndices]

        if technique =='individual':
            self.testData = pd.DataFrame(self.data.iloc[i]).transpose()
            self.trainingData = self.data.iloc[[i != k for k in range(len(self.data.index))]]


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
            ])#StandardScaler()),])
        self.dataScaler = Pipeline([('std_scaler', MinMaxScaler()),])

    def prepData(self, labelName):
        """
        Define the features & labels of the training and test sets
        Then transform the data via the pipeline
        After this is run, the data should be ready for training
        """
        self.trainingFeatures = self.trainingData.drop(labelName, axis=1)
        self.trainingLabels = self.trainingData[labelName].copy().values
        #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #    print(self.trainingFeatures.head())
        self.trainingFeatures = self.dataPipeline.fit_transform(self.trainingFeatures)
        populationIndex = self.attributes.index('Population2016')

        #Repeat each county in the training set by the log10 of its population.
        for i in range(len(self.trainingFeatures)):
            numRepeats = int(np.log10(self.trainingFeatures[i,populationIndex]))
            repeatedFeatures = np.repeat(self.trainingFeatures[i,:],numRepeats).reshape(len(self.trainingFeatures[i,:]),numRepeats).transpose()
            repeatedLabels = np.repeat(self.trainingLabels[i],numRepeats).reshape(numRepeats,)
            self.trainingFeatures = np.concatenate([self.trainingFeatures, repeatedFeatures])
            self.trainingLabels = np.concatenate([self.trainingLabels, repeatedLabels])
        print(len(self.trainingLabels))

        self.trainingFeatures = self.dataScaler.fit_transform(self.trainingFeatures)

        if labelName in self.testData.columns:
            self.testFeatures = self.testData.drop(labelName, axis=1)
            self.testLabels = self.testData[labelName].copy()
            self.testFeatures = self.dataPipeline.transform(self.testFeatures)
            self.testFeatures = self.dataScaler.transform(self.testFeatures)

        else:
            self.testFeatures = self.dataPipeline.transform(self.testData)
            self.testFeatures = self.dataScaler.transform(self.testData)


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
        self.rms = np.sqrt(np.mean(self.residuals.values**2))
        return self.rms

    def bubblePlot(self, xValues, yValues, sValues, cValues, fileName, xLabel, yLabel, minXValue, minYValue, maxXValue, maxYValue, cMap = cm.seismic, scale = 'linear'):
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
        #plt.plot(np.linspace(minXValue, maxXValue, 100), np.linspace(minYValue, maxYValue, 100), linewidth=0.5, color='white',linestyle='--')

        plt.ylim([minYValue, maxYValue])
        plt.xlim([minXValue, maxXValue])
        plt.ylabel(yLabel)
        plt.xlabel(xLabel)
        plt.xscale(scale)
        fmtr = mtick.StrMethodFormatter('{x:,g}%')
        fmtr2 = mtick.StrMethodFormatter('{x:,g}%')

        ax.xaxis.set_major_formatter(fmtr2)
        ax.yaxis.set_major_formatter(fmtr)

        plt.savefig('plots/'+fileName+'.pdf',bbox_inches='tight')
        #plt.show()

    def countyPlot(self, dataSeries, dataMin, dataMax, vMin, vMax, pltTitle, cLabel, cMap = cm.seismic, AK_value = False, pltSupTitle = False):
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

        if pltSupTitle:
            plt.suptitle(pltSupTitle,x=0.9,y=0.35)
        plt.savefig('plots/'+str(pltTitle)+'.pdf',bbox_inches='tight')
        #plt.show()

    def subCountyPlot(self, dataSeries, dataMin, dataMax, vMin, vMax, pltTitle, cLabel, cMap = cm.seismic, AK_value = False, pltSupTitle = False):
        """
        Plots a map of the US, with each county colored by a data series
        dataSeries: a single-column Pandas DataFrame with the indices given by integer FIPS codes
        vMin, vMax: minimum and maximum of the colorbar- should correspond to the minima and maxima of the data
        cLabel: Label of the colorbar
        """
        fig = plt.figure(figsize=(10.0,7.0))
        #Mainland
        ax = plt.axes([0.0,0.0,1.0,0.85],projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0, cutoff=-20), aspect=1.15, frameon=False)
        ax.set_extent([-95.0,-77., 29.,43.])
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
                ax.add_geometries(state, crs=ccrs.Miller(), facecolor=faceColor, edgecolor=edgeColor, linewidth=lineWidth)
            else:
                #Missing data goes in gray
                faceColor = 'gray'
                #Is the county in Hawaii, Alaska, or the mainland?
                ax.add_geometries(state, crs=ccrs.Miller(), facecolor=faceColor, edgecolor=edgeColor, linewidth=lineWidth)

        #Mark a black line around each state
        fileName = 'cb_2015_us_state_5m/cb_2015_us_state_5m.shp'
        lineWidth = 0.25
        for state, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
            id = int(record.__dict__['attributes']['GEOID'])
            facecolor = cMap(0.0)
            ax.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor=edgeColor, linewidth=lineWidth)

        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)
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

        if pltSupTitle:
            plt.suptitle(pltSupTitle,x=0.9,y=0.35)
        plt.savefig('plots/'+str(pltTitle)+'.pdf',bbox_inches='tight')
        #plt.show()

    def statePlot(self, dataSeries, pltTitle, dataMin, dataMax, vMin, vMax, cLabel, cMap=cm.seismic):
        """
        Make a map of the counties in a particular state
        The value to plot is contained in the dataSeries, and the index of the dataSeries is the FIPS id
        """
        fig = plt.figure(figsize=(10.0,7.0))
        #Mainland
        extent = self.stateExtent({int(v):k for k,v in self.stateCodes.items()}[int(str(dataSeries.index[0])[:-3])])
        ax = plt.axes([0.0,0.0,1.0,1.0],projection=ccrs.LambertConformal(central_longitude=np.mean([extent[0], extent[1]]), central_latitude=np.mean([extent[2], extent[3]]), cutoff=-20), aspect=1.15, frameon=False)
        ax.set_extent(extent)
        edgeColor = 'black'
        #Color in the census tracts
        fileName = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
        lineWidth = 0.0
        for state, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
            id = int(record.__dict__['attributes']['GEOID'])
            #Shannon County was renamed to Oglala Dakota county, and its FIPS code was changed
            if id == '46102':
                id = '46113'

            if id in dataSeries.index:
                #Normalize the value so it matches the color mapping
                faceColor = cMap((dataSeries.loc[id]-dataMin)/(dataMax-dataMin))
                ax.add_geometries(state, crs=ccrs.Miller(), facecolor=faceColor, edgecolor=edgeColor, linewidth=lineWidth)

        #Mark a black line around the state
        fileName = 'cb_2015_us_state_5m/cb_2015_us_state_5m.shp'
        for state, record in zip(shpreader.Reader(fileName).geometries(), shpreader.Reader(fileName).records()):
            id = int(record.__dict__['attributes']['GEOID'])
            if id == int(str(dataSeries.index[0])[:-3]):
                ax.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor='black', linewidth=1.0)

        ax.background_patch.set_visible(False)
        ax.outline_patch.set_visible(False)

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

    def stateExtent(self, name):
        """
        For plotting a single state at a time... I haven't gone through every state yet
        Returns latitude, longitude to be around state of interest
        """
        if name =='AL':
            return [-89.0, -84.5, 29.0, 34.5]
        elif name == 'AZ':
            return [-115.7, -108.5, 30.0, 35.7]
        elif name == 'AR':
            return [-95.43, -89.37, 32.78, 35.2]
        elif name == 'CA':
            return [-125.3, -113.7, 30.5, 41.5]
        elif name == 'CO':
            return [-110.0, -101.3, 36.14, 41.36]
        elif name == 'CT':
            return [-74.12, -71.47, 38.8, 39.9]
        elif name == 'DE':
            return [-76.0, -75.0, 36.6, 38.0]
        elif name == 'FL':
            return [-88.0,-79., 23., 31.]
        elif name == 'GA':
            return [-86.1, -80.44, 30.06, 34.0]
        elif name == 'HI':
            return [-162.0, -152., 18.,23.]
        elif name == 'ID':
            return [-118.0, -110.5, 39.2, 45.7]
        elif name == 'IL':
            return [-92.24, -86.91, 36.74, 40.4]
        elif name == 'IN':
            return [-88.43, -84.53, 35.9, 41.95]
        elif name == 'IA':
            return [-96.98, -89.8, 40.14, 43.87]
        elif name == 'KS':
            return [-102.8, -94.2, 35.04, 38.3]
        elif name == 'NC':
            return [-85.0,-74., 32.,36.]
        elif name == 'PA':
            return [-81.0, -74., 37., 40.]
        else:
            return [-150, -70, 20, 45]
        """
        elif name == 'RI':
        elif name == 'SC':
        elif name == 'SD':
        elif name == 'TN':
        elif name == 'TX':
        elif name == 'UT':
        elif name == 'VT':
        elif name == 'VA':
        elif name == 'WA':
        elif name == 'WV':
        elif name == 'WI':
        elif name == 'WY':
        elif name == 'ND':
        elif name == 'OH':
        elif name == 'OK':
        elif name == 'OR':
        elif name == 'KY':
        elif name == 'LA':
        elif name == 'ME':
        elif name == 'MD':
        elif name == 'MA':
        elif name == 'MI':
        elif name == 'MN':
        elif name == 'MS':
        elif name == 'MO':
        elif name == 'MT':
        elif name == 'NE':
        elif name == 'NV':
        elif name == 'NH':
        elif name == 'NJ':
        elif name == 'NM':
        elif name == 'NY':
        """



def main():

    model = ElectionAnalysis(regressionType='Linear', numDivisions=50, attributes=attribs)
    model.loadData()
    model.countyPlot(dataSeries = model.data['DrugPoisoning'], dataMin = 0.0, dataMax = 30.0, vMin = 0.0, vMax = 30.0, pltTitle = 'DrugPoisoning2015', cLabel = 'Drug Poisoning Rate per 100,000 People', cMap = cm.magma, AK_value = False)
    input('wait for key')

    caLocations = np.where((model.data.index<7000) & (model.data.index>6000))
    #Plot California results
    model.statePlot(pd.Series(index=model.data.index[caLocations], data=model.data['GOP2016'][caLocations]),dataMin=0.0, dataMax=1.0, vMin=0.0, vMax=100.0, cLabel = 'GOP Vote Share', pltTitle='2017-07-28-Election-Neural-Net/National_vote_fraction2')

    pass
if __name__ == '__main__':
    main()
