#Boilerplate code to load data from each of data spreadsheets
#Each data series needs to be normalized in a slightly different way, hence a different function for each series
import numpy as np
import pandas as pd
import plotter
from matplotlib import cm
import cartopy.io.shapereader as shpreader

#tf.logging.set_verbosity(tf.logging.INFO)

#Load data from Excel, CSV files into pandas DataFrames

acres_to_m2 = 4047.

def load_data():
    print "Adding data from spreadsheets..."
    #Election data
    election_results = pd.read_excel('data_spreadsheets/US_County_Level_Presidential_Results_08-16.xls', 'Sheet 1', index_col=0, header=1, parse_cols=range(1,100), skiprows=range(2,31))
    #Data from St. Louis GeoFRED
    food_stamps = pd.read_excel('data_spreadsheets/food_stamps_1989_2013.xls','Sheet0', index_col=0, header=0, parse_cols=[2,3,7,9]+range(11,28))
    population = pd.read_excel('data_spreadsheets/population_1970_2016.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,50))
    #Population is in thousands in the spreadsheet, we need to multiply everything by 1000.0:
    population = pd.DataFrame(population.values*1000.0, index=population.index, columns=population.columns)
    
    youth_population = pd.read_excel('data_spreadsheets/youth_population_1989_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    black_population = pd.read_excel('data_spreadsheets/black_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    white_population = pd.read_excel('data_spreadsheets/white_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    asian_population = pd.read_excel('data_spreadsheets/asian_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    indian_population = pd.read_excel('data_spreadsheets/indian_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    hispanic_population = pd.read_excel('data_spreadsheets/hispanic_population_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    commute_time = pd.read_excel('data_spreadsheets/commuting_time_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,50))
    unemployment_rate_0 = pd.read_excel('data_spreadsheets/unemployment_rate_1970_2017.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,400))
    unemployment_rate_1 = pd.read_excel('data_spreadsheets/unemployment_rate_1970_2017.xls','Sheet1', index_col=0, header=1, skiprows=0, parse_cols=range(2,400))
    bachelors_degrees = pd.read_excel('data_spreadsheets/bachelors_2010_2012.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    median_age = pd.read_excel('data_spreadsheets/median_age_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    rent_burdened = pd.read_excel('data_spreadsheets/rent_burdened_2010_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    homeownership_rate = pd.read_excel('data_spreadsheets/homeownership_rate_2009_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    income_inequality = pd.read_excel('data_spreadsheets/income_inequality_2010_2015.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,100))
    business_establishments = pd.read_excel('data_spreadsheets/business_establishments_1990_2016.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,200))
    median_income = pd.read_excel('data_spreadsheets/median_income_1989_2014.xls','Sheet0', index_col=0, header=1, skiprows=0, parse_cols=range(2,200))
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
    
    #Agricultural data:
    
    #Get county areas, in m^2
    filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
    geometries = pd.DataFrame([y[0].__dict__['attributes'] for y in zip(shpreader.Reader(filename).records())])
    geometries.index = map(int,map(str,geometries['GEOID']))

    #Corn planted
    corn_planted = pd.read_csv('data_spreadsheets/corn_planted_2016.csv')
    corn_planted = corn_planted[np.isfinite(corn_planted['County ANSI'])]
    corn_planted = corn_planted[np.isfinite(corn_planted['State ANSI'])]
    acres_planted = map(str,corn_planted['Value'].values)
    corn_indices = [None]*len(acres_planted)
    corn_planted.index = range(len(corn_planted))
    for i in range(len(acres_planted)):
        if ',' in acres_planted[i]:
            acres_planted[i] = acres_to_m2*float(acres_planted[i].split(',')[0]+acres_planted[i].split(',')[1])
        else:
            acres_planted[i] = acres_to_m2*float(acres_planted[i])
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
    acres_planted = map(str,cotton_planted['Value'].values)
    cotton_indices = [None]*len(acres_planted)
    cotton_planted.index = range(len(cotton_planted))
    for i in range(len(acres_planted)):
        if ',' in acres_planted[i]:
            acres_planted[i] = acres_to_m2*float(acres_planted[i].split(',')[0]+acres_planted[i].split(',')[1])
        else:
            acres_planted[i] = acres_to_m2*float(acres_planted[i])
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
    acres_planted = map(str,soybeans_planted['Value'].values)
    soybeans_indices = [None]*len(acres_planted)
    soybeans_planted.index = range(len(soybeans_planted))
    for i in range(len(acres_planted)):
        if ',' in acres_planted[i]:
            acres_planted[i] = acres_to_m2*float(acres_planted[i].split(',')[0]+acres_planted[i].split(',')[1])
        else:
            acres_planted[i] = acres_to_m2*float(acres_planted[i])
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
    acres_planted = map(str,winter_wheat_planted['Value'].values)
    winter_wheat_indices = [None]*len(acres_planted)
    winter_wheat_planted.index = range(len(winter_wheat_planted))
    for i in range(len(acres_planted)):
        if ',' in acres_planted[i]:
            acres_planted[i] = acres_to_m2*float(acres_planted[i].split(',')[0]+acres_planted[i].split(',')[1])
        else:
            acres_planted[i] = acres_to_m2*float(acres_planted[i])
        if len(str(int(float(winter_wheat_planted.loc[i,'County ANSI']))))>2:
            winter_wheat_indices[i] = int(float(str(winter_wheat_planted.loc[i,'State ANSI'])+str(winter_wheat_planted.loc[i,'County ANSI'])))
        elif len(str(int(float(winter_wheat_planted.loc[i,'County ANSI']))))>1:
            winter_wheat_indices[i] = int(float(str(winter_wheat_planted.loc[i,'State ANSI'])+'0'+str(winter_wheat_planted.loc[i,'County ANSI'])))
        else:
            winter_wheat_indices[i] = int(float(str(winter_wheat_planted.loc[i,'State ANSI'])+'00'+str(winter_wheat_planted.loc[i,'County ANSI'])))
        
    winter_wheat = pd.DataFrame(acres_planted, index=winter_wheat_indices,columns=['Winter_wheat'])

    for winter_wheat_index,winter_wheat_row in winter_wheat.iterrows():
        winter_wheat.loc[winter_wheat_index,'Winter_wheat'] = winter_wheat_row[0]/geometries.loc[winter_wheat_index]['ALAND']
    print "Done!"


    #Combine relevant data into a single DataFrame
    data= pd.concat([
        election_results.loc[:,'per_gop_2016']-election_results.loc[:,'per_gop_2012'],
        election_results.loc[:,'per_gop_2016'],
        election_results.loc[:,'per_gop_2012'],
        
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
        median_income.loc[:,'2014']
    ],axis=1)
    data.columns=[
        'GOPChange',
        'GOP2016',
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
        'Population1980',
    
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
    
        
    #Not all counties have crops grown in them
    data['Corn'].fillna(0.0,inplace=True)
    data['Cotton'].fillna(0.0,inplace=True)
    data['Soybeans'].fillna(0.0,inplace=True)
    data['WinterWheat'].fillna(0.0,inplace=True)
    data = remove_problem_counties(data)
    
    #First, normalize by converting everything to percents, etc. Don't have to worry about training vs test data here
    #Turnout
    data['Turnout2012'] = (data['VotesDem2012'].add(data['VotesGOP2012'])).div(data['Population2012']-data['Turnout2012'])
    data['Turnout2016'] = (data['VotesDem2016'].add(data['VotesGOP2016'])).div(data['Population2016']-data['Turnout2016'])
    
    data['White2015'] = data['White2015'].div(data['Population2016'])
    data['Black2015'] = data['Black2015'].div(data['Population2016'])
    data['Hispanic2015'] = data['Hispanic2015'].div(data['Population2016'])
    data['Asian2015'] = data['Asian2015'].div(data['Population2016'])
    data['Indian2015'] = data['Indian2015'].div(data['Population2016'])
    data['Youth2015'] = data['Youth2015'].div(data['Population2016'])

    data['White2009'] = data['White2009'].div(data['Population2009'])
    data['Black2009'] = data['Black2009'].div(data['Population2009'])
    data['Hispanic2009'] = data['Hispanic2009'].div(data['Population2009'])
    data['Asian2009'] = data['Asian2009'].div(data['Population2009'])
    data['Indian2009'] = data['Indian2009'].div(data['Population2009'])
    data['Youth2009'] = data['Youth2009'].div(data['Population2009'])
    
    data['Unemployment2016'] = data['Unemployment2016'].div(100.0)
    data['Unemployment2007'] = data['Unemployment2007'].div(100.0)
    data['Unemployment2011'] = data['Unemployment2011'].div(100.0)

    data['IncomeInequality'] = data['IncomeInequality'].div(100.0)
    data['MedianAge'] = data['MedianAge'].div(100.0)
    
    data['RentBurdened2015'] = data['RentBurdened2015'].div(100.0)
    data['RentBurdened2010'] = data['RentBurdened2010'].div(100.0)

    data['Homeownership'] = data['Homeownership'].div(100.0)
    data['FoodStamps'] = data['FoodStamps'].div(data['Population2016'])
    data['Bachelors'] = data['Bachelors'].div(100.0)
    data['CommuteTime'] = data['CommuteTime'].div(60.0)

    data['Businesses2016'] = data['Businesses2016'].div(data['Population2016'])
    data['Businesses2009'] = data['Businesses2009'].div(data['Population2009'])
    
    data['ClimateChange'] = data['ClimateChange'].div(100.0)
    return data
    
def remove_problem_counties(data):
    if 46103 in data.index:
        data = data.drop(46103)
    if 46105 in data.index:
        data = data.drop(46105)        
    if 46109 in data.index:
        data = data.drop(46109)
    if 46111 in data.index:
        data = data.drop(46111)
    if 46102 in data.index:
        data = data.drop(46102)
    return data
    
def clean_data(data):
    #Remove problem counties in South Dakota

    #Drop NaN values
    data = data.dropna(how='any')
    return data
