#Boilerplate code to load data from each of data spreadsheets
#Each data series needs to be normalized in a slightly different way, hence a different function for each series
import numpy as np
import pandas as pd
import plotter
from matplotlib import cm

#tf.logging.set_verbosity(tf.logging.INFO)

#Load data from Excel, CSV files into pandas DataFrames

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
    
    #Religion data from CSV files
    religion = pd.read_stata('data_spreadsheets/religion.dta')
    religion = religion.fillna(0.0)
    evangelicals = pd.DataFrame(religion.loc[:,'evanadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Evangelical'])
    #plotter.national_plot(evangelicals,data_min=0.0, data_max=0.85, vmin=0.0, vmax=85.0, plt_title='Evangelical',colorbar_label='Evangelical Protestant [\%]', use_cmap=cm.Reds, AK_value=False)
    
    protestant = pd.DataFrame(religion.loc[:,'mprtadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Protestant'])
    blackprotestant = pd.DataFrame(religion.loc[:,'bprtadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['BlackProtestant'])
    catholic = pd.DataFrame(religion.loc[:,'cathadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Catholic'])
    #plotter.national_plot(catholic,data_min=0.0, data_max=0.3, vmin=0.0, vmax=30.0, plt_title='Catholic',colorbar_label='Catholic [\%]', use_cmap=cm.BuPu, AK_value=False)
    
    #Multiple types of Judaism
    jewish = pd.DataFrame( religion.loc[:,'cjudadh'].div( religion.loc[:,'POP2010']).values,index= religion.loc[:,'fips'],columns=['Jewish'])
    jewish.add(pd.DataFrame( religion.loc[:,'ojudadh'].div( religion.loc[:,'POP2010']).values,index= religion.loc[:,'fips'],columns=['Jewish']))
    jewish.add(pd.DataFrame( religion.loc[:,'rjudadh'].div( religion.loc[:,'POP2010']).values,index= religion.loc[:,'fips'],columns=['Jewish']))
    jewish.add(pd.DataFrame( religion.loc[:,'rfrmadh'].div(religion.loc[:,'POP2010']).values,index= religion.loc[:,'fips'],columns=['Jewish']))
    
    muslim = pd.DataFrame(religion.loc[:,'mslmadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Muslim'])
    #plotter.national_plot(muslim,data_min=0.0, data_max=0.1, vmin=0.0, vmax=10.0, plt_title='Muslim',colorbar_label='Muslim [\%]', use_cmap=cm.Greens, AK_value=False)
    
    mormon = pd.DataFrame(religion.loc[:,'ldsadh'].div(religion.loc[:,'POP2010']).values,index=religion.loc[:,'fips'],columns=['Mormon'])
    #plotter.national_plot(mormon,data_min=0.0, data_max=0.75, vmin=0.0, vmax=75.0, plt_title='Mormons',colorbar_label='Mormon [\%]', use_cmap=cm.Blues, AK_value=False)
    
    #protestant = pd.DataFrame(map(float,religion[religion.columns[1061]][1:].values),index=map(int,religion['FIPS'][1:]),columns=['Protestant'])
    #blackprotestant = pd.DataFrame(map(float,religion[religion.columns[1184]][1:].values),index=map(int,religion['FIPS'][1:]),columns=['BlackProtestant'])
    #catholic = pd.DataFrame(map(float,religion[religion.columns[1310]][1:].values),index=map(int,religion['FIPS'][1:]),columns=['Catholic'])
    #jewish = pd.DataFrame(map(float,religion[religion.columns[1376]][1:].values),index=map(int,religion['FIPS'][1:]),columns=['Jewish'])
    #muslim = pd.DataFrame(map(float,religion[religion.columns[1337]][1:].values),index=map(int,religion['FIPS'][1:]),columns=['Muslim'])
    #mormon = pd.DataFrame(map(float,religion[religion.columns[1369]][1:].values),index=map(int,religion['FIPS'][1:]),columns=['Mormon'])
    print "Done!"


    #Combine relevant data into a single DataFrame
    data= pd.concat([
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
        unemployment_rate_1.loc[:,'2007 November'],
        unemployment_rate_1.loc[:,'2004 November'],
    
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
        mormon['Mormon']
    ],axis=1)

    data.columns=[
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
    
    #Remove problem counties
    if 46103 in data.index:
        data = data.drop(46103)
    if 46105 in data.index:
        data = data.drop(46105)        
    if 46109 in data.index:
        data = data.drop(46109)
    if 46111 in data.index:
        data = data.drop(46111)
        
    #Drop NaN values
    data = data.dropna(how='any')
    
    
    
    #Normalize columns:
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
    data['Unemployment2004'] = data['Unemployment2004'].div(100.0)

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

    data['Population2016'] = data['Population2016'].div(np.max(data['Population2016'].values))
    data['Population2012'] = data['Population2012'].div(np.max(data['Population2012'].values))
    data['Population2009'] = data['Population2009'].div(np.max(data['Population2009'].values))


    return data

def fips_exceptions(fips_id):
    #A number of counties have been rearranged over time or are otherwise weird
    #e.g. Kalowao county in Hawaii is tiny and barely populated
    #Bedford City, VA was independent, but merged with Bedford County in 2013
    #Shannon County, SD was renamed Oglala Dakota County, and it's entirely within an Indian preservation so lots of data is missing
    #Some counties in Alaska are also barely populated
    old_ids = ['51515', '15005','2275']
    #new_ids = ['51019', '02158', '46102', '02158']
    if str(int(fips_id)) in old_ids:
        return 1
    else:
        return 0


def add_data_column(X, Y, fips_indices, filename, date, data_col, norm_scheme = 'none', norm_factor=1.0):
    #Arguments: X: old data matrix
    #fips_indices: vector which contains the FIPS indices in the order of the output data
    #year: string of year/date of the data we want
    #data_col: which column we add data to in X
    
    print "Adding data: " + str(filename)
    
    #Find the correct column in the spreadsheet- e.g. if you want data from 2015, find the column labeled '2015'
    g = xlrd.open_workbook('data_spreadsheets/'+filename)
    found_col = False
    j = 0
    while not found_col:
        if j>2:
            print "Requested date not found!"
            return X, Y, fips_indices
        h = g.sheet_by_index(j)
        sheet_keys = map(str, h.row_values(start_colx = 3, rowx=1))
        sheet_values = map(int,range(3,3+len(h.row_values(start_colx = 3, rowx=1))))
        sheet_dict = dict(zip(sheet_keys, sheet_values))
        if str(date) in sheet_dict:
            spreadsheet_col = sheet_dict[str(date)]
            found_col = True
        else:
            j += 1
    
    
    tot_filled_vals = 0
    
    #Load the data into X, county-by-county. Loop through the array fips_indices so that we don't try to find a county that isn't there
    j = 0
    found_rows = np.zeros((4000))-0.01
    spreadsheet_fips = map(int,h.col_values(start_rowx=2, colx=2))
    spreadsheet_values = h.col_values(start_rowx=2, colx=spreadsheet_col)
    for fips_index in map(int,fips_indices):
        if fips_index in spreadsheet_fips:
            spreadsheet_row = spreadsheet_fips.index(fips_index)
            
            X_row = map(int,fips_indices).index(fips_index)
            value = spreadsheet_values[spreadsheet_row]
            if value != '':
                X[X_row, data_col] = value
                found_rows[j] = X_row
                j += 1
    found_rows = found_rows[np.nonzero(found_rows+0.01)]
    found_rows=map(int,found_rows)
    """
    for value in h.col_values(start_rowx=2, colx=spreadsheet_col):
        spreadsheet_fips = int(h.col_values(start_rowx=2, colx=2)[j])
        #If the FIPS code in the spreadsheet matches the FIPS code of the election data, everything is good
        if spreadsheet_fips in map(int,fips_indices):
            X_row = map(int,fips_indices).index(spreadsheet_fips)
            if value == '':
                #Missing data gets a value of -0.01 to keep track of it
                X[X_row, data_col] = -0.01
            else:
                X[X_row, data_col] = value
                tot_filled_vals += 1
            j += 1
        #If there's a weird exception, ignore that county data                 
        elif fips_exceptions(spreadsheet_fips):
            j += 1
        else:
            j += 1
    #Remove rows that were missing:
    Y = Y[np.nonzero(X[:,data_col]!=-0.01)]
    fips_indices = fips_indices[np.nonzero(X[:,data_col]!=-0.01)]
    X = X[np.nonzero(X[:,data_col]!=-0.01)]
    """
    #Normalize the column in an appropriate way
    if norm_scheme:
        X, Y, fips_indices = normalize_data(X, Y, fips_indices, data_col, norm_scheme, norm_factor)
    #Return the new data matrix X
    return X[found_rows], Y[found_rows], fips_indices[found_rows]
        
#Lots of counties don't grow particular crops, so we assume if it's not in the spreadsheet, that county doesn't grow any
#Return fraction of the county land that is used for growing each crop
def add_agriculture_data(X,Y,fips_indices, filename, norm_schme='none'):
    print "Adding data: " + str(filename)
    #Find the sheet & column in the data file corresponding to the requested year/date
    g = xlrd.open_workbook('data_spreadsheets/'+filename)
    found_col = False
    j = 0
    while not found_col:
        if j>2:
            print "Requested date not found!"
            return X, Y, fips_indices
        h = g.sheet_by_index(j)
        sheet_keys = map(str, h.row_values(start_colx = 3, rowx=1))
        sheet_values = map(int,range(3,3+len(h.row_values(start_colx = 3, rowx=1))))
        sheet_dict = dict(zip(sheet_keys, sheet_values))
        if str(date) in sheet_dict:
            spreadsheet_col = sheet_dict[str(date)]
            found_col = True
        else:
            j += 1
            
    #Find the next available column in the data matrix X
    data_col = 0
    while len(np.nonzero(X[:,data_col])[0])>0:
        data_col += 1
    
    #Load the data into X, county-by-county. j loops through the spreadsheet rows
    j = 0
    for value in h.col_values(start_rowx=2, colx=spreadsheet_col):
        spreadsheet_fips = int(h.col_values(start_rowx=2, colx=2)[j])
        #If the FIPS code in the spreadsheet matches the FIPS code of the election data, everything is good
        if spreadsheet_fips in map(int,fips_indices):
            X_row = map(int,fips_indices).index(spreadsheet_fips)
            if value == '':
                #Missing data gets a value of 0
                X[X_row, data_col] = 0.0
            else:
                if value == 0.0:
                    X[X_row, data_col] = 0.01
                else:
                    X[X_row, data_col] = value
                tot_filled_vals += 1
            j += 1
        #If there's a weird exception, ignore that county data                 
        elif fips_exceptions(spreadsheet_fips):
            j += 1
        else:
            j += 1
    #Normalize the column in an appropriate way
    X, Y, fips_indices = normalize_data(X, Y, fips_indices, data_col, norm_scheme, norm_factor)
    #Return the new data matrix X
    return X, Y, fips_indices

        
def add_religion_data(X,Y,fips_indices):
    file = open('data_spreadsheets/religion.pk1','rb')
    g = pickle.load(file)
    file.close()
    #7 major religions
    start_col = np.max(np.nonzero(X[0,:]))+1
    print start_col
    print start_col +7
    for fips_index in fips_indices:
        row = np.argmin(np.abs(g[:,0]-fips_index))
        X[np.argmin(np.abs(fips_indices-fips_index)),start_col:start_col+7] = g[row,1:]*0.01
    return X,Y, fips_indices
    
#Here we normalize everything to [0, 1.0], and remove rows that are incomplete.
#There should be enough complete rows to get the neural net to be trained well
def normalize_data(X, Y, fips_indices, data_col, norm_scheme, norm_factor=1.0, pop = np.zeros((1000))):
    #Commute times are listed in minutes- report as fraction of an hour
    if norm_scheme =='fraction_of_hour':
        X[:,data_col] *= 1.0/60.0
    if norm_scheme == 'percent':
        X[:,data_col] *= 1.0/100.0
        
    if norm_scheme == 'population':
        X[:,data_col] *= 1.0/pop
        
    if norm_scheme == 'population1000':
        X[:,data_col] *= 0.001/pop
        
    if norm_scheme == 'Y_population':
        Y[:,data_col] *= 1.0/pop
        
    if norm_scheme == 'subtraction':
        X[:,data_col] = 1000.*pop-X[:,data_col]

    if norm_scheme == 'none':
        X[:,data_col] *= 1.0
        
    if norm_scheme == 'migration':
        X[:,data_col] *= 0.001/pop
        X[:,data_col] = (X[:,data_col]-np.min(X[:,data_col]))/np.max(X[:,data_col])
        
    #Spread everything out between 0 and 1
    if norm_scheme =='renormalize':
        #print "Min of data: " + str(np.min(X[:,data_col]))
        #print "Max of data: " + str(np.max(X[:,data_col]))
        X[:,data_col] = (X[:,data_col]-np.min(X[:,data_col]))/(np.max(X[:,data_col])-np.min(X[:,data_col]))
        #print "Min of renormalized data: " + str(np.min(X[:,data_col]))
        #print "Max of renormalized data: " + str(np.max(X[:,data_col]))
        #print np.histogram(X[:,data_col], bins=np.linspace(-0.0, 1.0, 50))
        
    if norm_scheme == 'density':
        good_indices = []
        filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
        for record in shpreader.Reader(filename).records():
            shp_fips = int(record.__dict__['attributes']['GEOID'])
            if shp_fips in fips_indices:
                shp_area = record.__dict__['attributes']['ALAND']
                the_index = int(np.argmin(np.abs(fips_indices-shp_fips)))
                if shp_area == 0.0:
                    print record
                X[the_index,data_col] *= float(norm_factor)/float(shp_area)
                good_indices.append(the_index)
        Y = Y[good_indices,:]
        fips_indices = fips_indices[good_indices]
        X = X[good_indices,:]
        
    #Y = Y[np.nonzero(X[:,data_col])[0],:]
    #fips_indices = fips_indices[np.nonzero(X[:,data_col])[0]]
    #X = X[np.nonzero(X[:,data_col])[0],:]
        
    return X, Y, fips_indices
    
    
