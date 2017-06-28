#Boilerplate code to load data from each of data spreadsheets
#Each data series needs to be normalized in a slightly different way, hence a different function for each series
import xlrd
import numpy as np
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

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


def add_data_column(X, Y, fips_indices, filename, date, norm_scheme = 'none'):
    #Arguments: X: old data matrix
    #fips_indices: vector which contains the FIPS indices in the order of the output data
    #year: string of year/date of the data we want
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
    tot_filled_vals = 0
    #Ensure that an appreciable fraction (over 3000 out of 3140) of the counties have data for this year/date
    if len(np.nonzero(h.col_values(start_rowx=2, colx=spreadsheet_col))[0])>2500:
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
        X, Y, fips_indices = normalize_data(X, Y, fips_indices, data_col, norm_scheme, date)
        #Return the new data matrix X
        return X, Y, fips_indices
    else:
        #Return the old data matrix X
        print "Insufficient data for requested date- you only have " + str(len(np.nonzero(h.col_values(start_rowx=2, colx=spreadsheet_col))[0])) + " data points"
        return X, Y, fips_indices
        
#Here we normalize everything to [0, 1.0], and remove rows that are incomplete.
#There should be enough complete rows to get the neural net to be trained well
def normalize_data(X, Y, fips_indices, data_col, norm_scheme, pop = np.zeros((1000))):
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
        
    if norm_scheme == 'density':
        good_indices = []
        filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
        for record in shpreader.Reader(filename).records():
            shp_fips = int(record.__dict__['attributes']['GEOID'])
            if shp_fips in fips_indices:
                shp_area = record.__dict__['attributes']['ALAND']
                the_index = int(np.argmin(np.abs(fips_indices-shp_fips)))
                X[the_index,data_col] *= np.sqrt(1000.0/shp_area)
                good_indices.append(the_index)
        Y = Y[good_indices,:]
        fips_indices = fips_indices[good_indices]
        X = X[good_indices,:]
        
    Y = Y[np.nonzero(X[:,data_col])[0],:]
    fips_indices = fips_indices[np.nonzero(X[:,data_col])[0]]
    X = X[np.nonzero(X[:,data_col])[0],:]
        
    return X, Y, fips_indices
    
    
