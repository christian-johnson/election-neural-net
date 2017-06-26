from sklearn.neural_network import MLPRegressor as mlp
import numpy as np
import xlrd
import pickle

from matplotlib import cm
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader

import load_data as ld


def state_extent(name):
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
    """   
    """
    elif name == 'ND':
    elif name == 'OH':
    elif name == 'OK':
    elif name == 'OR':
    """
    """
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
    
    #Change to be around state of interest
    if name =='AL':
        return [-89.0, -84.5, 30.0, 35.4]
    elif name == 'AZ':
        return [-115.7, -108.5, 30.0, 35.7]
    elif name == 'AR':
        return [-95.43, -89.37, 32.78, 35.2]
    elif name == 'CA':
        return [-125.3, -113.7, 32.0, 40.0]
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
    
def state_plot_data_model(fips_indices, test_outputs, test_results, name):
    print name
    fig = plt.figure(figsize=(12.0, 5.0))
    ax1 = plt.axes([0.0, 0.0, 0.45, 1.0],projection=ccrs.Miller(), aspect=1.3)
    ax2 = plt.axes([0.45, 0.0, 0.45, 1.0],projection=ccrs.Miller(), aspect=1.3)
    ax1.set_extent(state_extent(name))
    ax2.set_extent(state_extent(name))

    data = test_results
    filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
    for state, record in zip(shpreader.Reader(filename).geometries(), shpreader.Reader(filename).records()):
        id = str(int(record.__dict__['attributes']['GEOID']))
        if id in dict(zip(map(str, map(int, fips_indices)),data)):
            min_val = -0.45
            max_val = 0.45
            data = test_outputs-0.5
            value = (dict(zip(map(str, map(int,fips_indices)),data))[id]-min_val)/(max_val-min_val)
            ax1.add_geometries(state, crs=ccrs.Miller(), facecolor=cm.bwr(value), edgecolor='black', linewidth=0.2)
            
            min_val = -0.45
            max_val = 0.45
            data = test_results-0.5
            value = (dict(zip(map(str, map(int,fips_indices)),data))[id]-min_val)/(max_val-min_val)
            ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=cm.bwr(value), edgecolor='black', linewidth=0.2)

    print "Plotting..."
    ax1.background_patch.set_visible(False)
    ax1.outline_patch.set_visible(False)
    ax2.background_patch.set_visible(False)
    ax2.outline_patch.set_visible(False)
    ax1.set_title('Data')
    ax2.set_title('Model')
    #Add colorbar
    axc = plt.axes([0.9, 0.25, 0.02, 0.5], frameon=False)
    cmap = cm.seismic
    norm = mpl.colors.Normalize(vmin=-45.0, vmax=+45.0)
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap,norm=norm,orientation='vertical')
    cb.set_label('Vote Margin [%]')
    
    #fig.suptitle('2016 US Presidential Election Results Residual: '+name)
    plt.savefig('plots/election_data_model_'+name+'.pdf',bbox_inches='tight')
    plt.show()


def state_plot_residual(fips_indices, test_outputs, test_results, name, X, Y, training_results, all_fips_indices):
    print name
    fig = plt.figure(figsize=(12.0, 7.0))
    ax1 = plt.axes([0.0, 0.0, 0.45, 1.0],projection=ccrs.Miller(), aspect=1.3)
    ax2 = plt.axes([0.45, 0.0, 0.45, 1.0],projection=ccrs.Miller(), aspect=1.3)
    ax1.set_extent(state_extent(name))
    ax2.set_extent(state_extent(name))
    
    data = test_results
    filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
    for state, record in zip(shpreader.Reader(filename).geometries(), shpreader.Reader(filename).records()):
        id = str(int(record.__dict__['attributes']['GEOID']))
        if id in dict(zip(map(str, map(int, fips_indices)),data)):

            min_val = -0.2
            max_val = 0.2
            data = test_outputs-test_results
            value = (dict(zip(map(str, map(int,fips_indices)),data))[id]-min_val)/(max_val-min_val)
            ax1.add_geometries(state, crs=ccrs.Miller(), facecolor=cm.bwr(value), edgecolor='black', linewidth=0.2)
            
            min_val = -5.0
            max_val = 5.0
            data = sigma(test_outputs-test_results, fips_indices, X, Y, training_results, all_fips_indices)
            value = (dict(zip(map(str, map(int,fips_indices)),data))[id]-min_val)/(max_val-min_val)
            ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=cm.bwr(value), edgecolor='black', linewidth=0.2)
            
    print "Plotting..."
    ax1.background_patch.set_visible(False)
    ax1.outline_patch.set_visible(False)
    ax2.background_patch.set_visible(False)
    ax2.outline_patch.set_visible(False)
    
    #Add colorbar
    axc = plt.axes([0.85, 0.25, 0.02, 0.5], frameon=False)
    cmap = cm.seismic
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap,norm=norm,orientation='vertical')
    cb.set_label('GOP Vote Fraction [%]')
    
    #fig.suptitle('2016 US Presidential Election Results Residual: '+name)
    plt.savefig('plots/election_residuals_'+name+'.pdf',bbox_inches='tight')
    plt.show()

#For sklearn, we need the training data organized as a numpy array, with rows = counties and columns = data points
#Ideally we will normalize everything to be within [-1, +1], and set missing values to be 0
#Strategy: loop through each data file (i.e. loop through columns)

#Return array of significances for each county, given the spread of uncertainty in similar counties
def sigma(resid, test_fips, X, Y, training_results, fips_indices):
    sigmas = np.zeros((len(resid)))
    j = 0
    for county_fips in test_fips:
        chi2_vals = np.zeros((3140))
        #Calculate chi2 between county inputs and the rest of the country
        county_row = np.argmin(np.abs(fips_indices-county_fips))
        chi2_vals = np.sum((X-X[county_row,:])**2, axis=1)
        training_indices = np.argsort(chi2_vals[np.nonzero(chi2_vals)])[:15]
        sigmas[j] = resid[j]/np.std(Y[training_indices]-training_results[training_indices])
        j += 1
    return sigmas


#Define plotting function, so we can see the results in a nice way
def national_plot(fips_indices, data):
    power = 0
    #data_dict = dict(zip(map(str,map(int,data[:,0])),map(float,data[:,45])))
    fig = plt.figure(figsize=(14.0,6.3))
    #Mainland
    ax = plt.axes([0.25,0,0.75,1],projection=ccrs.Miller(), aspect=1.3, frameon=False)
    ax.set_extent([-127.0,-65., 23.,43.])
    #Alaska
    ax2 = plt.axes([0,0.3,0.27,0.6],projection=ccrs.Miller(), aspect=1.7, frameon=False)
    ax2.set_extent([-180.0,-132., 47.,62.])
    #Hawaii
    ax3 = plt.axes([0.1, 0.05,0.2,0.3],projection=ccrs.Miller(), aspect=1.3, frameon=False)
    ax3.set_extent([-162.0,-152., 18.,23.])
    filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
    
    for state, record in zip(shpreader.Reader(filename).geometries(), shpreader.Reader(filename).records()):
        id = str(int(record.__dict__['attributes']['GEOID']))
        if id == '46102':
            id = '46113'
        if id in dict(zip(map(str, map(int, fips_indices)),data)):
            value = dict(zip(map(str, map(int,fips_indices)),data))[id]
            facecolor = cm.seismic(value)
            edgecolor = 'black'
            #Is the county in Hawaii, Alaska, or the mainland?
            if int(record.__dict__['attributes']['GEOID'])<2991 and int(record.__dict__['attributes']['GEOID'])>2013: 
                ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=0.2)
            elif int(record.__dict__['attributes']['GEOID'])<15010 and int(record.__dict__['attributes']['GEOID'])>15000: 
                ax3.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=0.2)
            else:
                ax.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=0.2)
        else:
            facecolor = 'gray'
            edgecolor = 'black'
            #Is the county in Hawaii, Alaska, or the mainland?
            if int(record.__dict__['attributes']['GEOID'])<2991 and int(record.__dict__['attributes']['GEOID'])>2013: 
                ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=cm.seismic(0.5288700180057424), edgecolor=edgecolor, linewidth=0.2)
            elif int(record.__dict__['attributes']['GEOID'])<15010 and int(record.__dict__['attributes']['GEOID'])>15000: 
                ax3.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=0.2)
            else:
                ax.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=0.2)                    
    print "Plotting..."
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)
    ax2.background_patch.set_visible(False)
    ax2.outline_patch.set_visible(False)
    ax3.background_patch.set_visible(False)
    ax3.outline_patch.set_visible(False)

    #Add colorbar
    axc = plt.axes([0.93, 0.1, 0.02, 0.5], frameon=False)
    cmap = cm.seismic
    norm = mpl.colors.Normalize(vmin=-100, vmax=100)
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap,norm=norm,orientation='vertical')
    cb.set_label('Vote Margin [%]')
    
    plt.savefig('plots/election_results.pdf',bbox_inches='tight')
    #plt.show()
    return 0

def state_plot(fips_indices, data, column, name, min_val, max_val):
    print name
    fig = plt.figure(figsize=(7.0, 7.0))
    ax = plt.axes(projection=ccrs.Miller(), aspect=1.3)
    #Change to be around state of interest
    if name=='PA':
        ax.set_extent([-81.0, -74., 37., 40.])
    elif name == 'FL':
        ax.set_extent([-88.0,-79., 23.,30.])
    elif name == 'NC':
        ax.set_extent([-85.0,-74., 32.,36.])
    else:
        ax.set_extent([-150, -70, 20, 45])
    filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
    for state, record in zip(shpreader.Reader(filename).geometries(), shpreader.Reader(filename).records()):
        id = str(int(record.__dict__['attributes']['GEOID']))
        if id in dict(zip(map(str, map(int, fips_indices)),data)):
            value = (dict(zip(map(str, map(int,fips_indices)),data))[id]-min_val)/(max_val-min_val)
            facecolor = cm.seismic(value)
            edgecolor = 'black'
            ax.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=0.2)
                             
    print "Plotting..."
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)

    #Add colorbar
    axc = plt.axes([0.85, 0.25, 0.02, 0.5], frameon=False)
    cmap = cm.seismic
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap,norm=norm,orientation='vertical')
    cb.set_label('Vote Residual [%]')
    
    fig.suptitle('2016 US Presidential Election Results Residual: '+name)
    plt.savefig('plots/election_residual_'+name+'.pdf',bbox_inches='tight')
    plt.show()
    return 0

#For sklearn, we need the training data organized as a numpy array, with rows = counties and columns = data points
#Ideally we will normalize everything to be within [-1, +1], and set missing values to be 0
#Strategy: loop through each data file (i.e. loop through columns)

#The data files from St. Louis Federal Reserve
datafiles = [
'commuting_time_2009_2015.xls',
'indian_population_2009_2015.xls',
'single_parent_2009_2015.xls',
'asian_population_2009_2015.xls',
'crime_rate_2005_2015.xls',
'median_age_2009_2015.xls',
'subprime_percent_1999_2016.xls',
'associates_degree_2009_2015.xls',
'disconnected_youth_2009_2015.xls',
'median_income_1989_2014.xls',
'unemployment_rate_1970_2017.xls',
'bachelors_2010_2012.xls',
'food_stamps_1989_2013.xls',
'net_migration_2009_2013.xls',
'white_population_2009_2015.xls',
'black_population_2009_2015.xls',
'hispanic_population_2009_2015.xls',
'population_1970_2016.xls',
'building_permits_1990_2015.xls',
'homeownership_rate_2009_2015.xls',
'poverty_rate_1989_2014.xls',
'business_establishments_1990_2016.xls',
'income_inequality_2010_2015.xls',
'rent_burdened_2010_2015.xls']

    

#Election results come from https://github.com/tonmcg/County_Level_Election_Results_12-16
#Caveat: for some reason Alaska doesn't report county-by-county data
#We'll have to make a few assumptions on how the counties of Alaska voted
reload_data = True
if reload_data:
    #Column 0 of Y contains the GOP vote fraction, i.e. GOP/(GOP+DEM)
    #Column 1 of Y contains the total GOP+DEM # of votes
    #From these numbers, you should be able to calculate the number of DEM votes or GOP votes
    Y = np.zeros((3140,2))
    g = xlrd.open_workbook('US_County_Level_Presidential_Results_08-16.xls')
    h = g.sheet_by_index(0)
    #We are going to ignore Alaska here because it doesn't report county-level results
    fips_indices = np.zeros((len(h.col_values(start_rowx=30, colx=1))))
    Y = np.zeros((len(h.col_values(start_rowx=31, colx=1)),2))
    X = np.zeros((len(h.col_values(start_rowx=31, colx=1)),24))
    i = 0
    for fips_id in map(int,h.col_values(start_rowx=31, colx=1)):
        fips_indices[i] = fips_id
        Y[i,0] = h.col_values(start_rowx=31, colx=6)[i]
        Y[i,1] = h.col_values(start_rowx=31, colx=4)[i]
        X[i,1] = h.col_values(start_rowx=31, colx=18)[i]
        i += 1
    #To make a nice plot national plot of the actual election results
    #national_plot(fips_indices, Y[:,0])
    
    #Population-based statistics first
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'population_1970_2016.xls', 2015)
    data_col = 2
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'black_population_2009_2015.xls', 2015)
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2015, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'black_population_2009_2015.xls', 2009)
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2009, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'white_population_2009_2015.xls', 2015)      
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2015, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'white_population_2009_2015.xls', 2009)      
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2009, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'asian_population_2009_2015.xls', 2015)      
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2015, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'asian_population_2009_2015.xls', 2009)      
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2015, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'indian_population_2009_2015.xls', 2015)      
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2015, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'indian_population_2009_2015.xls', 2009)      
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2009, pop = X[:,0]) 
    data_col += 1
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'hispanic_population_2009_2015.xls', 2015)      
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2015, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'hispanic_population_2009_2015.xls', 2009)      
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2009, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'food_stamps_1989_2013.xls', 2013)
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2013, pop = X[:,0]) 
    data_col += 1
    
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'food_stamps_1989_2013.xls', 2008)
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, data_col, 'population1000', 2008, pop = X[:,0]) 
    data_col += 1
    
    #Population density
    X, Y, fips_indices = ld.normalize_data(X, Y, fips_indices, 0, 'density', 2015)
    
    #Next, rates & percents
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices,'commuting_time_2009_2015.xls', 2015, 'fraction_of_hour')
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'unemployment_rate_1970_2017.xls','2016 October', 'percent')
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'unemployment_rate_1970_2017.xls','2007 October', 'percent')
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'bachelors_2010_2012.xls', 2012, 'percent')
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'median_age_2009_2015.xls', 2015, 'percent')
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'median_age_2009_2015.xls', 2009, 'percent')
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'rent_burdened_2010_2015.xls', 2015, 'percent')
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'homeownership_rate_2009_2015.xls', 2015, 'percent')
    X, Y, fips_indices = ld.add_data_column(X, Y, fips_indices, 'income_inequality_2010_2015.xls', 2015, 'percent')
    
    file = open('data.pk1', 'wb')
    pickle.dump([X,Y,fips_indices],file)
    file.close()

else:       
    print "Loading data..."
    file = open('data.pk1', 'rb')
    [X,Y,fips_indices] = pickle.load(file)
    file.close()     
    print "Done!"


#Initial Analysis: use PA as our control state (no electronic voting). PA FIPS codes begin with 42
#Use Florida as the test state (maybe hacked). FL FIPS codes begin with 12
#Coincidentally, PA and FL both have 67 counties. Who knew?
#Train the neural net on all other states





states_codes={'AL':'01','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','FL':'12','GA':'13','HI':'15','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}
states=['AL','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA','KS','KY','LA','ME','MD','MA','MI','MN','MS','MI','MT','NE','NV','NH','NJ','NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']
electoral_votes = [9,11,6,55,9,7,3,29,16,4,4,20,11,6,6,8,8,4,10,11,16,10,6,10,3,5,6,4,14,5,29,15,3,18,7,7,20,4,9,3,11,38,6,3,13,12,5,10,3]
electoral_counts = dict(zip(states, electoral_votes))

def run_neural_net(test_state, plot_state=False, states=states_codes):
    if test_state not in states:
        print "State not available!"
        return 0
    else:
        fips_min = int(states[test_state]+'000')
        fips_max = int(str(int(states[test_state])+1)+'000')
        #Inputs: the data matrix for the counties of interest
        #Outputs: the actual fraction GOP/(GOP+DEM)
        #FIPS: The county codes corresponding to each row in Inputs/Outputs/Pop
        #Pop: number of votes cast DEM+GOP in each county
        rows = X.shape[0]
        cols = X.shape[1]
        training_inputs = np.zeros((rows,cols))
        training_outputs = np.zeros((rows))
        training_fips = np.zeros((rows))
        training_pop = np.zeros((rows))

        test_inputs = np.zeros((rows, cols))
        test_outputs = np.zeros((rows))
        test_fips = np.zeros((rows))
        test_pop = np.zeros((rows))

        #Segment the data into one of the two categories (training or test)
        for i in range(len(fips_indices)):
            if int(fips_indices[i])>fips_min and int(fips_indices[i])<fips_max:
                test_inputs[i,:] = X[i,:]
                test_outputs[i] = Y[i,0]
                test_fips[i] = int(fips_indices[i])
                test_pop[i] = Y[i,1]
        
            else:
                training_inputs[i,:] = X[i,:]
                training_outputs[i] = Y[i,0]
                training_fips[i] = int(fips_indices[i])
                training_pop[i] = Y[i,1]
        
        #Remove the rows corresponding to counties that aren't classified in that category
        test_inputs = test_inputs[np.nonzero(test_fips)]
        test_outputs = test_outputs[np.nonzero(test_fips)]
        test_fips = test_fips[np.nonzero(test_fips)]
        test_pop = test_pop[np.nonzero(test_pop)]

        training_inputs = training_inputs[np.nonzero(training_fips)]
        training_outputs = training_outputs[np.nonzero(training_fips)]
        training_fips = training_fips[np.nonzero(training_fips)]
        training_pop = training_pop[np.nonzero(training_pop)]
        test_results = np.zeros((len(test_fips), 10))
        training_results = np.zeros((len(training_fips),10))
        
        #Run the neural net
        print "Initiating neural network..."
        trials = 10
        for i in range(trials):
            nn = mlp(verbose=True)
            "Training neural network..."
            nn.fit(training_inputs, training_outputs)
            print "Done!"
        
            tmp_results = nn.predict(test_inputs)
            test_results[:,i] = tmp_results
            
            training_results[:,i] = nn.predict(training_inputs)   
            """
            #Interpret results     
            bias = np.mean(test_results-test_outputs)*100.0
            print "Average deviation of " + str(np.mean(np.abs(test_results-test_outputs))*100.0)+"% +/-" + str(np.std(np.abs(test_outputs-test_results))*100)+"%"
            if bias>0.0:
                print "Average bias of " + str(np.abs(bias)) + "% in favor of Hillary"
            if bias<0.0:
                print "Average bias of " + str(np.abs(bias)) + "% in favor of Trump"
            """
                
        test_results = np.sum(test_results, axis=1)/float(trials)
        training_results = np.sum(training_results, axis=1)/float(trials)
        if plot_state:
            #state_plot_data_model(test_fips, test_outputs, test_results, test_state)
            state_plot_residual(test_fips, test_outputs, test_results, test_state, X, Y, training_results, fips_indices)
        return (test_outputs-test_results)*100.0, test_fips
