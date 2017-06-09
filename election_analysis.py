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


#Define plotting function, so we can see the results in a nice way
def national_plot(fips_indices, data, column):
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
                ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=0.2)
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
    
    fig.suptitle('2016 US Presidential Election Results')
    plt.savefig('plots/election_results.pdf',bbox_inches='tight')
    plt.show()
    return 0

def state_plot(fips_indices, data, column, name, min_val, max_val):
    fig = plt.figure(figsize=(7.0, 7.0))
    print name
    print fips_indices
    ax = plt.axes(projection=ccrs.Miller(), aspect=1.3)
    #Change to be around state of interest
    if name=='PA':
        ax.set_extent([-81.0, -74., 37., 40.])
    if name == 'FL':
        ax.set_extent([-88.0,-79., 23.,30.])
    if name == 'NC':
        ax.set_extent([-85.0,-74., 32.,36.])
        
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
reload_data = False
if reload_data:
    #Column 0 of Y contains the GOP vote fraction, i.e. GOP/(GOP+DEM)
    #Column 1 of Y contains the total GOP+DEM # of votes
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

#national_plot(fips_indices, Y, 0)

#Initial Analysis: use PA as our control state (no electronic voting). PA FIPS codes begin with 42
#Use Florida as the test state (maybe hacked). FL FIPS codes begin with 12
#Coincidentally, PA and FL both have 67 counties. Who knew?
#Train the neural net on all other states


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

control_inputs = np.zeros((rows, cols))
control_outputs = np.zeros((rows))
control_fips = np.zeros((rows))
control_pop = np.zeros((rows))

test_inputs = np.zeros((rows, cols))
test_outputs = np.zeros((rows))
test_fips = np.zeros((rows))
test_pop = np.zeros((rows))

states = ['AL', 'AK', 'AZ', 'AR', 'CA','CO']
control_state = 'PA'
test_state = 'FL'
#Segment the data into one of the three categories (training, control, test)
for i in range(len(fips_indices)):
    if int(fips_indices[i])>42000 and int(fips_indices[i])<43000:
        control_inputs[i,:] = X[i,:]
        control_outputs[i] = Y[i,0]
        control_fips[i] = int(fips_indices[i])
        control_pop[i] = Y[i,1]
        
    elif int(fips_indices[i])>37000 and int(fips_indices[i])<38000:
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
control_inputs = control_inputs[np.nonzero(control_fips)]
control_outputs = control_outputs[np.nonzero(control_fips)]
control_fips = control_fips[np.nonzero(control_fips)]
control_pop = control_pop[np.nonzero(control_pop)]

test_inputs = test_inputs[np.nonzero(test_fips)]
test_outputs = test_outputs[np.nonzero(test_fips)]
test_fips = test_fips[np.nonzero(test_fips)]
test_pop = test_pop[np.nonzero(test_pop)]

training_inputs = training_inputs[np.nonzero(training_fips)]
training_outputs = training_outputs[np.nonzero(training_fips)]
training_fips = training_fips[np.nonzero(training_fips)]
training_pop = training_pop[np.nonzero(training_pop)]



print "Initiating neural network..."
nn = mlp(verbose=True)
"Training neural network..."
nn.fit(training_inputs, training_outputs)
print "Done!"
control_results = nn.predict(control_inputs)
min_val = -0.25#np.min(control_outputs-control_results)#*control_pop)
max_val = 0.25#np.max(control_outputs-control_results)#*control_pop)
#print len(control_outputs)
print control_outputs
print control_results
print "Average deviation of " + str(np.mean(np.abs(control_results-control_outputs))*100.0)+"% +/-" + str(np.std(np.abs(control_outputs-control_results))*100)+"%"
bias = np.mean(control_results-control_outputs)*100.0
if bias>0.0:
    print "Average bias of " + str(np.abs(bias)) + "% in favor of Hillary"
if bias<0.0:
    print "Average bias of " + str(np.abs(bias)) + "% in favor of Trump"

#Data-model
state_plot(control_fips, control_outputs-control_results, 0, 'PA', min_val, max_val)



test_results = nn.predict(test_inputs)
print test_outputs
print test_results
print (test_outputs-test_results)*100.0
bias = np.mean(test_results-test_outputs)*100.0
print "Average deviation of " + str(np.mean(np.abs(test_results-test_outputs))*100.0)+"% +/-" + str(np.std(np.abs(test_outputs-test_results))*100)+"%"

if bias>0.0:
    print "Average bias of " + str(np.abs(bias)) + "% in favor of Hillary"
if bias<0.0:
    print "Average bias of " + str(np.abs(bias)) + "% in favor of Trump"

state_plot(test_fips, (test_outputs-test_results), 0, 'NC', min_val, max_val)



