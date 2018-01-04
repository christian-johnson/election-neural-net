from election_analysis import *


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
