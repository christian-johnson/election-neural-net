from election_analysis import *
from matplotlib import cm


attribs = [
            'GOP2012',

            'Turnout2012',

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
linearModel = ElectionAnalysis(attribs=attribs, regressionType='Linear', numDivisions=50, reloadData = True)
linearModel.makePipeline(demographicsChanges=True, economicChanges=True)

#Figure 3- National vote residuals
for i in range(50):
    print("i = " + str(i))
    linearModel.splitTestTrain(technique='hash', i=i)
    linearModel.prepData(labelName = 'GOP2016')
    linearModel.trainModel()
    linearModel.predictModel()
    linearModel.residModel()
    linearModel.concatenateResiduals()


print("RMS Elastic = " + str(100.0*linearModel.rmsModel()) + "%")
linearModel.countyPlot(pd.Series(index=linearModel.data.index, data=linearModel.results),dataMin=-0.2, dataMax=0.2, vMin=-20.0, vMax=20.0,cMap = cm.seismic, cLabel = 'GOP Vote Share Residual', pltTitle='test')

#Figure 4: Ensemble residuals
linearModel.bubblePlot(yValues = linearModel.results, xValues = linearModel.data['GOP2016'], sValues = linearModel.data['Population2016']/3400.0, cValues = linearModel.data['GOP2016'], fileName = 'test2', xLabel = 'GOP Vote Share', yLabel = 'GOP Vote Residual', minXValue = 0.0, minYValue = -0.2, maxXValue = 1.0, maxYValue = 0.2, cMap = cm.seismic)


#Drug poisoning GIF code
input('wait for key')

drugpoisoning = pd.read_csv('data_spreadsheets/drug_poisoning.csv').pivot(index='FIPS',columns='Year',values='Estimated Age-adjusted Death Rate, 16 Categories (in ranges)')
for column in drugpoisoning.columns:
    drugpoisoning[column] = drugpoisoning[column].str.split('-')
drugpoisoning = drugpoisoning.dropna()
for fips in drugpoisoning.index:
    for column in drugpoisoning.columns:
        drugpoisoning[column][fips] = float(drugpoisoning[column][fips][0].replace('>',''))

#Initialize the analysis object
for i in range(1999, 2016):
    print(i)
    linearModel.countyPlot(pd.Series(index=drugpoisoning.index, data=drugpoisoning[i]),dataMin=0.0, dataMax=30.0, vMin=0.0, vMax=30.0, cMap = cm.inferno, cLabel = 'Drug Poisoning Deaths per 100,000 per year', pltTitle='drugPoisoning'+str(i), pltSupTitle = str(i))
