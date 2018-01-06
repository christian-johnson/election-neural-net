from election_analysis import *


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


#Initialize the analysis object
linearModel = ElectionAnalysis(regressionType='Linear', numDivisions=50, attributes=attribs)
linearModel.loadData()
linearModel.cleanData()

#Figure 1: Land area devoted to crops
#linearModel.countyPlot(pd.Series(index=linearModel.data.index, data=linearModel.data['Cotton'].values+linearModel.data['WinterWheat'].values+linearModel.data['Soybeans'].values+linearModel.data['Corn'].values),dataMin=0.0, dataMax=1.0, vMin=0.0, vMax=100.0, cLabel = 'Land Devoted to Growing Crops', pltTitle='2017-07-28-Election-Neural-Net/crops2', cMap=cm.Greens)
#Figure 2: Climate change belief
#linearModel.countyPlot(pd.Series(index=linearModel.data.index, data=linearModel.data['ClimateChange'].values),dataMin=0.3, dataMax=0.65, vMin=30.0, vMax=65.0, cLabel = 'Belief in Human-Caused Climate Change', pltTitle='2017-07-28-Election-Neural-Net/climateChange2', cMap=cm.plasma)
#Figure 3: Change in unemployment 2007-2016
linearModel.countyPlot(pd.Series(index=linearModel.data.index, data=linearModel.data['Unemployment2016'].values-linearModel.data['Unemployment2007'].values),dataMin=-0.08, dataMax=0.08, vMin=-8.0, vMax=8.0, cLabel = 'Change in Unemployment Rate 2007-2016', pltTitle='2017-07-28-Election-Neural-Net/unemployment_change2', cMap=cm.RdYlGn_r)


linearModel.makePipeline(demographicsChanges=True, economicChanges=True)

#Figure 3- National vote residuals
for i in range(50):
    linearModel.splitTestTrain(technique='hash', i=i)
    linearModel.prepData(labelName = 'GOP2016')
    linearModel.trainModel()
    linearModel.predictModel()
    linearModel.residModel()
    linearModel.concatenateResiduals()

#Initialize the analysis object
mlpModel = ElectionAnalysis(regressionType='MLP', numDivisions=50, attributes=attribs)
mlpModel.loadData()
mlpModel.cleanData()
mlpModel.makePipeline(demographicsChanges=True, economicChanges=True)

#Figure 3- National vote residuals
for i in range(50):
    mlpModel.splitTestTrain(technique='hash', i=i)
    mlpModel.prepData(labelName = 'GOP2016')
    mlpModel.trainModel()
    mlpModel.predictModel()
    mlpModel.residModel()
    mlpModel.concatenateResiduals()

#Initialize the analysis object
randomforestModel = ElectionAnalysis(regressionType='RandomForest', numDivisions=50, attributes=attribs)
randomforestModel.loadData()
randomforestModel.cleanData()
randomforestModel.makePipeline(demographicsChanges=True, economicChanges=True)

#Figure 3- National vote residuals
for i in range(50):
    randomforestModel.splitTestTrain(technique='hash', i=i)
    randomforestModel.prepData(labelName = 'GOP2016')
    randomforestModel.trainModel()
    randomforestModel.predictModel()
    randomforestModel.residModel()
    randomforestModel.concatenateResiduals()

#Figure 4: Ensemble residuals
linearModel.countyPlot(pd.Series(index=linearModel.data.index, data=pd.concat([linearModel.results,linearModel.results,linearModel.results], axis=1).mean(axis=1)),dataMin=-0.2, dataMax=0.2, vMin=-20.0, vMax=20.0, cLabel = 'GOP Vote Share Residual', pltTitle='2017-07-28-Election-Neural-Net/vote_residmap2')


print("RMS MLP = " + str(100.0*mlpModel.rmsModel()) + "%")
print("RMS Random Forest = " + str(100.0*randomforestModel.rmsModel()) + "%")
print("RMS Elastic = " + str(100.0*linearModel.rmsModel()) + "%")
print("RMS Ensemble = " + str(100.0*np.sqrt(np.mean((pd.concat([linearModel.results,mlpModel.results,randomforestModel.results], axis=1).mean(axis=1).values)**2))))
