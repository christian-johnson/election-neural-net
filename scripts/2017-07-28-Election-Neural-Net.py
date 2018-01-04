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
        'MedianIncome'
    ]


#Initialize the analysis object
model = ElectionAnalysis(regressionType='MLP', numDivisions=50, attributes=attribs)
model.loadData()
model.cleanData()
#Figure 1- national vote results
#model.countyPlot(pd.Series(index=model.data.index, data=model.data['GOP2016'].values),dataMin=0.0, dataMax=1.0, vMin=0.0, vMax=100.0, cLabel = 'GOP Vote Share', pltTitle='2017-07-28-Election-Neural-Net/National_vote_fraction2', AK_value=0.5288700180057424)
model.makePipeline(demographicsChanges=False, economicChanges=False)

#Figure 2- Florida vote & prediction
#model.splitTestTrain(technique='statewise', i=)
#model.prepData(labelName = 'GOP2016')
#model.trainModel()
#model.predictModel()

#model.statePlot()
#model.statePlot()

#Figure 3- National vote residuals
for i in list(model.stateCodes.keys()):
    model.splitTestTrain(technique='statewise', i=i)
    model.prepData(labelName = 'GOP2016')
    model.trainModel()
    model.predictModel()
    model.residModel()
    model.concatenateResiduals()

model.countyPlot(pd.Series(index=model.data.index, data=model.results.values),dataMin=-0.2, dataMax=0.2, vMin=-20.0, vMax=20.0, cLabel = 'GOP Vote Share Residual', pltTitle='2017-07-28-Election-Neural-Net/vote_residmap2')
#Figure 4- Bubbble plot of residuals versus vote share
model.bubblePlot(100.0*model.data['GOP2016'].values, 100.0*model.results.values, sValues=model.data['Population2016'].values/4250., cValues=10.*model.results.values+0.5, fileName='2017-07-28-Election-Neural-Net/residual_vs_vote3', xLabel='GOP Vote Fraction', yLabel='GOP Vote Residual', minXValue=0.0, minYValue=-20.0, maxXValue=100.0, maxYValue=20.0, cMap = cm.coolwarm)

#Figure 5- Injected votes residuals

#Figure 6- Vote change 2012-2016

#Figure 7- Turnout

#Figure 8- Turnout residuals
