from election_analysis import *

attribs = [
        'GOP2012',
        'AL_GOP2017',
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
model = ElectionAnalysis(regressionType='Linear', numDivisions=10, attributes=attribs)
model.loadData()
model.cleanData()
model.makePipeline(demographicsChanges=True, economicChanges=True)

for i in range(67):
    model.splitTestTrain(technique='individual', i=i)
    model.prepData(labelName = 'AL_GOP2017')
    model.trainModel()
    model.predictModel()
    model.residModel()
    model.concatenateResiduals()
#model.statePlot(pd.Series(index=model.data.index, data=model.data['AL_GOP2017'].values),dataMin=0.0, dataMax=1.0, vMin=0.0, vMax=100.0, cLabel = 'GOP Vote Share- 2017 Special Election', pltTitle='2018-1-8-alabama/alabama2017')
#model.statePlot(pd.Series(index=model.data.index, data=model.data['GOP2016'].values),dataMin=0.0, dataMax=1.0, vMin=0.0, vMax=100.0, cLabel = 'GOP Vote Share- 2016 Presidential Election', pltTitle='2018-1-8-alabama/alabama2016')
#model.statePlot(pd.Series(index=model.data.index, data=model.data['AL_GOP2017'].values-model.data['GOP2016'].values),dataMin=-0.2, dataMax=0.2, vMin=-20.0, vMax=20.0, cLabel = 'GOP Vote Share Change 2016-2017', pltTitle='2018-1-8-alabama/alabamachange')
#model.statePlot(pd.Series(index=model.data.index, data=model.results),dataMin=-0.2, dataMax=0.2, vMin=-20.0, vMax=20.0, cLabel = 'GOP Vote Share Residual', pltTitle='2018-1-8-alabama/alabamaresidual')
model.bubblePlot(xValues=100.0*model.data['AL_GOP2017'].values, yValues=100.0*(model.data['AL_GOP2017']-model.data['GOP2016']), sValues=model.data['Population2016'].values/1000.0, cValues=model.data['GOP2016'].values, fileName='2018-1-8-alabama/vote_vs_votechange', xLabel='2017 Alabama Special Election GOP Vote Share', yLabel='Change in GOP Vote Share 2016-2017', minXValue=0.0, minYValue=-30.0, maxXValue=100.0, maxYValue=20.0, cMap = cm.seismic)

#model.bubblePlot(xValues=100.0*model.data['GOP2016'].values, yValues=100.0*model.data['AL_GOP2017'], sValues=model.data['Population2016'].values/1000.0, cValues=model.data['GOP2016'].values, fileName='2018-1-8-alabama/vote2016_vs_vote2017', xLabel='2016 Presidential Election GOP Vote Share', yLabel='2017 Alabama Special Election GOP Vote Share', minXValue=0.0, minYValue=0.0, maxXValue=100.0, maxYValue=100.0, cMap = cm.seismic)
