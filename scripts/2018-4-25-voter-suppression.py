from election_analysis import *

def stateIndices(model, state):
    testIndices= np.where((model.data.index>int(model.stateCodes[state])*1000) & (model.data.index<(int(model.stateCodes[state])+1)*1000))[0]
    return testIndices

attribs = [
            'GOPChange',
            'GOP2016',
            'GOP2012',

            #'AL_GOP2017',

            #'Turnout2012',

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
            'Youth2012',

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

#Initialize the analysis object
model = ElectionAnalysis(regressionType='Linear', numDivisions=50, attribs=attribs, reloadData=False)
#Do the machine learning to figure out how much turnout is expected
model.makePipeline(demographicsChanges=True, economicChanges=True)
model.splitTestTrain(technique='statewise', i=['WI','MS','VA', 'GA', 'IN', 'TN'])
model.prepData(labelName = 'Turnout2016')
model.trainModel()
model.predictModel()
model.residModel()
model.rmsModel()

#Get the results & plot them
print('RMS = ' + str(model.rms))
print('Weighted mean turnout residual (among all strict voter ID states): ' + str(100.*np.sum(model.data['Population2016'][model.residuals.index].values * model.residuals.values)/np.sum(model.data['Population2016'][model.residuals.index].values))+ "%")
wiscoIndices = stateIndices(model, 'WI')
mississippiIndices = stateIndices(model, 'MS')
virginiaIndices = stateIndices(model, 'VA')
georgiaIndices = stateIndices(model, 'GA')
tennesseeIndices = stateIndices(model, 'TN')
indianaIndices = stateIndices(model, 'IN')
print('Total turnout residual in Wisconsin: ' + str(100.*np.sum((model.data.iloc[wiscoIndices]['Population2016'].values * model.residuals[model.data.iloc[wiscoIndices].index].values)/np.sum(model.data.iloc[wiscoIndices]['Population2016'].values)))+ "%")
print('Total turnout residual in Mississippi: ' + str(100.*np.sum((model.data.iloc[mississippiIndices]['Population2016'].values * model.residuals[model.data.iloc[mississippiIndices].index].values)/np.sum(model.data.iloc[mississippiIndices]['Population2016'].values)))+ "%")
print('Total turnout residual in Virginia: '+ str(100.*np.sum((model.data.iloc[virginiaIndices]['Population2016'].values * model.residuals[model.data.iloc[virginiaIndices].index].values)/np.sum(model.data.iloc[virginiaIndices]['Population2016'].values)))+ "%")
print('Total turnout residual in Georgia: ' + str(100.*np.sum((model.data.iloc[georgiaIndices]['Population2016'].values * model.residuals[model.data.iloc[georgiaIndices].index].values)/np.sum(model.data.iloc[georgiaIndices]['Population2016'].values)))+ "%")
print('Total turnout residual in Tennessee: ' + str(100.*np.sum((model.data.iloc[tennesseeIndices]['Population2016'].values * model.residuals[model.data.iloc[tennesseeIndices].index].values)/np.sum(model.data.iloc[tennesseeIndices]['Population2016'].values)))+ "%")
print('Total turnout residual in Indiana: ' + str(100.*np.sum((model.data.iloc[indianaIndices]['Population2016'].values * model.residuals[model.data.iloc[indianaIndices].index].values)/np.sum(model.data.iloc[indianaIndices]['Population2016'].values)))+ "%")

originalVote = np.sum(model.data.iloc[wiscoIndices]['VotesGOP2016'].values)/(np.sum(model.data.iloc[wiscoIndices]['VotesGOP2016'].values)+np.sum(model.data.iloc[wiscoIndices]['VotesDem2016'].values))
newRepublican = (1.0-model.residuals[model.data.iloc[wiscoIndices].index].values)*model.data.iloc[wiscoIndices]['VotesGOP2016'].values
newDemocrat = (1.0-model.residuals[model.data.iloc[wiscoIndices].index].values)*model.data.iloc[wiscoIndices]['VotesDem2016'].values
newVote = np.sum(newRepublican)/(np.sum(newRepublican)+np.sum(newDemocrat))
print('Wisconsin actual vote: ' + str(100*originalVote) + '% corrected vote: ' + str(100.*newVote)+'%')

originalVote = np.sum(model.data.iloc[tennesseeIndices]['VotesGOP2016'].values)/(np.sum(model.data.iloc[tennesseeIndices]['VotesGOP2016'].values)+np.sum(model.data.iloc[tennesseeIndices]['VotesDem2016'].values))
newRepublican = (1.0-model.residuals[model.data.iloc[tennesseeIndices].index].values)*model.data.iloc[tennesseeIndices]['VotesGOP2016'].values
newDemocrat = (1.0-model.residuals[model.data.iloc[tennesseeIndices].index].values)*model.data.iloc[tennesseeIndices]['VotesDem2016'].values
newVote = np.sum(newRepublican)/(np.sum(newRepublican)+np.sum(newDemocrat))
print('Tennessee actual vote: ' + str(100*originalVote) + '% corrected vote: ' + str(100.*newVote)+'%')

originalVote = np.sum(model.data.iloc[georgiaIndices]['VotesGOP2016'].values)/(np.sum(model.data.iloc[georgiaIndices]['VotesGOP2016'].values)+np.sum(model.data.iloc[georgiaIndices]['VotesDem2016'].values))
newRepublican = (1.0-model.residuals[model.data.iloc[georgiaIndices].index].values)*model.data.iloc[georgiaIndices]['VotesGOP2016'].values
newDemocrat = (1.0-model.residuals[model.data.iloc[georgiaIndices].index].values)*model.data.iloc[georgiaIndices]['VotesDem2016'].values
newVote = np.sum(newRepublican)/(np.sum(newRepublican)+np.sum(newDemocrat))
print('Georgia actual vote: ' + str(100*originalVote) + '% corrected vote: ' + str(100.*newVote)+'%')

originalVote = np.sum(model.data.iloc[virginiaIndices]['VotesGOP2016'].values)/(np.sum(model.data.iloc[virginiaIndices]['VotesGOP2016'].values)+np.sum(model.data.iloc[virginiaIndices]['VotesDem2016'].values))
newRepublican = (1.0-model.residuals[model.data.iloc[virginiaIndices].index].values)*model.data.iloc[virginiaIndices]['VotesGOP2016'].values
newDemocrat = (1.0-model.residuals[model.data.iloc[virginiaIndices].index].values)*model.data.iloc[virginiaIndices]['VotesDem2016'].values
newVote = np.sum(newRepublican)/(np.sum(newRepublican)+np.sum(newDemocrat))
print('Virginia actual vote: ' + str(100*originalVote) + '% corrected vote: ' + str(100.*newVote)+'%')

#model.subCountyPlot(pd.Series(index=model.residuals.index, data=model.residuals.values),dataMin=-0.2, dataMax=0.2, vMin=-20.0, vMax=20.0, cLabel = 'Turnout Residual', pltTitle='2018-4-25-voter-suppression/wiscoResiduals', cMap = cm.BrBG)
model.bubblePlot(xValues=100.*(model.data['Black2015'][model.residuals.index].values), yValues = 100.*model.residuals.values, sValues=model.data['Population2016'][model.residuals.index].values/1500.0, cValues=model.data['GOP2016'][model.residuals.index].values, fileName = '2018-4-25-voter-suppression/turnoutresiduals_race', xLabel='Black Population', yLabel='Turnout Residual', scale ='log', minXValue=1.0, minYValue=-20.0, maxXValue=100.0, maxYValue=20.0, cMap = cm.seismic)
#model.bubblePlot(xValues=(model.data['MedianIncome'][model.residuals.index].values)/max((model.data['MedianIncome'][model.residuals.index].values)), yValues = 100.*model.residuals.values, sValues=model.data['Population2016'][model.residuals.index].values/10000.0, cValues=model.data['GOP2016'][model.residuals.index].values, fileName = '2018-4-25-voter-suppression/turnoutresiduals_medianincome', xLabel='Median Income', yLabel='Turnout Residual', minXValue=0.0, minYValue=-20.0, maxXValue=1.0, maxYValue=20.0, cMap = cm.seismic)
model.bubblePlot(xValues=(model.data['Hispanic2015'][model.residuals.index].values)/max((model.data['Hispanic2015'][model.residuals.index].values)), yValues = model.residuals.values, sValues=model.data['Population2016'][model.residuals.index].values/10000.0, cValues=model.data['GOP2016'][model.residuals.index].values, fileName = '2018-4-25-voter-suppression/turnoutresiduals_hispanic', xLabel='Hispanic Population', yLabel='Turnout Residual', minXValue=0.0, minYValue=-20.0, maxXValue=1.0, maxYValue=20.0, cMap = cm.seismic)

#Make other plots for the blog post
#model.countyPlot(pd.Series(index=model.data.index, data=model.data['Turnout2016'].values-model.data['Turnout2012'].values),dataMin=-0.1, dataMax=0.1, vMin=-10.0, vMax=10.0, cLabel = 'Change in Voter Turnout 2012-2016', cMap = cm.BrBG, pltTitle='2018-4-25-voter-suppression/turnoutChange')
model.bubblePlot(xValues=(100.*model.data['Black2015'].values), yValues = 100.*model.data['Turnout2016'].values-100.*model.data['Turnout2012'].values, sValues=model.data['Population2016'].values/10000.0, cValues=model.data['GOP2016'].values, fileName = '2018-4-25-voter-suppression/turnoutchange_race', xLabel='Black Population', yLabel='Turnout Change', minXValue=0.01, minYValue=-20.0, maxXValue=100.0, maxYValue=20.0, cMap = cm.seismic, scale = 'log')
#model.bubblePlot(xValues=(model.data['GOP2016'].values), yValues = model.data['Turnout2016'].values-model.data['Turnout2012'].values, sValues=model.data['Population2016'].values/10000.0, cValues=(model.data['Black2015'].values)**(1./2.), fileName = '2018-4-25-voter-suppression/turnoutchange_vote', xLabel='GOP Vote Fraction', yLabel='Turnout Change', minXValue=0.0, minYValue=-0.2, maxXValue=1.0, maxYValue=0.2, cMap = cm.YlGn)
#model.bubblePlot(xValues=(model.data['PopulationDensity2016'].values/np.max(model.data['PopulationDensity2016'].values))**(1./8.), yValues = model.data['Turnout2016'].values-model.data['Turnout2012'].values, sValues=model.data['Population2016'].values/10000.0, cValues=model.data['GOP2016'].values, fileName = '2018-4-25-voter-suppression/turnoutchange_popdensity', xLabel='Population Density', yLabel='Turnout Change', minXValue=0.0, minYValue=-0.2, maxXValue=1.0, maxYValue=0.2, cMap = cm.seismic)
