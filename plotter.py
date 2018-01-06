from matplotlib import cm
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from matplotlib import rcParams
from matplotlib.pyplot import rc
import matplotlib.ticker as mtick
import numpy as np

#Set matplotlib parameters to be able to use Greek letters in plots
#rcParams['text.usetex'] = True
#rc('text.latex', preamble=r'\usepackage{amsmath}')
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 14
rcParams['axes.labelcolor'] = '#787878'

rcParams['xtick.labelsize'] =12
rcParams['ytick.labelsize'] =12
rcParams['xtick.major.size'] = 6
rcParams['ytick.major.size'] = 6
rcParams['xtick.color'] = '#787878'
rcParams['ytick.color'] = '#787878'

rcParams['text.color'] = '#787878'
rcParams['font.size'] = 14




def state_plot_data_model(dataseries1, dataseries2, name, data_min, data_max, vmin, vmax, plt_title, cb2_label, use_cmap = cm.seismic):
    fig = plt.figure(figsize=(14.0, 7.0))
    ax1 = plt.axes([0.0, 0.0, 0.45, 1.0],projection=ccrs.Miller(), aspect=1.3)
    ax2 = plt.axes([0.45, 0.0, 0.45, 1.0],projection=ccrs.Miller(), aspect=1.3)
    ax1.set_extent(state_extent(name))
    ax2.set_extent(state_extent(name))

    filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
    states_codes={'AL':'01','AZ':'04','AR':'05','CA':'06','CO':'08','CT':'09','DE':'10','FL':'12','GA':'13','HI':'15','ID':'16','IL':'17','IN':'18','IA':'19','KS':'20','KY':'21','LA':'22','ME':'23','MD':'24','MA':'25','MI':'26','MN':'27','MS':'28','MO':'29','MT':'30','NE':'31','NV':'32','NH':'33','NJ':'34','NM':'35','NY':'36','NC':'37','ND':'38','OH':'39','OK':'40','OR':'41','PA':'42','RI':'44','SC':'45','SD':'46','TN':'47','TX':'48','UT':'49','VT':'50','VA':'51','WA':'53','WV':'54','WI':'55','WY':'56'}

    the_lw = 0.0
    for state, record in zip(shpreader.Reader(filename).geometries(), shpreader.Reader(filename).records()):
        id = int(record.__dict__['attributes']['GEOID'])
        if id>int(states_codes[name])*1000 and id<(1+int(states_codes[name]))*1000:
            if id in dataseries1.index:
                value = (dataseries1.loc[id]-data_min)/(data_max-data_min)
                facecolor = use_cmap(value)
                edgecolor = 'black'
                #Is the county in Hawaii, Alaska, or the mainland?
                ax1.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)
            else:
                facecolor = 'gray'
                edgecolor = 'black'
                #Is the county in Hawaii, Alaska, or the mainland?
                ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)

            if id in dataseries2.index:
                value = (dataseries2.loc[id]-data_min)/(data_max-data_min)
                facecolor = use_cmap(value)
                edgecolor = 'black'
                #Is the county in Hawaii, Alaska, or the mainland?
                ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)
            else:
                facecolor = 'gray'
                edgecolor = 'black'
                #Is the county in Hawaii, Alaska, or the mainland?
                ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)

    filename = 'cb_2015_us_state_5m/cb_2015_us_state_5m.shp'
    the_lw = 0.5
    for state, record in zip(shpreader.Reader(filename).geometries(), shpreader.Reader(filename).records()):
        id = int(record.__dict__['attributes']['GEOID'])
        if int(id)==int(states_codes[name]):
            facecolor = use_cmap(0.0)
            edgecolor = 'black'
            #Is the county in Hawaii, Alaska, or the mainland?
            ax1.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor=edgecolor, linewidth=the_lw)
            ax2.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor=edgecolor, linewidth=the_lw)

    ax1.background_patch.set_visible(False)
    ax1.outline_patch.set_visible(False)
    ax2.background_patch.set_visible(False)
    ax2.outline_patch.set_visible(False)

    ax1.set_title('Data')
    ax2.set_title('Model')
    cbar_step=(vmax-vmin)/5

    axc2 = plt.axes([0.92, 0.25, 0.02, 0.5], frameon=False)
    norm2 = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb2 = mpl.colorbar.ColorbarBase(axc2, cmap=use_cmap,norm=norm2,orientation='vertical')
    cb2.ax.set_yticklabels(['{:.0f}\%'.format(x) for x in np.arange(vmin, vmax+cbar_step, cbar_step)])
    cb2.set_label(cb2_label)

    plt.savefig('plots/'+str(plt_title)+'.pdf',bbox_inches='tight')
    #plt.show()
    return 0


#Define plotting function, so we can see the results in a nice way
def national_plot(dataseries, data_min, data_max, vmin, vmax, plt_title,colorbar_label, use_cmap = cm.seismic, AK_value = False):
    fig = plt.figure(figsize=(10.0,7.0))
    #Mainland
    ax = plt.axes([0.0,0.0,1.0,1.0],projection=ccrs.LambertConformal(central_longitude=-96.0, central_latitude=39.0, cutoff=-20), aspect=1.15, frameon=False)
    ax.set_extent([-120.0,-74., 21.,47.])
    #Alaska
    ax2 = plt.axes([0.0,0.0,0.26,0.26],projection=ccrs.LambertConformal(central_longitude=-156.0, central_latitude=53.5, cutoff=-20), aspect=1.3, frameon=False)
    ax2.set_extent([-180.0,-132., 45.,62.])
    #Hawaii
    ax3 = plt.axes([0.25, 0.00,0.2,0.2],projection=ccrs.LambertConformal(central_longitude=-157.0, central_latitude=20.5, cutoff=-20), aspect=1.15, frameon=False)
    ax3.set_extent([-162.0,-154., 18.,23.])
    filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
    the_lw = 0.0

    for state, record in zip(shpreader.Reader(filename).geometries(), shpreader.Reader(filename).records()):
        id = int(record.__dict__['attributes']['GEOID'])
        if id == '46102':
            id = '46113'
        if id in dataseries.index:
            value = (dataseries.loc[id]-data_min)/(data_max-data_min)
            facecolor = use_cmap(value)
            edgecolor = 'black'
            #Is the county in Hawaii, Alaska, or the mainland?
            if int(record.__dict__['attributes']['GEOID'])<2991 and int(record.__dict__['attributes']['GEOID'])>2013:
                ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)
            elif int(record.__dict__['attributes']['GEOID'])<15010 and int(record.__dict__['attributes']['GEOID'])>15000:
                ax3.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)
            else:
                ax.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)
        else:
            facecolor = 'gray'
            edgecolor = 'black'
            #Is the county in Hawaii, Alaska, or the mainland?
            if int(record.__dict__['attributes']['GEOID'])<2991 and int(record.__dict__['attributes']['GEOID'])>2013:
                if AK_value:
                    ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=cm.seismic(AK_value), edgecolor=edgecolor, linewidth=the_lw)
                else:
                    ax2.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)

            elif int(record.__dict__['attributes']['GEOID'])<15010 and int(record.__dict__['attributes']['GEOID'])>15000:
                ax3.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)
            else:
                ax.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)

    filename = 'cb_2015_us_state_5m/cb_2015_us_state_5m.shp'
    the_lw = 0.25
    for state, record in zip(shpreader.Reader(filename).geometries(), shpreader.Reader(filename).records()):
        id = int(record.__dict__['attributes']['GEOID'])
        facecolor = use_cmap(0.0)
        edgecolor = 'black'
        #Is the county in Hawaii, Alaska, or the mainland?
        if int(record.__dict__['attributes']['GEOID'])==2:
            ax2.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor=edgecolor, linewidth=the_lw)
        elif int(record.__dict__['attributes']['GEOID'])==15:
            ax3.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor=edgecolor, linewidth=the_lw)
        else:
            ax.add_geometries(state, crs=ccrs.Miller(), facecolor='none', alpha=1.0, edgecolor=edgecolor, linewidth=the_lw)

    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)
    ax2.background_patch.set_visible(False)
    ax2.outline_patch.set_visible(False)
    ax3.background_patch.set_visible(False)
    ax3.outline_patch.set_visible(False)
    #Add colorbar
    axc = plt.axes([0.25, 0.95, 0.5, 0.012], frameon=False)
    cmap = use_cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    num_ticks = 9
    cbar_step=float((vmax-vmin)/(num_ticks-1.0))
    cb = mpl.colorbar.ColorbarBase(axc, ticks=np.linspace(vmin, vmax, num_ticks),cmap=cmap,norm=norm,orientation='horizontal')
    cb.set_ticklabels(['{:.0f}%'.format(x) for x in np.arange(vmin, vmax+cbar_step, cbar_step)])
    cb.ax.xaxis.set_ticks_position('top')

    cb.set_label(colorbar_label, fontdict = {
        'horizontalalignment' : 'center'
        })
    #plt.savefig('plots/'+str(plt_title)+'.pdf',bbox_inches='tight')
    plt.show()
    return 0

#Define plotting function, so we can see the results in a nice way
def state_plot(dataseries,name, data_min, data_max, vmin, vmax, plt_title,colorbar_label, use_cmap = cm.seismic, AK_value = False):
    fig = plt.figure(figsize=(10.0,10.0))
    ax = plt.axes(projection=ccrs.Miller(), aspect=1.3)
    ax.set_extent(state_extent(name))

    filename = 'cb_2015_us_county_5m/cb_2015_us_county_5m.shp'
    the_lw = 0.1
    for state, record in zip(shpreader.Reader(filename).geometries(), shpreader.Reader(filename).records()):
        id = int(record.__dict__['attributes']['GEOID'])
        if id == '46102':
            id = '46113'
        if id in dataseries.index:
            value = (dataseries.loc[id]-data_min)/(data_max-data_min)
            facecolor = use_cmap(value)
            edgecolor = 'black'
            ax.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)
        else:
            facecolor = 'gray'
            edgecolor = 'black'
            ax.add_geometries(state, crs=ccrs.Miller(), facecolor=facecolor, edgecolor=edgecolor, linewidth=the_lw)
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)

    #Add colorbar
    axc = plt.axes([0.93, 0.1, 0.02, 0.5], frameon=False)
    cmap = use_cmap
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    cb = mpl.colorbar.ColorbarBase(axc, cmap=cmap,norm=norm,orientation='vertical')
    cb.set_label(colorbar_label)

    plt.savefig('plots/'+str(plt_title)+'.pdf',bbox_inches='tight')
    #plt.show()
    return 0
