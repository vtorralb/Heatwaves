import matplotlib
from scipy import signal
import os
import datetime
import time
import copy
import shutil
import sys
import xarray as xr
from function_read import *
import numpy as np
from scipy.stats import gaussian_kde
import calendar
import locale
locale.setlocale( locale.LC_ALL , 'en_US' )
from netCDF4 import num2date, date2num
from joblib import Parallel, delayed
import joblib
from HWMI import *
import cftime

def parallelized_HWMIs_computation(ilat, ilon):
        HWMIyear, HWlstyear, HWstartmembyear, HWendmembyear, ndayexedthresholdyear, DDthresholdyear, subHWarrayyear, fitsubHWarrayyear, dataMeanarrayyear, impossible_fit  = calc_HWMIyear(target[ilon,ilat,:,:,:], cross_valid = cv, percent_thresh = percent_thresh, duration_min = duration_min)
        HWMI[:,:,ilat,ilon]=np.array(HWMIyear)
        ndayexedthreshold[:,:,ilat,ilon]=np.array(ndayexedthresholdyear)
        DDthreshold[:,:,ilat,ilon]=np.array(DDthresholdyear)
        fitsubHWarray[:,:, :, ilat,ilon]=np.array(fitsubHWarrayyear)
        subHWarray[:,:, :, ilat,ilon]=np.array(subHWarrayyear)
        dataMeanarray[:,:,ilat,ilon]=np.array(dataMeanarrayyear)
        impossible_fit_list.append((impossible_fit,ilat,ilon))
        return()


#var='tmax'
var='atemp2m_night'
#expname='ECMWF-5'
#expname='dwd-21'
expname='cmcc-35'
nmemb=40
reg_name ='Europe'
season='15MJJA'
pathPrevi = "/data/csp/vt17420/C3S/"+expname+"/daily/Europe/"+var+"_r360x180/"
start_year=1993
end_year=2016
nyear=end_year-start_year+1
lats_bnds = np.array([25,70])
lons_bnds = np.array([-15, 60])
nlat = lats_bnds[1]-lats_bnds[0]
nlon = lons_bnds[1]-lons_bnds[0]


### PERCENTILE THRESHOLD
#percent_thresh = 95
percent_thresh = 90
cv='CV'
cv_str = cv
### MINIMAL DURATION OF A HW
duration_min = 3

nday=109
ndayseas = nday//duration_min +1
target = np.ma.zeros((nyear, nday,nmemb, nlat,nlon))
for i,iyear in enumerate(range(start_year,end_year+1)):
    file=pathPrevi+var+'_'+str(iyear)+'_15MJJA.nc'
    print(file)
    varf = xr.open_dataset(file)
    data1= varf.to_array()
    print(data1.shape)
    data1= varf.to_array()[0,:,:,:,:]
    target[i,:,:,:,:]=data1
    #target[i,:,:,:,:]=np.transpose(data1,[0,3,1,2])
    

target = np.transpose(target,[4,3,0,1,2])
print(target.shape)
HWMI = np.zeros((nyear,nmemb,nlat,nlon))
HW = np.zeros((nyear,nmemb,nday,nlat,nlon))
ndayexedthreshold = np.zeros((nyear,nmemb,nlat,nlon))
DDthreshold = np.zeros((nyear,nmemb,nlat,nlon))
fitsubHWarray = np.zeros((nyear, ndayseas, nmemb, nlat, nlon))
subHWarray = np.zeros((nyear, ndayseas, nmemb, nlat, nlon))
impossible_fit_list = []
dataMeanarray = np.zeros((nyear,nmemb,nlat,nlon))
  
Parallel(n_jobs=-1, timeout = 5*3600, verbose = 20, require='sharedmem', mmap_mode='w+')(delayed(parallelized_HWMIs_computation)(ilat, ilon) for ilat in range(nlat) for ilon in range(nlon))


season_start_day=[5,15]
season_start_day=[8,31]
dirout = "/data/csp/vt17420/C3S/"+expname+"/seasonal/HWMI_"+var+"/" 
if not os.path.isdir(dirout):
    os.makedirs(dirout)

for j,jmemb in enumerate(range(0, nmemb)):
    memb_str = 'memb_'+str(jmemb)
    parameters_str = reg_name+"_"+var+'_'+season+"_"+cv_str+'_percent%i'%(percent_thresh)+'_daymin%i'%(duration_min)+"_ref%i-%i_"%(start_year, end_year)+memb_str
    varout1 = "HWMI"+"_"+var+'_'+parameters_str
    print(parameters_str)
    nrealisation=1

    for i,iyear in enumerate(range(start_year, end_year+1)):
        #print iyear
        fileout=dirout+varout1+"_%i.nc"%(iyear) #0%i01, monstart)
        print(fileout)
        if len(glob(fileout))==1:
            os.remove(fileout)
        fout=netCDF4.Dataset(fileout, "w")
        #fin=netCDF4.Dataset(targetflst[iyear])
        lat = fout.createDimension('lat', nlat)
        lon = fout.createDimension('lon', nlon) 
        rea = fout.createDimension('realisation', nrealisation) 
        timedim = fout.createDimension('time', None) 
        times = fout.createVariable('time', np.float64, ('time',)) 
        latitudes = fout.createVariable('lat', np.float32, ('lat',)) 
        longitudes = fout.createVariable('lon', np.float32,  ('lon',)) 

        # Time variable
        times.units = 'hours since 0001-01-01 00:00:00'  
        times.calendar = 'gregorian' 
        times[:]=date2num(datetime(iyear,season_start_day[0],season_start_day[1]),units = times.units, calendar = times.calendar) 
        latitudes[:] = varf.coords['latitude']
        lonaux = varf.coords['longitude']
        #lonaux[lonaux<0]=lonaux[lonaux<0]+360
        longitudes[:] = lonaux
        latitudes.units = 'degree_north'  
        longitudes.units = 'degree_east' 


        # Create the HWMI 4-d variable
        HWMIfile = fout.createVariable(varout1, np.float32, ('time','realisation','lat','lon'))
        fout.description = 'HWMI index (Russo et al. 2014) for ' + season + ' computed in cross validation'
        fout.history = 'computed from python script by C.Prodhomme & S.Lecestre' + time.ctime(time.time())
        fout.source = 'HWMI for ' + expname
        latitudes.units = 'degree_north'  
        longitudes.units = 'degree_east' 
        HWMIfile.units = 'Probability'

        # Create the nb of days 4-d variable
        expercentfile = fout.createVariable("nbdaygtpercentpct", np.float32, ('time','realisation','lat','lon'))
        expercentfile.units = 'Number of days'  


        # Create the nb of days 4-d variable
        DDthresholdfile = fout.createVariable("DDthreshold", np.float32, ('time','realisation','lat','lon'))
        DDthresholdfile.units = 'degree'

        # Create the SSTmean 4-d variable
        datameanfile = fout.createVariable(var, np.float32, ('time', 'realisation', 'lat', 'lon'))
        datameanfile.units = 'degree'

        # Write the HWMI variable
        HWMIaux=HWMI[i:i+1,j:j+1,:,:]
        HWMIfile[0:1,0:1,:,:]=HWMIaux

        # Write the number of days
        exedaux=ndayexedthreshold[i:i+1,j:j+1,:,:]
        expercentfile[0:1,0:1,:,:]=exedaux 

        # Write the DDthreshold
        exedaux=DDthreshold[i:i+1,j:j+1,:,:]
        DDthresholdfile[0:1,0:1,:,:]=exedaux 

        # Write the SSTmean
        dataaux = dataMeanarray[i:i+1,j:j+1,:,:]
        datameanfile[0:1,0:1,:,:]=dataaux
        fout.close()    
        
