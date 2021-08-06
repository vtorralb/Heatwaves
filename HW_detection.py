import numpy as np
import os
import datetime
import time
import copy
import shutil
import sys
from function_read import *
import joblib
from joblib import Parallel, delayed

def detectHW1year(field, lat, lon, args, allowdist=1):
    """
    field : np.array fitsubHW 
    lat, lon : np.arrays corresponding to lat, lon range.
    allowdist : neighbourhood geometrical radius. The temporal radius is fixed to one. 
    """
    expname, reg_name, memb_str, season, parameters_str, start_year, lats_reg, lons_reg = args
    #print("lon",lon)
    #print(lat)
    nlat= len(lat)
    nlon=len(lon)
    #print(field.shape)
    #print(field)
    HWwhere = np.ma.where(field>0) #select indices ix of field where field[ix]>0 --> There is a HW at ix
    #Maybe field>0.05 would be better?
    
    #print(HWwhere)
    nHWpoints = HWwhere[0].shape[0] #number of points with a HW
    #print(nHWpoints)
    
    if nHWpoints != 0:
        #transform HWwhere in a list of points
        HWpoint = []
        for iHW in range(nHWpoints):
            HWpoint.append((HWwhere[0][iHW],HWwhere[1][iHW], HWwhere[2][iHW]))
            # 0 --> time variable : day
            # 1,2 --> space variable : lat, lon
            
        #    
        #_______sort heatwave points by neigbours________
        #
        
        HWpointaux=list(HWpoint) #make a copy
        HW = []
        iHW = 0
        iyear=0
        #initialize the list of seeds with the first point
        seedlist = [HWpointaux[0]]
        #remove seed from the list
        HWpointaux=HWpointaux[1:]
        #print seedlist
        #run over all the points
        while len(HWpointaux)>0: #still some points we did not reach
            
            #create a list to store the points of one HW
            ptHWlst = [] 
            while len(seedlist)>0:
                #print(seedlist)
                #remove the seed from the list of seeds and keep the current seed point 
                seedpoint=seedlist[0]
                #print seedlist
                #add the seed to the heatwave
                #print ptHWlst
                #check neighbours for spatial and temporal dimensions
                listnei = spacelist_neighbors_highdist2(seedpoint, allowdist)
                            
                # adding temporal neighbours            
                neibef = (seedpoint[0]-1, seedpoint[1], seedpoint[2])
                
                #if not(neibef in listnei): #&(dist>0):
                #    listnei.append(neibef)
                    
                neiaft = (seedpoint[0]+1, seedpoint[1], seedpoint[2])
                #if not(neiaft in listnei): #&(dist>0):
                #    listnei.append(neiaft)
                    
                listnei = listnei+[neibef, neiaft]
    
                if reg_name != "global":
                    #remove element outside the limits (avoid useless parcours of HWpointaux)
                    listnei = [nei for nei in listnei if all(0 <= x < y for x, y in zip(nei, field.shape))]
                    #need to have lats_range between 0 and 179 et lons_range between 0 and 359
                
                for nei in listnei:
                    if not(nei in ptHWlst): #Not interested if neighbour has already been looked for
                        if nei in HWpointaux: #if neighbour point is indeed part of the HW
                            #add the neighbourg to the seedlist
                            seedlist.append(nei)
                            #remove the neigbourg from the heatwave list
                            HWpointaux.remove(nei)
                            #print(seedlist)
                #add point to HW list
                ptHWlst.append(seedpoint)
                #
                seedlist=seedlist[1:]
                
            #once the seed list is empty take a new seed it remains points
            #print HWpointaux
            if len(HWpointaux)>0:
                seedlist = [HWpointaux[0]]
                HWpointaux=HWpointaux[1:]
            #keep the list of point for each HW
            HW.append(ptHWlst)
    else:
        HW = []
    #    
    #_______END of sort heatwave points by neigbours________
    #
    
    return HW

def computeHWcara(field, mask, lat, lon, HW):
    """
    field : np.array of shape (ndayseas, nlat, nlon)
    """
    #
    #create containers to stored Heatwave informations
    HWampli = []
    HWstart = []
    HWend = []
    fieldHWlst = []
    #loop over all the HW
    #for pointlst in HW:
    for iHW in range(len(HW)):
        #print("Dealing with new heatwave")
        pointlst = HW[iHW]
        #loop over all the point part of the HW
        # unmask HW point only over land
        start = 10000
        end = -1
        #create a fully masked field
        fieldHW = np.zeros(field.shape)
        for point in pointlst:
            #exclude oceanic points for HW calculations
            #print(iHW)
            #print(point)
            #print(mask.shape)
            #if not(mask[point[1], point[2]]):
            fieldHW[point[0], point[1], point[2]] = field[point[0], point[1], point[2]]
            
        # keep heatwave values only if some land points have been found  
        # print(fieldHW.shape, mask.shape)
        fieldHW = np.ma.array(fieldHW, mask=mask[:,0, :,:]) #)
        #fieldHW = np.ma.array(fieldHW) #)
        ampli = area_av(fieldHW, 1, 2, lat, lon, opt="mean")
        
        #print(np.ma.where(fieldHW>0))
        if np.ma.where(fieldHW>0)[0].size != 0:
            start = np.min(np.ma.where(fieldHW>0)[0])
            end = np.max(np.ma.where(fieldHW>0)[0])
        else:
            start = None
            end = None
        #print('ampli : ', ampli)
        #check if HW is only oceanic
        
        if np.ma.sum(ampli)>0:
            HWampli.append(ampli)
            HWstart.append(start) #start date
            HWend.append(end) #end date
            fieldHWlst.append(np.ma.sum(fieldHW, axis=0))
        else:
            print("Ce test n est pas verifie")
    return(HWampli, HWstart, HWend, fieldHWlst)



def export_allHWs_iyear(iyear, nHW_i, dirout, ampliHWs_iyear, fieldHWslist_iyear, args, args2):
    expname, reg_name, memb_str, season, parameters_str, start_year, lats_reg, lons_reg = args
    ndayseas, nlat, nlon = args2
    fileout = dirout+'Ampli_Fields_'+parameters_str+"_%i_%i.nc"%(iyear+start_year,iyear+start_year+1)
    if len(glob(fileout))==1:
        os.remove(fileout)
    fout=netCDF4.Dataset(fileout, "w")
    
    # Create Dimensions
    lat = fout.createDimension('lat', nlat)
    lon = fout.createDimension('lon', nlon) 
    #rea = fout.createDimension('realization', nmemb) 
    timedim = fout.createDimension('time', None)
    HWdim = fout.createDimension('nHW', nHW_i) #To adapt for each year
    days = fout.createDimension('days', ndayseas) # Check with chloe, maybe use time instead
    # Create Variables
    times = fout.createVariable('time', np.float64, ('time',)) 
    latitudes = fout.createVariable('lat', np.float32, ('lat',)) 
    longitudes = fout.createVariable('lon', np.float32,  ('lon',)) 
    HWvar = fout.createVariable('nHW', np.int_, ('nHW',))
    daysvar = fout.createVariable('days', np.float32, ('days'))
       
    
    # Time variable
    # Useless, keep just in case
    times.units = 'hours since 0001-01-01 00:00:00' 
    times.calendar = 'gregorian' 
    times[:]=date2num(datetime((start_year+iyear),12,1), units = times.units, calendar = times.calendar)
    
    # Fill Variables
    
    latitudes[:] = lats_reg
    lonaux = lons_reg
    lonaux[lonaux<0]=lonaux[lonaux<0]+360
    longitudes[:] = lonaux
    latitudes.units = 'degree_north'  
    longitudes.units = 'degree_east'
    
    # WHAT VARIABLE DO WE WANT TO EXPORT?
    # Create Export Variables
    

    Amplifile = fout.createVariable('Ampli', np.float32, ('nHW', 'days'))
    Fieldfile = fout.createVariable('Field', np.float32, ('nHW', 'lat', 'lon'))
    
    fout.description = 'HW Amplitude and Fields computed from HWMI indexes files' #find something better with Chloe
    fout.history = 'computed from python script by C.Prodhomme and S.Lecestre' + time.ctime(time.time())
    fout.source = 'HW Amplitude and Fields for '+ expname
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    
    Amplifile.units = 'Amplitude per days'
    Fieldfile.units = 'Amplitude per surface'
    
    fout.description = 'HW Amplitude and Fields computed from HWMI indexes files' #find something better with Chloe
    fout.history = 'computed from python script by C.Prodhomme and S.Lecestre' + time.ctime(time.time())
    fout.source = 'HW Amplitude and Fields for '+ expname
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    
    Amplifile.units = 'Amplitude per days'
    Fieldfile.units = 'Amplitude per surface'
    
    
    # Write Ampli and Fields
    # To do so, transform list of d-arrays into (d+1)-arrays
    ampliaux = np.zeros((nHW_i, ndayseas))
    maskFalse2 = np.zeros((nHW_i, nlat, nlon), dtype = bool)
    fieldaux = np.ma.array(np.zeros((nHW_i, nlat, nlon)), mask = maskFalse2)
    for iHW in range(nHW_i):
        ampliaux[iHW, :] = np.array(ampliHWs_iyear[iHW][:])
        fieldaux[iHW, :, :] = np.array(fieldHWslist_iyear[iHW][:,:])
        fieldaux[iHW, :, :].mask = np.array(fieldHWslist_iyear[iHW][:,:].mask) #np.array to make an indpt copy
    Amplifile[:,:] = ampliaux[:,:]
    Fieldfile[:,:,:] = fieldaux[:,:,:]
    
    fout.close()
    print(fileout)
    return 
    
def calc_HW_MY(mod, mask, lat, lon, args, allowdist=1):
    """
    mod : data np.array of shape (nyear, ndayseas, nmemb, nlon, nlat)
    """
    expname, reg_name, memb_str, season, parameters_str, start_year, lats_reg, lons_reg = args
    nyear, ndayseas, nmemb, nlat, nlon = mod.shape
    
    #HWamplimembyear = []
    #HWstartmembyear = []
    #HWendmembyear = []
    #fieldlstmembyear = []
    for imemb in range(nmemb):
        HWampliyears = []
        HWstartyears = []
        HWendyears = []
        fieldlstyears = []
        
        def thread(iyear):
            print('thread iyear : ', iyear, ' started')
            field=mod[iyear,:, imemb,:,:]
            HW = detectHW1year(field, lat, lon, args, allowdist)
            #print('HWs length :', len(HW))
            HWampli_iyear, HWstart_iyear, HWend_iyear, fieldHWslst_iyear = computeHWcara(field, mask, lat, lon, HW)
            HWampliyears.append(HWampli_iyear)
            HWstartyears.append(HWstart_iyear)
            HWendyears.append(HWend_iyear)
            fieldlstyears.append(fieldHWslst_iyear)
            nHW_i = len(HW)
            dirout = "/cnrm/pastel/USERS/lecestres/NO_SAVE/data/"+expname+'/'+memb_str+'/'+season+'/Ampli_Fields_'+parameters_str+'/'
            if not(os.path.isdir(dirout)):
                os.makedirs(dirout)
            args2 = ndayseas, nlat, nlon
            export_allHWs_iyear(iyear, nHW_i, dirout, HWampli_iyear, fieldHWslst_iyear, args, args2)
            print('thread iyear : ', iyear, ' done')
        CPUs = os.cpu_count()
        Parallel(n_jobs=min(CPUs, nyear), verbose = 20, require='sharedmem', mmap_mode='w+')(delayed(thread)(iyear) for iyear in range(nyear))
        #HWamplimembyear.append(HWampliyear)
        #HWstartmembyear.append(HWstartyear)
        #HWendmembyear.append(HWendyear)
        #fieldlstmembyear.append(fieldlstyear)
        
            
    #return(HWamplimembyear, HWstartmembyear, HWendmembyear, fieldlstmembyear)
    return()
    
def spacelist_neighbors_highdist(point, allowdist=3):
    """
    point: list of length 3 : (time,lat,lon)
    """
    if allowdist == 1:
        neisouth = (point[0], (point[1]-1), (point[2] +180)%360 - 180)
        neinorth = (point[0], (point[1]+1), (point[2] +180)%360 - 180)
        neiwest = (point[0], (point[1]), (point[2]-1 +180)%360 - 180)
        neieast = (point[0], (point[1]), (point[2]+1 +180)%360 - 180)
        neiNW = (point[0], (point[1]+1), (point[2]-1 +180)%360 - 180)
        neiNE = (point[0], (point[1]+1), (point[2]+1 +180)%360 - 180)
        neiSW = (point[0], (point[1]-1), (point[2]-1 +180)%360 - 180)
        neiSE = (point[0], (point[1]-1), (point[2]+1 +180)%360 - 180)
        listnei = [neisouth, neinorth, neiwest, neieast, neiNW, neiNE, neiSW, neiSE]
    
    else:
        spaceneighbours = []
        allowdistx=allowdist+1
        allowdisty=allowdist+1
        for idistx in range(allowdistx):
            for idisty in range(allowdisty):
                # adding space neighbours
                nei1 = (point[0], (point[1]-idisty), (point[2]-idistx + 180)%360 - 180)
                nei2 = (point[0], (point[1]-idisty), (point[2]+idistx + 180)%360 - 180)
                nei3 = (point[0], (point[1]+idisty), (point[2]+idistx + 180)%360 - 180)
                nei4 = (point[0], (point[1]+idisty), (point[2]-idistx + 180)%360 - 180)
                spaceneighbours += [nei1, nei2, nei3, nei4]
        listnei = list(set(listnei)) #sort and leave duplicates terms 
        if point in listnei:
            #should be always true
            listnei.remove(point)
    
    return(listnei)

    

def spacelist_neighbors_highdist2(point, allowdist=3):
    """
    point: list of length 3 : (time,lat,lon)
    """
    if allowdist == 1:
        neisouth = (point[0], (point[1]-1), point[2]%360)
        neinorth = (point[0], (point[1]+1), point[2]%360)
        neiwest = (point[0], (point[1]), (point[2]-1)%360)
        neieast = (point[0], (point[1]), (point[2]+1)%360)
        neiNW = (point[0], (point[1]+1), (point[2]-1)%360)
        neiNE = (point[0], (point[1]+1), (point[2]+1)%360)
        neiSW = (point[0], (point[1]-1), (point[2]-1)%360)
        neiSE = (point[0], (point[1]-1), (point[2]+1)%360)
        listnei = [neisouth, neinorth, neiwest, neieast, neiNW, neiNE, neiSW, neiSE]
    
    else:
        spaceneighbours = []
        allowdistx=allowdist+1
        allowdisty=allowdist+1
        for idistx in range(allowdistx):
            for idisty in range(allowdisty):
                # adding space neighbours
                nei1 = (point[0], (point[1]-idisty), (point[2]-idistx)%360)
                nei2 = (point[0], (point[1]-idisty), (point[2]+idistx)%360)
                nei3 = (point[0], (point[1]+idisty), (point[2]+idistx)%360)
                nei4 = (point[0], (point[1]+idisty), (point[2]-idistx)%360)
                spaceneighbours += [nei1, nei2, nei3, nei4]
        listnei = list(set(listnei)) #sort and leave duplicates terms 
        if point in listnei:
            #should be always true
            listnei.remove(point)
    
    return(listnei)