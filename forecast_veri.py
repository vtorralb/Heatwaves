import numpy as np
import numpy.ma as ma

def comp_clim(varin, posclim, cross_val=-1):
    """ Compute climatology with or without Cross validation:
    - Varin: input variable on which to compute the climatology
    - posclim: position of the time dimension where to compute the climatology
    - cross val: if -1, the clim is computed using all the years if >=0 indice of 
    the year to be exlueded when computing the climatology"""
    
    #create a local copy
    #var=ma.array(varin, copy=True)
    if cross_val!=-1:
        #mask_cv=mask=np.concatenate((arr_inf,arr_cv, arr_sup), axis=posclim)
        var=np.delete(varin, cross_val, axis=posclim)
        #var.mask=mask_cv #(var.mask+mask_cv)>=1
    else:
        var=varin
    #compute climatology excluding missing start dates
    clim=np.ma.mean(var, axis=posclim)
    #climext=np.expand_dims(clim, axis=posclim).repeat(var.shape[posclim], axis=posclim)

    return(clim)
    #anom=var-climext
    #return(anom)
    
def comp_anom_cv(varin, posanom, posens=-1):
    """ Compute anomaly in Cross validation:
    - Varin: input variable on which to compute the anomalies
    - posanom: position of the time dimension where to compute the anomalies
    - posens: if -1 the anomalies are computed separatly for each members
                if >=0 give the position of the member dimension and ensemble mean 
                is computed"""
    
    
    #compute ensemble mean
    var=ma.array(varin, copy=True)
    if posens != -1:
        var=np.ma.mean(var, axis=posens)
    
    #print(var.mask)
    dims=var.shape
    climext=np.expand_dims(comp_clim(var, posanom, cross_val=0), axis=posanom)
    for itimes in range(1,dims[posanom]):
        climext=np.concatenate((climext, np.expand_dims(comp_clim(var, posanom, cross_val=itimes), axis=posanom)), axis=posanom)
        
    return(var-climext)


def comp_anom(varin, posanom, posens=-1):
    """ Compute anomalies WITHOUT cross validation
    - Varin: input variable on which to compute the anomalies
    - posanom: position of the time dimension where to compute the anomalies
    - posens: if -1 the anomalies are computed separatly for each members
                if >=0 give the position of the member dimension and ensemble mean 
                is computed"""
    
    #compute ensemble mean
    var=ma.array(varin, copy=True)
    if posens != -1:
        var=np.ma.mean(var, axis=posens)
    
    #print(var.mask)
    dims=var.shape
    climext=np.expand_dims(comp_clim(var, posanom, cross_val=-1), axis=posanom)
    for itimes in range(1,dims[posanom]):
    
        climext=np.concatenate((climext, np.expand_dims(comp_clim(var, posanom, cross_val=-1), axis=posanom)), axis=posanom)
    

    return(var-climext)


