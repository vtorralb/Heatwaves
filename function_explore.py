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