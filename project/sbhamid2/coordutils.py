import numpy as np
import constants as gpsconstants
import datetime as datetime
import pytz as pytz

_IE24  = {'a': 6378388.0, 'invf': 297.0}
_IE67  = {'a': 6378160.0, 'invf': 298.247}
_WGS72 = {'a': 6378135.0, 'invf': 298.26}
_GRS80 = {'a': 6378137.0, 'invf': 298.257222101}
_WGS84 = {'a': 6378137.0, 'invf': 298.257223563}

def ECEF_to_LLA(posvel_ECEF, ellipsoid=_WGS84, normalize=False, in_degrees=True):
    """
    Returns lla position in a record array with keys 'lat', 'lon' and 'alt'. 
    lla is calculated using the closed-form solution.
    
    For more information, see
     - Datum Transformations of GPS Positions
       https://microem.ru/files/2012/08/GPS.G1-X-00006.pdf
       provides both the iterative and closed-form solution.

     >>> #ECE Building and Mount Everest
     >>> array([(40.11497089608554, -88.22793631642435, 203.9925799164921), 
                (27.98805865809616, 86.92527453636706, 8847.923165871762)], 
         dtype=[('lat', '<f8'), ('lon', '<f8'), ('alt', '<f8')])
    
    @type  posvel_ECEF: np.ndarray
    @param posvel_ECEF: ECEF position in array of shape (3,N)
    @type  ellipsoid: dict
    @param ellipsoid: Reference ellipsoid (eg. _WGS84 = {'a': 6378137.0, 'invf': 298.257223563}). 
    Class headers contains some pre-defined ellipsoids: _IE24, _IE67, _WGS72, _GRS80, _WGS84
    @type  normalize: bool
    @param normalize: Its default value is False; True will cause longitudes returned 
    by this function to be in the range of [0,360) instead of (-180,180]. 
    Setting to False will cause longitudes to be returned in the range of (-180,180].
    @rtype : numpy.ndarray
    @return: Position in a record array with keys 'lat', 'lon' and 'alt'.
    """
                                  
    from numpy import sqrt, cos, sin, pi, arctan2 as atan2
    #python subtlety only length 1 arrays can be converted to Python scalars
    #if we use the functions from math, we have to use map

    a    = ellipsoid['a']
    invf = ellipsoid['invf']
    f = 1.0/invf
    b = a*(1.0-f)
    e  = sqrt((a**2.0-b**2.0)/a**2.0)
    ep = sqrt((a**2.0-b**2.0)/b**2.0)
    
    xyz = np.asarray(posvel_ECEF)
    x = xyz[0,:]
    y = xyz[1,:]      
    z = xyz[2,:]
    
    # Create the record array.
    cols = np.shape(xyz)[1]
    lla  = np.zeros(cols, dtype = { 'names' : ['lat', 'lon', 'alt'], 
                                  'formats' : ['<f8', '<f8', '<f8']})
                                  
    lon = atan2(y, x)
    p = sqrt(x**2.0+y**2.0)
    theta = atan2(z*a,p*b)
    lat = atan2((z+(ep**2.0)*(b)*(sin(theta)**3.0)),(p-(e**2.0)*(a)*(cos(theta)**3.0)))
    N = a/sqrt(1.0-((e**2.0)*(sin(lat)**2.0)))
    alt = p/cos(lat)-N
    
    if normalize:
        lon = np.where(lon < 0.0, lon + 2.0*pi, lon)
    
    lla['alt'] = alt
    
    if in_degrees:
        lla['lat'] = lat*180.0/pi
        lla['lon'] = lon*180.0/pi
    else:
        lla['lat'] = lat
        lla['lon'] = lon        
    
    return lla

def LLA_to_ECEF(lat,lon,alt, ellipsoid=_WGS84):
    '''
    For more information, see
     - Datum Transformations of GPS Positions
       https://microem.ru/files/2012/08/GPS.G1-X-00006.pdf
       provides both the iterative and closed-form solution.

    '''
    from numpy import sqrt, cos, sin, pi, arctan2 as atan2
    #python subtlety only length 1 arrays can be converted to Python scalars
    #if we use the functions from math, we have to use map

    a    = ellipsoid['a']
    invf = ellipsoid['invf']
    f = 1.0/invf
    b = a*(1.0-f)
    e  = sqrt((a**2.0-b**2.0)/a**2.0)
    
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    alt = np.array(alt)
    
    N = a/sqrt(1-e**2*sin(lat)**2)
    X = (N+alt)*cos(lat)*cos(lon) 
    Y = (N+alt)*cos(lat)*sin(lon)
    Z = (((b**2)/(a**2))*N+alt)*sin(lat)
    
    posvel_ECEF = np.matrix(np.zeros([3,len(lat)]))
    posvel_ECEF[0,:] = X
    posvel_ECEF[1,:] = Y
    posvel_ECEF[2,:] = Z
    
    return posvel_ECEF

def ECEF_to_ENU(refState=None, curState=None, diffState=None):
    """
    Returns 3xN matrix in enu order
    - Toolbox for attitude determination 
       Zhen Dai, ZESS, University of Siegen, Germany

    @type  curState: 3x1 or 8x1 matrix
    @param curState: current position in ECEF                 
    @type  refState: 3x1 or 8x1 matrix
    @param refState: reference position in ECEF
    @rtype : tuple
    @return: (numpy.matrix (3,N) (ENU), numpy.matrix (3,3) R_ECEF2ENU)
    """  
        
    lla = ECEF_to_LLA(refState,in_degrees=False)
    lat = lla['lat'][0]
    lon = lla['lon'][0]

    slon = np.sin(lon)
    clon = np.cos(lon)
    slat = np.sin(lat)
    clat = np.cos(lat)

    R_ECEF2ENU = np.mat([[ -slon, clon, 0.0],
                         [ -slat*clon, -slat*slon, clat],
                         [  clat*clon,  clat*slon, slat]])
    
    xyz0 = refState[0:3,:]

    if (curState is not None) and (diffState is None):    
        xyz  = curState[0:3,:]
        dxyz = xyz - np.tile(xyz0, (1,np.shape(curState)[1]))
    elif (curState is None) and (diffState is not None):
        dxyz = diffState[0:3,:]
    else:
        print("Unknown error, shape of states:")
        print('refState: '+str(np.shape(refState)))
        print('curState: '+str(np.shape(curState)))
        print('diffState: '+str(np.shape(diffState)))
        
    matENU = R_ECEF2ENU*dxyz

    return matENU, R_ECEF2ENU
    
def ENU_to_ECEF(refState=None, diffState=None, R_ECEF2ENU= None):
    """
    """  
    if R_ECEF2ENU is None:
        lla = ECEF_to_LLA(refState, in_degrees=False)
        lat = lla['lat'][0]
        lon = lla['lon'][0]
    
        slon = np.sin(lon)
        clon = np.cos(lon)
        slat = np.sin(lat)
        clat = np.cos(lat)
    
        R_ECEF2ENU = np.mat([[ -slon, clon, 0.0],
                             [ -slat*clon, -slat*slon, clat],
                             [  clat*clon,  clat*slon, slat]])
                             
    R_ENU2ECEF = R_ECEF2ENU.T
    
    xyz0 = refState[0:3,:]
    dxyz = diffState[0:3,:]
        
    matECEF = R_ENU2ECEF*dxyz+np.tile(xyz0, (1,np.shape(dxyz)[1]))

    return matECEF

def ENU_to_elaz(ENU):
    
    ENU = np.asarray(ENU)
    east  = ENU[0,:]
    north = ENU[1,:]      
    up    = ENU[2,:]
    
    # Create the record array.
    cols = np.shape(ENU)[1]
    elazd = np.zeros(cols, dtype = { 'names' : ['ele', 'azi', 'dist'], 
                                   'formats' : ['<f8', '<f8', '<f8']})
    
    horz_dist = np.sqrt(east**2 + north**2)
    elazd['ele']  = np.arctan2(up, horz_dist)        
    elazd['azi']  = np.arctan2(east, north)
    elazd['dist'] = np.sqrt(east**2 + north**2 + up**2)

    return elazd

def elaz(Rx, Sats):
    """
    Function: elaz(Rx, Sats)
    ---------------------
    Calculate the elevation and azimuth from a single receiver to multiple satellites.

    Inputs:
    -------
        Rx   : 1x3 vector containing [X, Y, Z] coordinate of receiver
        Sats : Nx3 array  containing [X, Y, Z] coordinates of satellites
        

    Outputs:
    --------
        elaz : Nx3 array containing the elevation and azimuth from the
               receiver to the requested satellites. Elevation and azimuth are
               given in decimal degrees.   

    Notes:
    ------
        Based from Jonathan Makela's GPS_elaz.m script

    History:
    --------
        7/15/15 Created, Jonathan Makela (jmakela@illinois.edu)

    """
    
    # check for 1D case:
    dim = len(Rx.shape)
    if dim == 1:
        Rx = np.reshape(Rx,(1,3))
        
    dim = len(Sats.shape)
    if dim == 1:
        Sats = np.reshape(Sats,(1,3))
    
    # Convert the receiver location to WGS84
    lla = ecef2lla(Rx)

    # Create variables with the latitude and longitude in radians
    lat = np.deg2rad(lla[0,0])
    lon = np.deg2rad(lla[0,1])

    # Create the 3 x 3 transform matrix from ECEF to VEN
    VEN = np.array([[ np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)],
                    [-np.sin(lon), np.cos(lon), 0.],
                    [-np.sin(lat)*np.cos(lon), -np.sin(lat)*np.sin(lon), np.cos(lat)]])

    # Replicate the Rx array to be the same size as the satellite array
    Rx_array = np.ones_like(Sats) * Rx

    # Calculate the pseudorange for each satellite
    p = Sats - Rx_array
    
    # Calculate the length of this vector
    n = np.array([np.sqrt(p[:,0]**2 + p[:,1]**2 + p[:,2]**2)])
    
    # Create the normalized unit vector
    p = p / (np.ones_like(p) * n.T)
    
    # Perform the transform of the normalized psueodrange from ECEF to VEN
    p_VEN = np.dot(VEN, p.T)
    
    # Calculate elevation and azimuth in degrees
    ea = np.zeros([Sats.shape[0],2])
    ea[:,0] = np.rad2deg((np.pi/2. - np.arccos(p_VEN[0,:])))
    ea[:,1] = np.rad2deg(np.arctan2(p_VEN[1,:],p_VEN[2,:]))
    
    return ea
    
def ecef2lla(xyz):
    """
    Function: ecef2lla(xyz)
    ---------------------
    Converts ECEF X, Y, Z coordinates to WGS-84 latitude, longitude, altitude

    Inputs:
    -------
        xyz : 1x3 vector containing [X, Y, Z] coordinate
        

    Outputs:
    --------
        lla : 1x3 vector containing the converted [lat, lon, alt]
              (alt is in [m])  

    Notes:
    ------
        Based from Jonathan Makela's GPS_WGS84.m script

    History:
    --------
        7/21/12 Created, Timothy Duly (duly2@illinois.edu)

    """
    # Load in GPS constants
    gpsconst = gpsconstants.gpsconsts()

    x = xyz[0]
    y = xyz[1]
    z = xyz[2]

    run = 1

    lla = np.zeros_like(xyz)
    # Compute longitude:
    lla[1] = np.rad2deg(np.arctan2(y,x))

    # guess iniital latitude (assume you're on surface, h=0)
    p = np.sqrt(x**2+y**2)
    lat0 = np.arctan(z/p*(1-gpsconst.e**2)**-1)

    while (run == 1):
        # Use initial latitude to estimate N:
        N = gpsconst.a**2 / np.sqrt(gpsconst.a**2 * (np.cos(lat0))**2+gpsconst.b**2*(np.sin(lat0))**2)

        # Estimate altitude 
        h = p/np.cos(lat0)-N
    
        # Estimate new latitude using new height:
        lat1 = np.arctan(z/p*(1-((gpsconst.e**2*N)/(N+h)))**-1)
        
        if (abs(lat1-lat0) < gpsconst.lat_accuracy_thresh).any():
            run = 0
        
        # Replace our guess latitude with most recent estimate:
        lat0 = lat1

    # load output array with best approximation of latitude (in degrees)
    # and altiude (in meters)
    
    lla[0] = np.rad2deg(lat1)
    lla[2] = h
    
    return lla
    
def lla2ecef(lla):
    """
    Function: lla2ecef(lla)
    ---------------------
    Converts WGS-84 latitude, longitude, altitude to ECEF X, Y, Z coordinates

    Inputs:
    -------
        lla : 1x3 vector containing the converted [lat, lon, alt]
              (alt is in [m])  

    Outputs:
    --------
        xyz : 1x3 vector containing [X, Y, Z] coordinate

    Notes:
    ------
        Based from Jonathan Makela's GPS_ECEF.m script

    History:
    --------
        7/21/12 Created, Timothy Duly (duly2@illinois.edu)
        9/11/12 Updated to include vectorization.

    """
    
    # Load in GPS constants
    gpsconst = gpsconstants.gpsconsts()
    
    # check for 1D case:
    dim = len(lla.shape)
    if dim == 1:
        lla = np.reshape(lla,(1,3))

    # convert lat and lon to radians
    lat = np.deg2rad(lla[:,0])
    lon = np.deg2rad(lla[:,1])
    alt = lla[:,2];

    xyz = np.array(np.zeros(lla.shape))

    N = gpsconst.a**2/np.sqrt((gpsconst.a*np.cos(lat))**2+(gpsconst.b*np.sin(lat))**2)

    # Calculate the X-coordinate
    xyz[:,0] = (N+alt)*np.cos(lat)*np.cos(lon)

    # Calculate the Y-coordinate
    xyz[:,1] = (N+alt)*np.sin(lon)*np.cos(lat)

    # Calculate the Z-coordinate
    xyz[:,2] = (N*(1-gpsconst.e**2)+alt)*np.sin(lat)
    
    return np.array(xyz)
    
def utc2gps(dt, leapSeconds = 17):
    """
    Function: utc2gps(dt, leapSeconds)
    ---------------------
    Return the gpsTime and gpsWeek based on the requested time. Based, in part
    on https://www.lsc-group.phys.uwm.edu/daswg/projects/glue/epydoc/lib/python2.4/site-packages/glue/gpstime.py

    Inputs:
    -------
        dt : a datetime (in UTC) to be converted
        leapSeconds : (optional; default = 17) correction for GPS leap seconds
        
    Outputs:
    --------
        gpsTime : the converted GPS time of week [sec] 
        gpsWeek - the GPS week number (without considering rollovers)

    Notes:
    ------
        Based from Jonathan Makela's GPS_GMT2GPS_week.m script

    History:
    --------
        7/15/15 Created, Jonathan Makela (jmakela@illinois.edu)
        
    ToDo:
    --------
        1) Make leapSeconds calculate automatically based on a historical table and the requested date

    """
    
    # Define the GPS epoch
    gpsEpoch = datetime.datetime(1980,1,6,0,0,0)
    gpsEpoch = gpsEpoch.replace(tzinfo = pytz.utc)
    
    # Check if requested time is timezone aware.  If not, assume it is a UT time
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo = pytz.utc)
    
    # Calculate the time delta from epoch
    delta_t = dt - gpsEpoch

    # The total number of seconds (add in GPS leap seconds)
    secsEpoch = delta_t.total_seconds()+leapSeconds
    
    # The gpsTime is the total seconds since epoch mod the number of seconds in a week
    secsInWeek = 604800.
    gpsTime = np.mod(secsEpoch, secsInWeek)
    gpsWeek = int(np.floor(secsEpoch/secsInWeek))

    return gpsTime, gpsWeek