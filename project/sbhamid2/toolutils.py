import constants as gpsconstants
import numpy as np
import datetime as datetime
import pytz as pytz


def FindSat(ephem, prn, times):
    # Function to coarse calculate satellite positions given the GPS almanac.
    # The calculation can either be done for one satellite and multiple times,
    # or for multiple satellites at one time.
    # Inputs:
    #       ephem - an ephem structure containing the ephemeris information for a satellite.  
    #               See ECE456_fileutils.py for a description of the structure format.
    #       times - the GPS time to compute the satellite position(s) for. t can be a single value or a vector.
    #
    # Outputs:
    #       satLoc - a (N x 5) array containing the GPStime, X, Y, and Z locations in ECEF
    #       coordinates calculated from the ephemerides.  X, Y, and Z are
    #       given in meters.  GPStime is in seconds since the midnight transition from the
    #       previous Saturday/Sunday.  There are N different solutions,
    #       determined by the length of the 't' vector.
    #               satLoc = [GPStime PRN ECEFx ECEFy ECEFz ;
    #                          . . . . .
    #                         GPStime PRN ECEFx ECEFy ECEFz];
    # Revision history:
    #       7-15-2015: Written by J. Makela based on MATLAB code

    # Load in GPS constants
    gpsconst = gpsconstants.gpsconsts()
    
    # The array to hold the satellite locations in
    satLoc = []
    
    # Define the delta time from TOE.  Correct for possible week crossovers.  Note that
    # there are 604800 seconds in a GPS week.  This is admittadly a bit of a hack.
    dt = times - ephem['TOE']
    if dt > 302400: 
        dt -= 604800.   # Time is in the next week
    if dt < -302400: 
        dt += 604800.  # Time is in the previous week
    
    # Calculate the mean anomaly
    M = ephem['M0'] + (np.sqrt(gpsconst.muearth) * ephem['sqrta']**-3) * dt
    
    # Compute the eccentric anomaly from mean anomaly using the Newton-Raphson method
    # to solve for E in:
    #  f(E) = M - E + e * sin(E) = 0
    E = M
    for i in np.arange(0,10):
        f = M - E + ephem['e'] * np.sin(E)
        dfdE = ephem['e']*np.cos(E) - 1.
        dE = -f / dfdE
        E = E + dE
    
    # Calculate the true anomaly from the eccentric anomaly
    sinnu = np.sqrt(1-ephem['e']**2)*np.sin(E)/(1-ephem['e']*np.cos(E))
    cosnu = (np.cos(E)-ephem['e'])/(1-ephem['e']*np.cos(E))
    nu = np.arctan2(sinnu,cosnu)
    
    # Calcualte the argument of latitude
    phi0 = nu + ephem['omega']
    phi = phi0
    
    # Calculate the longitude of ascending node
    Omega = ephem['Omega0'] - gpsconst.OmegaEDot*(times)+ephem['Omega_dot']*dt
    
    # Calculate orbital radius
    r = (ephem['sqrta']**2)*(1-ephem['e']*np.cos(E))
    
    # Calculate the inclination
    i = ephem['i0']
    
    # Find the position in the orbital plane
    xp = r*np.cos(phi)
    yp = r*np.sin(phi)
    
    # Find satellite position in ECEF coordinates
    ECEFx = xp*np.cos(Omega) - yp*np.cos(i)*np.sin(Omega)
    ECEFy = xp*np.sin(Omega) + yp*np.cos(i)*np.cos(Omega)
    ECEFz = yp*np.sin(i)
        
    satLoc = np.vstack((times,prn,ECEFx,ECEFy,ECEFz)).T
    
    return satLoc


def cuboid_data(center, size):
    """Create a data array for cuboid plotting.
    ============= ================================================
    Argument      Description
    ============= ================================================
    center        center of the cuboid, triple
    size          size of the cuboid, triple, (x_length,y_width,z_height)
    :type size: tuple, numpy.array, list
    :param size: size of the cuboid, triple, (x_length,y_width,z_height)
    :type center: tuple, numpy.array, list
    :param center: center of the cuboid, triple, (x,y,z) """

    # suppose axis direction: x: to left; y: to inside; z: to upper
    # get the (left, outside, bottom) point
    o = [a - b / 2 for a, b in zip(center, size)]
    #print('ref: ', o)
    
    # get the length, width, and height
    l, w, h = size
    x = [[o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in bottom surface
          [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in upper surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]],  # x coordinate of points in outside surface
         [o[0], o[0] + l, o[0] + l, o[0], o[0]]]  # x coordinate of points in inside surface
    
    y = [[o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in bottom surface
         [o[1], o[1], o[1] + w, o[1] + w, o[1]],  # y coordinate of points in upper surface
         [o[1], o[1], o[1], o[1], o[1]],          # y coordinate of points in outside surface
         [o[1] + w, o[1] + w, o[1] + w, o[1] + w, o[1] + w]]    # y coordinate of points in inside surface
    
    z = [[o[2], o[2], o[2], o[2], o[2]],                        # z coordinate of points in bottom surface
         [o[2] + h, o[2] + h, o[2] + h, o[2] + h, o[2] + h],    # z coordinate of points in upper surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]],                # z coordinate of points in outside surface
         [o[2], o[2], o[2] + h, o[2] + h, o[2]]]                # z coordinate of points in inside surface
    
    four_surfaces = {}
    fir_surf = np.transpose([  [o[0],     o[0] + l, o[0] + l,  o[0] ], 
                               [o[1],     o[1],     o[1],      o[1] ], 
                               [o[2],     o[2],     o[2] + h,  o[2] + h] ])
    four_surfaces['1'] = fir_surf
    
    sec_surf = np.transpose([  [o[0],     o[0] + l, o[0] + l,  o[0]], 
                               [o[1] + w, o[1] + w, o[1] + w,  o[1] + w], 
                               [o[2],     o[2],     o[2] + h,  o[2] + h] ])
    four_surfaces['2'] = sec_surf
    
    thd_surf = np.transpose([  [o[0] + l, o[0] + l, o[0] + l,  o[0] + l], 
                               [o[1] + w, o[1],     o[1],      o[1] + w], 
                               [o[2],     o[2],     o[2] + h,  o[2] + h] ])
    four_surfaces['3'] = thd_surf
    
    fth_surf = np.transpose([  [o[0],     o[0],     o[0],      o[0]], 
                               [o[1] + w, o[1],     o[1],      o[1] + w], 
                               [o[2],     o[2],     o[2] + h,  o[2] + h] ])
    four_surfaces['4'] = fth_surf
    four_surfaces['c'] = center
    
    #print('first_surf: ', fir_surf)
    #print('second_surf: ', sec_surf)
    #print('third_surf: ', thd_surf)
    #print('fourth_surf: ', fth_surf)
    
    return np.array(x), np.array(y), np.array(z), four_surfaces


def intersect_LinePlane(rayDirection, rayPoint, planeNormal, planePoint, epsilon=1e-6): 
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        return np.array([-1,-1,-1])
        #raise RuntimeError("no intersection or line is within plane")
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    
    return Psi

def get_bldnormal(plane, center):
        mdp = intersect_lines(plane[0], plane[2], plane[1], plane[3])
        #print('middle point: ', mdp, surfs['c'])              
        bld_normal = center-mdp
        #print('bld_normal: ', bld_normal)
    
        return mdp, bld_normal
    
def get_angle(action, Rx_ENU, surfs): 
    
    store_normals = []
    store_wrx = []
    store_idx = []
    for i in range(1,5):
        mdp, bld_normal = get_bldnormal(surfs[str(i)], surfs['c'])
        ang_wbld = np.dot(bld_normal, action)/np.linalg.norm(bld_normal) 
        #print('mdp: ', mdp, 'bld_normal: ', bld_normal, 'action: ', action, 'ang_wbld: ', ang_wbld)
        
        if (abs(ang_wbld)<=1e-7): 
            rx_vector = mdp-np.array([Rx_ENU[0,0], Rx_ENU[1,0], Rx_ENU[2,0]])
            ang_wrx = np.dot(rx_vector, bld_normal)/ (np.linalg.norm(bld_normal)*np.linalg.norm(rx_vector))
            store_normals.append(bld_normal)
            store_wrx.append(ang_wrx)
            store_idx.append(i)
            #print('rx_vector: ', np.linalg.norm(rx_vector), ang_wrx, np.rad2deg(math.acos(ang_wrx)))
            #if (ang_wrx>=0.0): 
            #    return ang_wbld, ang_wrx, bld_normal, i
        
    return store_wrx, store_normals, store_idx

def mirror_image(poly_pts, pnt_eval): 
    [a,b,c,d] = get_plane(poly_pts[0], poly_pts[1], poly_pts[2])
    
    x1, y1, z1 = pnt_eval
    
    k =(d - a*x1 - b*y1 - c*z1)/float((a * a + b * b + c * c)) 
    #print('On plane: ', [a*k+x1, b*k+y1, c*k+z1])
    x2 = 2*a*k + x1 
    y2 = 2*b*k + y1 
    z2 = 2*c*k + z1 
    #x3 = 2 * x2-x1 
    #y3 = 2 * y2-y1 
    #z3 = 2 * z2-z1 
    
    return [x2, y2, z2]
    
    
def get_plane(p1, p2, p3): 
    v1 = p3 - p1
    v2 = p2 - p1

    # the cross product is a vector normal to the plane
    cp = np.cross(v1, v2)
    a, b, c = cp

    # This evaluates a * x3 + b * y3 + c * z3 which equals d
    d = np.dot(cp, p3)

    #print('The equation is {0}x + {1}y + {2}z = {3}'.format(a, b, c, d))

    return [a,b,c,d]


# A utility function to calculate 
# area of triangle formed by (x1, y1), 
# (x2, y2) and (x3, y3) 
def area(x1, y1, x2, y2, x3, y3): 
    return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0) 


#area of polygon poly
def poly_area(poly, unit_normal):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal)
    return abs(result/2)

def NotBlocked(poly_pts, poly_normal, pt_eval): 
    unit_normal = poly_normal/np.linalg.norm(poly_normal)
    
    # Calculate area of rectangle A = 0 B =1 C = 2 D = 3 
    A = poly_area(poly_pts, unit_normal) 
    
    # Calculate area of triangle PAB 
    A1_pts = [pt_eval, poly_pts[0], poly_pts[1]]
    A1 = poly_area(A1_pts, unit_normal) 

    # Calculate area of triangle PBC 
    A2_pts = [pt_eval, poly_pts[1], poly_pts[2]]
    A2 = poly_area(A2_pts, unit_normal) 

    # Calculate area of triangle PCD 
    A3_pts = [pt_eval, poly_pts[3], poly_pts[2]]
    A3 = poly_area(A3_pts, unit_normal) 

    # Calculate area of triangle PDA 
    A4_pts = [pt_eval, poly_pts[3], poly_pts[0]]
    A4 = poly_area(A4_pts, unit_normal)
    diff = abs(A1+A2+A3+A4-A)
    #print('All areas: ', A, A1, A2, A3, A4, diff)

    # Check if sum of A1, A2, A3 
    # and A4 is same as A 
    if (diff<1e-3): 
        return False ## blocked
    else: 
        return True ## not blocked



#####
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

def test():
    center = [0, 0, 0]
    length = 32 * 2
    width = 50 * 2
    height = 100 * 2
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X, Y, Z, surfs = cuboid_data(center, (length, width, height))
    #print('surf: ', surfs)
    
    ax.plot_surface(X, Y, Z, color='b', rstride=1, cstride=1, alpha=0.1)
    ax.set_xlabel('X')
    ax.set_xlim(-100, 100)
    ax.set_ylabel('Y')
    ax.set_ylim(-100, 100)
    ax.set_zlabel('Z')
    ax.set_zlim(-100, 100)
    plt.show()
    
    return surfs
    
    
    
#
# intersections.py
#
# Python for finding line intersections
#   intended to be easily adaptable for line-segment intersections
#

import math

def intersect_lines(pt1, pt2, pt3, pt4): 
    A = np.transpose([-pt2+pt1, pt4-pt3 ])
    B = np.array(pt1-pt3).reshape(-1,1)

    st = np.matmul(np.linalg.inv(np.matmul(np.transpose(A),A)), np.matmul(np.transpose(A),B))
    mid_pnt = (1-st[0])*pt1 + st[0]*pt2
    mid_pnt2 = (1-st[1])*pt3 + st[1]*pt4
    #print('sol: ', st)
    #print('vals: ', mid_pnt, mid_pnt2)    
    
    #if(np.array_equal(mid_pnt,mid_pnt2)):
    return mid_pnt
