import numpy as np
import math 
import coordutils 
import toolutils 
import matplotlib.pyplot as plt

def perform_least_sqrs(satPos, pseudoranges, rxPos0, iterations=10):

    sats =  [*pseudoranges] #list(pseudoranges.keys())
    numSats = len(sats)
    
    A = np.matrix(np.zeros((numSats,3)))
    b = np.matrix(np.zeros((numSats,1)))
    rxPos = rxPos0
    
    # Iteratively get the receiver location.
    for i in range(iterations):
        # Gather information from each satellite to initialize the b vector and A matrix.
        for idx in range(numSats):
            #print('prn: ', sats[idx], ' satPos:', satPos[idx], ' rxPos: ', np.linalg.norm(satPos[idx]-rxPos0), \
            #      ' pseudo: ', pseudoranges[str(sats[idx])])
            
            # Fill in values to b vector.
            b[idx,0] = pseudoranges[str(sats[idx])] - np.linalg.norm(satPos[idx]-rxPos) 

            # Fill in values to A matrix (linearized equation).
            A[idx,0:3] = ( -(satPos[idx] - rxPos)/(np.linalg.norm(satPos[idx]-rxPos)) ).T

        # If A is not full rank, we cannot proceed.
        if np.linalg.matrix_rank(A) != 3:
            print('Error: the linear least squares estimate of the receiver',
                  'position requires that the geometry A matrix be of rank 4. ',
                  'A is currently of rank: '+str(np.linalg.matrix_rank(A)))
            print('A matrix:'+str(A))

        # Run linear least squares to get the position update.
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)

        # Now apply the position update.
        rxPos = rxPos + np.array([x[0,0], x[1,0], x[2,0]])
        
        if np.linalg.norm(x) < 1.0e-7:
            break
    
    if np.linalg.norm(x) > 1.0e-5:
        print('Error: update is greater than 1.0e-5m after '+str(iterations)+' iterations.')

    return rxPos


def generate_trajectory(speed, traj): #blk_len, 
    n = 0
    remspeed = speed
    storeENU = []
    while( n<(len(traj)-1) ): 
        sourceENU = traj[n]
        destENU = traj[n+1] 
    
        dist = np.linalg.norm(destENU-sourceENU)
        losENU = (1/dist)*(destENU-sourceENU) 
        
        m = 1
        while(1): 
            if(m==1):
                if(n!=0):
                    coeffs = [1, 2*np.dot(resENU, losENU), np.linalg.norm(resENU)**2-speed**2]
                    rts = np.roots(coeffs)
                    if((rts==0).any()):
                        remspeed = 0.0
                    else:
                        remspeed = rts[rts>0][0]
                newENU = (sourceENU + remspeed*losENU)
            else: 
                newENU = (storeENU[-1] + speed*losENU)
            
            if(np.linalg.norm(sourceENU-newENU)<dist): 
                storeENU.append([newENU[0], newENU[1], newENU[2]])
            else:
                resENU = (dist-np.linalg.norm(sourceENU-storeENU[-1]))*losENU
                break
            m +=1
        n +=1        
    return storeENU 


def gen_satpositions(data, t_gps, el_mask, Ref_ECEF, Rx_ECEF):
    Rx_ENU, Rx_ECEF2ENU = coordutils.ECEF_to_ENU(Ref_ECEF, Rx_ECEF)
    
    visible_sats = {}
    visible_sats['prn'] = []
    visible_sats['satLoc'] = []
    visible_sats['pr_true'] = []
    visible_sats['elaz'] = []
                 
    for j in range(32): 
        prn = int(j)
        toe_diff = np.transpose(data['Toe'].values)[prn] - t_gps
        toe_diff[np.isnan(toe_diff)]=1e10
        idx = np.argmin(abs(toe_diff))
    
        ephem = {}
        ephem['TOE'] = int(data['Toe'].values[idx][prn])
        ephem['M0'] = data['M0'].values[idx][prn]
        ephem['sqrta'] = data['sqrtA'].values[idx][prn]
        ephem['e'] = data['Eccentricity'].values[idx][prn]
        ephem['omega'] = data['omega'].values[idx][prn]
        ephem['Omega0'] = data['Omega0'].values[idx][prn]
        ephem['i0'] = data['Io'].values[idx][prn]
        ephem['Omega_dot'] = data['OmegaDot'].values[idx][prn]
    
        sat_ECEF = toolutils.FindSat(ephem, prn, t_gps)
        sat_ENU, sat_ECEF2ENU = coordutils.ECEF_to_ENU( Ref_ECEF, sat_ECEF[0][2:5].reshape(-1,1))
    
        satrx, _ = coordutils.ECEF_to_ENU( Rx_ECEF, sat_ECEF[0][2:5].reshape(-1,1))
        sat_elaz = coordutils.ENU_to_elaz(satrx)
        if np.rad2deg(sat_elaz[0][0])>el_mask: 
            visible_sats['prn'].append(prn)
            visible_sats['satLoc'].append([ sat_ENU[0,0], sat_ENU[1,0], sat_ENU[2,0] ])
            visible_sats['pr_true'].append(np.linalg.norm(sat_ENU-Rx_ENU))
            visible_sats['elaz'].append([ np.rad2deg(sat_elaz[0][0]), np.rad2deg(sat_elaz[0][1]) ])

    return visible_sats

def compute_multipath(Rx_ENU, Rx_vel, dlength, dl, HEIGHT, visible_sats, eps, plot = False, prnt = False):
        
    ### Plotting
    if plot:
        sat_len = 200
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter3D(Rx_ENU[0,0], Rx_ENU[1,0], Rx_ENU[2,0], 'k', s=50)
        
        ### Plotting all the buildings in the surroundings
        dh = 2
        for i in range(len(HEIGHT)): 
            for j in range(len(HEIGHT[0])): 
                center = [(j+0.5)*dlength, (i+0.5)*dlength, dh/2.0]
                X, Y, Z, _ = toolutils.cuboid_data(center, (dl, dl, dh))
                ax.plot_surface(X, Y, Z, color='b', rstride=1, cstride=1, alpha=0.1) #''''''

    n_blds = 4
    x_blds = [round(Rx_ENU[0,0]/dlength)-0.5, \
              round(Rx_ENU[0,0]/dlength)-0.5, \
              round(Rx_ENU[0,0]/dlength)+0.5, \
              round(Rx_ENU[0,0]/dlength)+0.5]
    y_blds = [round(Rx_ENU[1,0]/dlength)-0.5, \
              round(Rx_ENU[1,0]/dlength)+0.5, \
              round(Rx_ENU[1,0]/dlength)-0.5, \
              round(Rx_ENU[1,0]/dlength)+0.5]
    #print(x_blds, y_blds, Rx_ENU)
    
    ### Store the potential reflection surfaces of the buildings of interest
    refl_surfs = {}
    for k in range(n_blds): 
        refl_surfs[str(k)]={}
        jj = int(x_blds[k]-0.5)
        ii = int(y_blds[k]-0.5)
        #print('jj: ', jj, 'ii: ', ii, 'h: ', HEIGHT) #[ii][jj]
        
        center = [(jj+0.5)*dlength, (ii+0.5)*dlength, HEIGHT[ii][jj]/2.0]
        X, Y, Z, surfs = toolutils.cuboid_data(center, (dl, dl, HEIGHT[ii][jj]))   
        store_wrx, store_normals, store_idx = toolutils.get_angle(Rx_vel, Rx_ENU, surfs)
        if (store_wrx[0]>=0.0):
            refl_surfs[str(k)]['pln_pts'] = surfs[str(store_idx[0])]
            refl_surfs[str(k)]['pln_normal'] = store_normals[0]
            refl_surfs[str(k)]['anti_pts'] = surfs[str(store_idx[1])]
            refl_surfs[str(k)]['anti_normal'] = store_normals[1]
            refl_surfs[str(k)]['h'] = 2*surfs['c'][2] ## height
        else: 
            refl_surfs[str(k)]['pln_pts'] = surfs[str(store_idx[1])]
            refl_surfs[str(k)]['pln_normal'] = store_normals[1]
            refl_surfs[str(k)]['anti_pts'] = surfs[str(store_idx[0])]
            refl_surfs[str(k)]['anti_normal'] = store_normals[0]        
            refl_surfs[str(k)]['h'] = 2*surfs['c'][2] ## height
        ### Plotting
        if plot:
            ax.plot_surface(X, Y, Z, color='r', rstride=1, cstride=1, alpha=0.1)
    
    store_ranges = {}
    pr_mp = {}
    d_el = np.deg2rad(0) ### to calculate distances
    desired_az = np.deg2rad(np.arange(0,360,36))     ### To calculate distances
    
    for i in range(len(visible_sats['prn'])):
        prn = visible_sats['prn'][i]
        store_ranges[str(prn)] = []
        rayPoint = np.array([Rx_ENU[0,0], Rx_ENU[1,0], Rx_ENU[2,0]])
               
        #### Correct positions of satellites
        rayDirection = np.array([visible_sats['satLoc'][i][0]-Rx_ENU[0,0], \
                                 visible_sats['satLoc'][i][1]-Rx_ENU[1,0], \
                                 visible_sats['satLoc'][i][2]-Rx_ENU[2,0]])

        ### Plotting
        if plot:
            x_sat = rayPoint[0] + (sat_len/visible_sats['pr_true'][i])*rayDirection[0] 
            y_sat = rayPoint[1] + (sat_len/visible_sats['pr_true'][i])*rayDirection[1]
            z_sat = rayPoint[2] + (sat_len/visible_sats['pr_true'][i])*rayDirection[2]
            ax.scatter3D(x_sat, y_sat, z_sat, s=50)
            ax.text(x_sat, y_sat, z_sat, "SV_"+str(prn), color='k')
        
        counter = {}
        counter['los'] = []
        counter['bck'] = []
        counter['nlos'] = []
        height_buildings = {}
        for k in range(n_blds):
            planeNormal = refl_surfs[str(k)]['pln_normal']
            planePoint = refl_surfs[str(k)]['pln_pts'][0]
            inst_pnt = toolutils.intersect_LinePlane(rayDirection, rayPoint, planeNormal, planePoint)
            
            ### distances-> 3d buildings
            for azz in desired_az:
                buildingDir = np.array([np.sin(azz)*np.cos(d_el), \
                                        np.cos(azz)*np.cos(d_el), \
                                        np.sin(d_el)])
                build_pnt = toolutils.intersect_LinePlane(buildingDir, rayPoint, planeNormal, planePoint)
                #print('az: ', azz, 'building_pnt: ', build_pnt)
                if (all(build_pnt==-1)): 
                    continue
                if not (toolutils.NotBlocked(refl_surfs[str(k)]['pln_pts'], planeNormal, build_pnt)): 
                    #print('yes in view of elevation and azimuth')
                    height_buildings[str(azz)] = refl_surfs[str(k)]['h']
                    #print('prn: ', prn, 'azimuth: ', azz, 'elevation: ', d_el, 'height: ', height_buildings[str(prn)])  
                    
            if (toolutils.NotBlocked(refl_surfs[str(k)]['pln_pts'], planeNormal, inst_pnt)): 
                counter['los'].append(True) ## Not, blocked
            else: 
                counter['los'].append(False) ## Yes, blocked
                ### Plotting
                if plot:
                    ax.plot([Rx_ENU[0,0],x_sat], [Rx_ENU[1,0],y_sat], [Rx_ENU[2,0],z_sat], 'y--')           
                
            ### surface on the backside
            anti_pnt = toolutils.intersect_LinePlane(rayDirection, rayPoint, \
                                                 refl_surfs[str(k)]['anti_normal'], \
                                                 refl_surfs[str(k)]['anti_pts'][0])

            if (toolutils.NotBlocked(refl_surfs[str(k)]['anti_pts'], refl_surfs[str(k)]['anti_normal'], anti_pnt)): 
                counter['bck'].append(True) # Not blocked by backside wall
            else: 
                counter['bck'].append(False)
                ### Plotting
                if plot:
                    ax.plot([Rx_ENU[0,0],x_sat], [Rx_ENU[1,0],y_sat], [Rx_ENU[2,0],z_sat], 'y--')           
                
            #### Mirror images position of satellites 
            invsatLoc = toolutils.mirror_image(refl_surfs[str(k)]['pln_pts'], visible_sats['satLoc'][i]) 
            invrayDirection = np.array([invsatLoc[0]-Rx_ENU[0,0], \
                                        invsatLoc[1]-Rx_ENU[1,0], \
                                        invsatLoc[2]-Rx_ENU[2,0]])

            mirr_inst = toolutils.intersect_LinePlane(invrayDirection, rayPoint, planeNormal, planePoint)
            
            if (toolutils.NotBlocked(refl_surfs[str(k)]['pln_pts'], planeNormal, mirr_inst)): 
                counter['nlos'].append(False)
            else: 
                counter['nlos'].append(True) ## should be blocked for NLOS to exist
                rho = np.linalg.norm(np.squeeze(Rx_ENU)-mirr_inst) + np.linalg.norm(mirr_inst-visible_sats['satLoc'][i])
                store_ranges[str(prn)].append(rho)
                ### Plotting
                if plot:
                    ax.plot([Rx_ENU[0,0], mirr_inst[0]], [Rx_ENU[1,0],mirr_inst[1]], [Rx_ENU[2,0],mirr_inst[2]], 'r')
                    ax.plot([mirr_inst[0], x_sat], [mirr_inst[1], y_sat], [mirr_inst[2], z_sat], 'r')  
                    
        if prnt:
            print('PRN: ', prn, 'BLD: ', k, 'Counter: ', counter)
                
        ### The LOS signal is not blocked any front surfaces then plot it
        if( all(counter['los']) and all(counter['bck']) ):
            store_ranges[str(prn)].append(np.linalg.norm(np.squeeze(Rx_ENU)-visible_sats['satLoc'][i]))
            if plot:
                ax.plot([Rx_ENU[0,0],x_sat], [Rx_ENU[1,0],y_sat], [Rx_ENU[2,0],z_sat], 'b')
                
        #### Just status observing   
        if (all(counter['los'])): ### all are true then LOS is not blocked
            if ( all(counter['bck']) ): 
                if ( any(counter['nlos']) ):
                    ### 4. Multipath signal, both NLOS and LOS
                    pr_mp[str(prn)] = np.mean(store_ranges[str(prn)]) + np.random.randint(200,900)/10.0
                    if prnt:
                        prnt('PRN: ', prn, 'Multipath: both NLOS and LOS ', \
                              store_ranges[str(prn)], visible_sats['pr_true'][i]-pr_mp[str(prn)]) 
                else: 
                    ### 3. Only LOS signal is present
                    pr_mp[str(prn)] = store_ranges[str(prn)][0] + np.random.randint(1,100)/10.0
                    if prnt:
                        prnt('PRN: ', prn, 'Only LOS: ', store_ranges[str(prn)], visible_sats['pr_true'][i]-pr_mp[str(prn)])
            else: 
                if ( any(counter['nlos']) ):
                    ### 2. Only NLOS signal is present
                    pr_mp[str(prn)] = store_ranges[str(prn)][0] + np.random.randint(300,1000)/10.0
                    if prnt:
                        prnt('PRN: ', prn, 'Only NLOS, BCK LOS is blocked: ', \
                              store_ranges[str(prn)], visible_sats['pr_true'][i]-pr_mp[str(prn)])
                else: 
                    ### 1. no signal-> satellite blocked completely
                    pr_mp[str(prn)] = visible_sats['pr_true'][i] + np.random.randint(700,1200)/10.0
                    if prnt:
                        prnt('PRN: ', prn, 'No signal, BCK LOS and NLOS blocked: ', \
                              store_ranges[str(prn)], visible_sats['pr_true'][i]-pr_mp[str(prn)])                    
        else:  #->atleast one false-> being blocked
            if ( any(counter['nlos']) ):
                ### 2. Only NLOS signal is present
                pr_mp[str(prn)] = store_ranges[str(prn)][0] + np.random.randint(300,1000)/10.0
                if prnt:
                    prnt('PRN: ', prn, 'Only NLOS, LOS is blocked: ', \
                          store_ranges[str(prn)], visible_sats['pr_true'][i]-pr_mp[str(prn)])
            else: 
                ### 1. no signal-> satellite blocked completely
                pr_mp[str(prn)] = visible_sats['pr_true'][i] + np.random.randint(700,1200)/10.0
                if prnt:
                    prnt('PRN: ', prn, 'No signal, LOS and NLOS blocked: ', \
                          store_ranges[str(prn)], visible_sats['pr_true'][i]-pr_mp[str(prn)])
                
        #pr_mp[str(prn)] = visible_sats['pr_true'][i] ### remove this to get multipath effects!!!
        
    ### Setting the plot limits
    if plot:
        ax.set_zlabel('UP (m)')
        ax.set_xlabel('EAST (m)')
        ax.set_ylabel('NORTH (m)')
        ax.view_init(elev=47, azim = -9)
        plt.savefig('yo'+str(eps)+'.png')
        plt.show()
    
    return pr_mp, height_buildings


#######################################################################

def compute_multipathPrintMMM(Rx_ENU, Rx_vel, dlength, dl, HEIGHT, visible_sats):
    
    ### Intialize the figures
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
        
    ### Plotting all the buildings in the surroundings
    '''for i in range(len(WORLD)): 
        for j in range(len(WORLD[0])): 
            if (WORLD[i][j]=='x'):   
                x0 = (j+0.5)*dlength
                y0 = (i+0.5)*dlength
                center = [x0, y0, dh/2.0]
                #print('center: ', center)
                X, Y, Z, _ = utils.cuboid_data(center, (dl, dl, dh))
                ax.plot_surface(X, Y, Z, color='b', rstride=1, cstride=1, alpha=0.1)
    ax.scatter3D(Rx_ENU[0,0], Rx_ENU[1,0], Rx_ENU[2,0], 'k', s=50) '''

    n_blds = 4
    #Rx_vel = [0, 1, 0]
    x_blds = [round(Rx_ENU[0,0]/dlength)-0.5, \
              round(Rx_ENU[0,0]/dlength)-0.5, \
              round(Rx_ENU[0,0]/dlength)+0.5, \
              round(Rx_ENU[0,0]/dlength)+0.5]
    y_blds = [round(Rx_ENU[1,0]/dlength)-0.5, \
              round(Rx_ENU[1,0]/dlength)+0.5, \
              round(Rx_ENU[1,0]/dlength)-0.5, \
              round(Rx_ENU[1,0]/dlength)+0.5]
    print(x_blds, y_blds)

    
    ### Store the potential reflection surfaces of the buildings of interest
    refl_surfs = {}
    for k in range(n_blds): 
        refl_surfs[str(k)]={}
        #n_blds==2!!: jj = int(x_blds[k]) #ii = int(y_blds[k]) #x0 = (jj+0.5)*dlength #y0 = (ii+0.5)*dlength
        jj = int(x_blds[k]-0.5)
        ii = int(y_blds[k]-0.5)
        x0 = (jj+0.5)*dlength
        y0 = (ii+0.5)*dlength
        center = [x0, y0, HEIGHT[ii][jj]/2.0]
        #print('center: ', center, 'height: ', HEIGHT[ii][jj])
        X, Y, Z, surfs = toolutils.cuboid_data(center, (dl, dl, HEIGHT[ii][jj]))                  
        store_wrx, store_normals, store_idx = toolutils.get_angle(Rx_vel, Rx_ENU, surfs)
        #print('store wrx: ', store_wrx)
        #print('store normals: ', store_normals)
        #print('store idx: ', store_idx)        
        if (store_wrx[0]>=0.0):
            refl_surfs[str(k)]['pln_pts'] = surfs[str(store_idx[0])]
            refl_surfs[str(k)]['pln_normal'] = store_normals[0]
            refl_surfs[str(k)]['anti_pts'] = surfs[str(store_idx[1])]
            refl_surfs[str(k)]['anti_normal'] = store_normals[1]
        else: 
            refl_surfs[str(k)]['pln_pts'] = surfs[str(store_idx[1])]
            refl_surfs[str(k)]['pln_normal'] = store_normals[1]
            refl_surfs[str(k)]['anti_pts'] = surfs[str(store_idx[0])]
            refl_surfs[str(k)]['anti_normal'] = store_normals[0]                        
        print(' ')        
        ax.plot_surface(X, Y, Z, color='r', rstride=1, cstride=1, alpha=0.1)
        
    #print('reflection surfaces: ', refl_surfs)
    print(' ')
    
    store_ranges = {}
    for i in range(len(visible_sats['prn'])):
        prn = visible_sats['prn'][i]
        store_ranges[str(prn)] = []
        rayPoint = np.array([Rx_ENU[0,0], Rx_ENU[1,0], Rx_ENU[2,0]])
               
        #### Correct positions of satellites
        rayDirection = np.array([visible_sats['satLoc'][i][0]-Rx_ENU[0,0], \
                                 visible_sats['satLoc'][i][1]-Rx_ENU[1,0], \
                                 visible_sats['satLoc'][i][2]-Rx_ENU[2,0]])
        
        x_sat = rayPoint[0] + (sat_len/visible_sats['pr_true'][i])*rayDirection[0] 
        y_sat = rayPoint[1] + (sat_len/visible_sats['pr_true'][i])*rayDirection[1]
        z_sat = rayPoint[2] + (sat_len/visible_sats['pr_true'][i])*rayDirection[2]
        ax.scatter3D(x_sat, y_sat, z_sat, s=50)
        ax.text(x_sat, y_sat, z_sat, "SV_"+str(prn), color='k')
        
        counter = {}
        counter['los'] = []
        counter['bck'] = []
        counter['nlos'] = []
        for k in range(n_blds):
            planeNormal = refl_surfs[str(k)]['pln_normal']
            planePoint = refl_surfs[str(k)]['pln_pts'][0]
            inst_pnt = toolutils.intersect_LinePlane(rayDirection, rayPoint, planeNormal, planePoint)
            print('LOS intersect pnt: ', inst_pnt)
        
            if (toolutils.check(refl_surfs[str(k)]['pln_pts'], planeNormal, inst_pnt)): 
                print("PRN: ", prn, 'No, Satellite not blocked by BLD: ', k) 
                counter['los'].append(1)
            else: 
                print("PRN: ", prn, 'Yes, Satellite is blocked by BLD: ', k) 
                counter['los'].append(0)
                
            ### surface on the backside
            anti_pnt = toolutils.intersect_LinePlane(rayDirection, rayPoint, \
                                                 refl_surfs[str(k)]['anti_normal'], \
                                                 refl_surfs[str(k)]['anti_pts'][0])
            print('Anti LOS intersect pnt: ', anti_pnt)

            if (toolutils.check(refl_surfs[str(k)]['anti_pts'], refl_surfs[str(k)]['anti_normal'], anti_pnt)): 
                print("PRN: ", prn, 'No, Satellite not blocked by back of BLD: ', k)
                counter['bck'].append(1)
            else: 
                print("PRN: ", prn, 'Yes, Satellite is blocked by back of BLD: ', k) 
                counter['bck'].append(0)
                ax.plot([Rx_ENU[0,0],x_sat], [Rx_ENU[1,0],y_sat], [Rx_ENU[2,0],z_sat], 'y--')           
                
            #### Mirror images position of satellites 
            invsatLoc = toolutils.mirror_image(refl_surfs[str(k)]['pln_pts'], visible_sats['satLoc'][i]) 
            invrayDirection = np.array([invsatLoc[0]-Rx_ENU[0,0], \
                                        invsatLoc[1]-Rx_ENU[1,0], \
                                        invsatLoc[2]-Rx_ENU[2,0]])

            mirr_inst = toolutils.intersect_LinePlane(invrayDirection, rayPoint, planeNormal, planePoint)
            print('NLOS intersection pnt: ', mirr_inst)
            
            if (toolutils.check(refl_surfs[str(k)]['pln_pts'], planeNormal, mirr_inst)): 
                print("PRN: ", prn, 'No, Virtual satellite not blocked by BLD: ', k) 
                counter['nlos'].append(1)
            else: 
                print("PRN: ", prn, 'Yes, Virtual satellite is blocked by BLD: ', k) 
                counter['nlos'].append(0)
                first_nlos = Rx_ENU[0,0]-mirr_inst[0]
                print('PRN: ', prn, 'Rho: ', \
                      np.linalg.norm(np.squeeze(Rx_ENU)-mirr_inst) + np.linalg.norm(mirr_inst-visible_sats['satLoc'][i]))
                
                rho = np.linalg.norm(np.squeeze(Rx_ENU)-mirr_inst) + \
                np.linalg.norm(mirr_inst-visible_sats['satLoc'][i])
                store_ranges[str(prn)].append(rho)
                ax.plot([Rx_ENU[0,0], mirr_inst[0]], [Rx_ENU[1,0],mirr_inst[1]], [Rx_ENU[2,0],mirr_inst[2]], 'r')
                ax.plot([mirr_inst[0], x_sat], [mirr_inst[1], y_sat], [mirr_inst[2], z_sat], 'r')  
                
                ''' ### Will exclude this for now! i will add it later if rest of the stuff work
                nlosDirection = -np.array([visible_sats['satLoc'][i][0]-mirr_inst[0], \
                                          visible_sats['satLoc'][i][1]-mirr_inst[1], \
                                          visible_sats['satLoc'][i][2]-mirr_inst[2]])
                print('NLOS reflection plane: ', k, (k+1)%n_blds)
                nlos_pnt = utils.intersect_LinePlane(nlosDirection, mirr_inst, \
                                                     refl_surfs[str((k+1)%n_blds)]['pln_normal'], \
                                                     refl_surfs[str((k+1)%n_blds)]['pln_pts'][0])
                print('nlos intersection point: ', nlos_pnt)
                if (utils.check(refl_surfs[str((k+1)%n_blds)]['pln_pts'], \
                                refl_surfs[str((k+1)%n_blds)]['pln_normal'], nlos_pnt)): 
                    print("PRN: ", prn, 'No, NLOS signal is not blocked by BLD: ', k) 
                    ax.plot([Rx_ENU[0,0], mirr_inst[0]], [Rx_ENU[1,0],mirr_inst[1]], [Rx_ENU[2,0],mirr_inst[2]], 'r')
                    ax.plot([mirr_inst[0], x_sat], [mirr_inst[1], y_sat], [mirr_inst[2], z_sat], 'r')  
                else: 
                    print("PRN: ", prn, 'Yes, NLOS signal is not blocked by BLD: ', k) 
                    ax.plot([Rx_ENU[0,0], mirr_inst[0]], [Rx_ENU[1,0],mirr_inst[1]], [Rx_ENU[2,0],mirr_inst[2]], 'y--')
                    ax.plot([mirr_inst[0], x_sat], [mirr_inst[1], y_sat], [mirr_inst[2], z_sat], 'y--')  '''
           
        print('PRN: ', prn, 'BLD: ', k, 'Counter: ', counter)
        print('')
                
        ### The LOS signal is not blocked any front surfaces then plot it
        if( all(i==1 for i in counter['los']) and all(i==1 for i in counter['bck']) ):
            print('PRN: ', prn, 'Rho: ', np.linalg.norm(np.squeeze(Rx_ENU)-visible_sats['satLoc'][i]) )
            store_ranges[str(prn)].append(np.linalg.norm(np.squeeze(Rx_ENU)-visible_sats['satLoc'][i]))
            ax.plot([Rx_ENU[0,0],x_sat], [Rx_ENU[1,0],y_sat], [Rx_ENU[2,0],z_sat], 'b')
        print(" ")
        
    ### Setting the plot limits
    #ax.set_xlim(-10, (len(WORLD)+1)*dlength)
    #ax.set_ylim(-10, (len(WORLD[0])+1)*dlength)
    #ax.set_xlim3d([min(np.transpose(store_center)[0])-3*dlength, max(np.transpose(store_center)[0])+ 3*dlength])
    #ax.set_ylim3d([min(np.transpose(store_center)[1])-3*dlength, max(np.transpose(store_center)[1])+ 3*dlength])
    ax.set_zlabel('Z-ENU (m)')
    #ax.set_zlim(-10, 100)
    #plt.savefig('time'+str(idx)+'.png')
    ax.set_xlabel('X-ENU (m)')
    ax.set_ylabel('Y-ENU (m)')
    #ax.view_init(elev=90, azim = -90)
    plt.show()
    
    return store_ranges

def compute_staticSVsMMM(data, t_gps, Ref_ECEF, no_sats = 32):
    
    #store_ECEF = []
    store_ENU = []
    for j in range(no_sats): 
        prn = int(j)
        toe_diff = np.transpose(data['Toe'].values)[prn] - t_gps
        toe_diff[np.isnan(toe_diff)]=1e10
        idx = np.argmin(abs(toe_diff))
    
        ephem = {}
        ephem['TOE'] = int(data['Toe'].values[idx][prn])
        ephem['M0'] = data['M0'].values[idx][prn]
        ephem['sqrta'] = data['sqrtA'].values[idx][prn]
        ephem['e'] = data['Eccentricity'].values[idx][prn]
        ephem['omega'] = data['omega'].values[idx][prn]
        ephem['Omega0'] = data['Omega0'].values[idx][prn]
        ephem['i0'] = data['Io'].values[idx][prn]
        ephem['Omega_dot'] = data['OmegaDot'].values[idx][prn]
    
        sat_ECEF = toolutils.FindSat(ephem, prn, t_gps)
        sat_ENU, _ = coordutils.ECEF_to_ENU( Ref_ECEF, sat_ECEF[0][2:5].reshape(-1,1))
        
        #store_ECEF.append([ sat_ECEF[0,0], sat_ECEF[1,0], sat_ECEF[2,0] ])
        store_ENU.append([ sat_ENU[0,0], sat_ENU[1,0], sat_ENU[2,0] ])

    return store_ENU #store_ECEF, 


