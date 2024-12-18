# ----PYTHON---- #
from typing import Union, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm 

from .base import base

class Universe(base):

    def evolve(self):

        """
        Main Function
        
        In this function, the positions of the particles are initialized and are then moved forward in time.
        For each step, the components of the displacement vector are randomly sampled from a gaussian distribution and are
        added to the actual position of the particles. Periodic boundary conditions, domains with different diffusion coefficients, and hard boundaries 
        are taken into account.

        There are two "universe" objects that are considered during the setup and the evolution of the system:

            - A "wrapped" universe (self.w_universe) in which positions are bounded by the limits of the box (self.size_x, and self.size_y).
            - An "unwrapped" universe (self.u_universe) in which positions are not bounded by the limits of the box. This can be used for further analysis (e.g., MSD)

        To prevent a slow down of the simulation due to an increase in the stored trajectory, two methods are applied:

            - Frames are stored only every self.nstxout steps in w_storage/u_storage
            - Arrays w_storage/u_storage are written to disk every self.nstchk steps.

        Therefore self.nstchk // self.nstxout frames are stored in a single checkpoint file.

        """

        #---------------------------------------------------------------------------------------------------------------------
        #Initialization Step

        #Populate the universe uniformly, but take hard boundaries into account
        self.w_universe = self.populate_universe_uniform()
        self.u_universe = np.copy( self.w_universe )       #Unwrapped, and wrapped universe are starting from the same positions
        
        #Number of frames in a checkpoint file
        nstchk_frames = self.nstchk // self.nstxout
        
        #Setup storage for positions -> Shape: (Number of frames in checkpoint file, Number of Particles, Number of dimensions)
        #The first checkpoint file has one frame more, because it contains also the init positions
        w_storage  = np.zeros( ( nstchk_frames + 1, self.N, self.dim ), dtype = np.float32 )
        u_storage  = np.zeros( ( nstchk_frames + 1, self.N, self.dim ), dtype = np.float32 )

        #Setup storage for time steps -> Shape: (Number of frames in checkpoint file)
        time_array = np.zeros(  nstchk_frames, dtype = np.float32 )
        
        #Store initial frame
        w_storage[0] = self.w_universe
        u_storage[0] = self.u_universe
        
        #Store initial time
        time_array[0] = 0

        #---------------------------------------------------------------------------------------------------------------------
        #Main Iteration

        #Iterate over the requested number of steps.
        #It starts at 1, because time step 0 is the initial distribution of the points in the universe.
        #It ends at self.nsteps + 1 to ensure that the requested time is reached
        for i in tqdm(range(1, self.nsteps + 1) ):
            
            #-----------------------------------------------------------------------------------------------------------------
            #Evolve particle position

            #Check diffusion coefficients
            self.effective_d_coeffs = self.generate_diffusion()

            #Move particles in time
            self.forward_in_time()
            
            #Apply boundary conditions
            self.apply_pbc()

            #Apply hard boundaries
            self.hard_boundaries()

            #Apply boundary conditions
            self.apply_pbc()
            
            #-----------------------------------------------------------------------------------------------------------------
            #Store particle positions

            #Unwrap self.w_universe and store the positions in self.u_universe
            for j, size in enumerate([self.size_x, self.size_y, self.size_z][:self.dim]):
                self.u_universe[:, j] = self.w_universe[:, j] - np.floor( (self.w_universe[:, j] - self.u_universe[:, j]) / size + 0.5 ) * size

            #Write current state of the universe into storage arrays
            if not i % self.nstxout: 

                #Calculate current time
                time = self.dt * i
                
                #--------------------------------------------------------------------------------------------------
                #Fill storage arrays

                #Current frame index in the current checkpoint file
                chk_frame_index = (i // self.nstxout) % nstchk_frames
                
                w_storage[ chk_frame_index, :, : ] = self.w_universe
                u_storage[ chk_frame_index, :, : ] = self.u_universe
                
                time_array[ frame_number ] = time

                #If check pointing is requested then write current storage arrays to disk and renew them
                if not i % self.nstchk:
                
                    #Number of current check point
                    chk_number = str(i // self.nstchk)

                    #Write arrays to disk
                    np.save( arr = w_storage,  file = self.output +   f"_wrap.{chk_number.zfill(5)}" )
                    np.save( arr = u_storage,  file = self.output + f"_unwrap.{chk_number.zfill(5)}" )
                    
                    np.save( arr = time_array, file = self.output +   f"_time.{chk_number.zfill(5)}" )
        
                    #Setup storage for positions -> Shape: (Number of frames in checkpoint file, Number of Particles, Number of dimensions)
                    w_storage  = np.zeros( ( nstchk_frames, self.N, self.dim ), dtype = np.float32 )
                    u_storage  = np.zeros( ( nstchk_frames, self.N, self.dim ), dtype = np.float32 )

                    #Setup storage for time steps -> Shape: (Number of frames in checkpoint file)
                    time_array = np.zeros(  nstchk_frames, dtype = np.float32 )


    def populate_universe_uniform(self):

        init_pos = np.random.rand( self.N, self.dim )
        
        for i, size in enumerate([self.size_x, self.size_y, self.size_z][:self.dim]):
            init_pos[:, i] *= size
        
        #Resample points if they are in hard boundaries
        if not any(self.hard_boundaries_geometry): pass

        else:
        
            print('Found applied hard boundaries!')
            print('Start resampling...')


            check = True
            
            while check:
            
                in_bound_idx = np.array([], dtype = np.int64)

                for key, geometry in self.hard_boundaries_geometry.items():

                    if 'c' in key:

                        in_bound_idx_key = self.check_circ_cond(pos = init_pos,
                                                                mid = geometry[0].reshape(1, 2),
                                                                r = geometry[1] )
                    
                    elif 'p' in key:

                        in_bound_idx_key = self.check_square_cond(pos = init_pos,
                                                                  Lx = geometry[4],
                                                                  Ly = geometry[5],
                                                                  mid = geometry[6].reshape(1, 2))  
                    else: raise ValueError(f'Key {key} not known!') 

                    in_bound_idx = np.append( in_bound_idx, in_bound_idx_key )

                init_pos[ in_bound_idx ] = np.random.rand( in_bound_idx.shape[0], self.dim )           

                for i, size in enumerate([self.size_x, self.size_y, self.size_z][:self.dim]):
                    
                    init_pos[ in_bound_idx , i] *= size
                
                if not in_bound_idx.size > 0: check = False

            print('Resampling finished!')

        assert init_pos[:, 0].max() <= self.size_x 
        assert init_pos[:, 1].max() <= self.size_y
        if self.dim == 3: assert init_pos[:, 2].max() <= self.size_z 
        
        assert init_pos[:, 0].min() >= 0
        assert init_pos[:, 1].min() >= 0
        if self.dim == 3: assert init_pos[:, 2].min() >= 0

        return init_pos

    def forward_in_time(self):

        factor = np.sqrt( 2 * self.effective_d_coeffs * self.dt ).reshape(-1, 1)

        self.displace = factor * np.random.randn( self.N, self.dim )

        #Move particles
        self.w_universe += self.displace

    def apply_pbc(self):

        #Apply periodic boundary conditions
        for i, size in enumerate([self.size_x, self.size_y, self.size_z][:self.dim]):
            self.w_universe[:, i] %= size

    def generate_diffusion(self):

        if type( self.d_coeffs ) == float: return np.repeat(self.d_coeffs, self.N)
        
        elif self.d_coeffs.ndim == 2:

            grid_indices = np.int64( np.floor( self.w_universe  / self.grid ) )

            return self.d_coeffs[ grid_indices[:, 0], grid_indices[:, 1] ]
        
        else:
            raise ValueError("Currently I cannot handle the provided diffusion coefficients. Either enter float or 2-dimensional numpy array")

    def apply_pbc_vector(self, vec):

        assert vec.ndim == 2, 'Vector has not a dimension of 2'
        
        #Apply PBC
        vec[:, 0] = np.where(vec[:, 0] >    self.size_x / 2, vec[:, 0] - self.size_x, vec[:, 0])
        vec[:, 0] = np.where(vec[:, 0] <= - self.size_x / 2, vec[:, 0] + self.size_x, vec[:, 0])

        vec[:, 1] = np.where(vec[:, 1] >    self.size_y / 2, vec[:, 1] - self.size_y, vec[:, 1])
        vec[:, 1] = np.where(vec[:, 1] <= - self.size_y / 2, vec[:, 1] + self.size_y, vec[:, 1])

        return vec
    
    def check_circ_cond(self, pos, mid, r):

        assert mid.ndim == 2, 'Middle point is not two-dimensional'

        circ_coor = pos - mid
        circ_coor = self.apply_pbc_vector( circ_coor )

        circ_dist = np.linalg.norm(circ_coor, axis = 1)

        return np.where(circ_dist <= r)[0]
        

    def check_square_cond(self, pos, Lx, Ly, mid): 
        
        assert mid.ndim == 2, 'Middle point is not two-dimensional'

        squa_coor = (pos - mid)
        squa_coor = self.apply_pbc_vector( squa_coor )

        cond_x = np.logical_and( (-Lx/2 <= squa_coor[:, 0]), (squa_coor[:, 0] <= Lx/2) )
        cond_y = np.logical_and( (-Ly/2 <= squa_coor[:, 1]), (squa_coor[:, 1] <= Ly/2) )

        return np.where( np.logical_and(cond_x, cond_y))[0]

    @staticmethod
    def calc_reflection(d, n):

        """
        Calculate reflection vector of vector d with normal n

        d := numpy.ndarray
            displacement vectors of particles that are reflected
        n := numpy.ndarray
            normal of geometry
        """

        #Vectorized form
        #d.shape = (Nr, 2)
        #n.shape = (n , 2)

        return d - 2 * np.sum(d * n, axis = 1).reshape(-1, 1) * n

    def get_normals_circle(self, prev_points, points, d, mid, r):

        """
        Calculate the normal vector of a two dimensional circle for a point

        points := numpy.ndarray
            coordinates of reflected points
        prev_points := numpy.ndarray
            previous coordinates of reflected points
        d := numpy.ndarray
            displace vector
        r := float
            radius of the circle
        
        """

        prev_points = prev_points.reshape(-1,2)
        points      = points.reshape(-1,2)

        #-------------------------------------
        M = (prev_points - mid)
        M = self.apply_pbc_vector(M)
        #-------------------------------------
        A = d[:, 0]**2 + d[:, 1]**2
        B = M[:, 0] * d[:, 0] + M[:, 1] * d[:, 1]
        C = M[:, 0]**2 + M[:, 1]**2
        #-------------------------------------

        lam1 = 2 * B + 2 * np.sqrt(B**2 - A * ( C - r**2 ) )
        lam1 /= 2 * A
        
        lam2 = 2 * B - 2 * np.sqrt(B**2 - A * ( C - r**2 ) )
        lam2 /= 2 * A

        lam = np.abs( np.vstack((lam1, lam2)) ).min(0)
        lam = lam.reshape(-1, 1)
        
        assert lam.shape[0] == points.shape[0], 'Lambda has the wrong shape'

        #-------------------------------------
        intersection = prev_points + lam * d
        intersection[:, 0] %= self.size_x
        intersection[:, 1] %= self.size_y
        
        #Check if intersection is between old positions and new positions

        p_inter = intersection - points
        p_inter = self.apply_pbc_vector(p_inter)
        
        prev_inter = intersection - prev_points
        prev_inter = self.apply_pbc_vector(prev_inter)

        d = self.apply_pbc_vector(d)

        p_inter2    = np.sum(   (p_inter)**2, axis = 1)
        prev_inter2 = np.sum((prev_inter)**2, axis = 1)
        prev_d2      = np.sum(d**2, axis = 1)

        check = p_inter2 + prev_inter2 + 2 * np.sqrt(p_inter2 * prev_inter2)

        if not np.all( np.abs(check - prev_d2) < 1E-8 ):
            print(lam1, lam2)
            print(lam)
            print('A',A)
            print('B',B)
            print('C',C)
            print('r', r)
            print('mid', mid)

            print("Something went wrong. Write debug files!")
            np.save(arr = prev_points, file = 'prev_positions.debug.npy')
            np.save(arr = points,      file = 'positions.debug.npy')
            np.save(arr = d,           file = 'displace.debug.npy')
            np.save(arr = d,           file = 'displace.debug.npy')

            raise ValueError('Intersection is not between positions!')

        #Check if intersection is on circle boundary

        dist2mid = intersection - mid
        dist2mid = self.apply_pbc_vector( dist2mid )

        dist2mid = np.linalg.norm(dist2mid, axis = 1)

        if not np.all( np.abs(dist2mid - r) < 1E-8 ):

            print("Something went wrong. Write debug files!")
            np.save(arr = prev_points, file = 'prev_positions.debug.npy')
            np.save(arr = points,      file = 'positions.debug.npy')
            np.save(arr = d,           file = 'displace.debug.npy')

            raise ValueError('Intersection is not on circle!')
        
        #-------------------------------------

        norm = intersection - mid
        #Apply PBC
        norm = self.apply_pbc_vector(norm)

        norm /= np.linalg.norm( norm, axis = 1).reshape(-1, 1)

        return norm, intersection

    @staticmethod
    def perpDot(a, b):
        
        a_vert = np.copy(a)
        a_vert = np.flip(a_vert, axis = 1)
        a_vert[:, 0] = -1 * a_vert[:, 0]

        return np.dot(a_vert, b)

    def get_point_between_points(self, p, d, e1, e2, Lx, Ly):

        c = e1 - p
        c = self.apply_pbc_vector(c)
        
        vert = (e2 - e1)
        vert = self.apply_pbc_vector(vert.reshape(1, -1) )[0]

        t = self.perpDot(c , vert) / self.perpDot(d, vert)

        intersection = p + t.reshape(-1, 1) * d

        intersection = self.apply_pbc_vector(intersection)

        #----------------------------------------------------------
        true_intersection = np.ones( p.shape[0] , dtype = bool)
        #Check if point is on vector

        e1_inter = intersection - e1
        e1_inter = self.apply_pbc_vector(e1_inter)
        
        e2_inter = intersection - e2
        e2_inter = self.apply_pbc_vector(e2_inter)

        e1_inter2 = np.sum((e1_inter)**2, axis = 1)
        e2_inter2 = np.sum((e2_inter)**2, axis = 1)
        e1_e22    = np.sum(vert**2)

        check = e1_inter2 + e2_inter2 + 2 * np.sqrt(e1_inter2 * e2_inter2)

        cond = (check - e1_e22) >= 1E-8
        
        true_intersection[ cond ] = False
        
        #---------------------------------------------------------
        #Check if point is on vector

        p_inter = intersection - p
        p_inter = self.apply_pbc_vector(p_inter)

        prev_pos = p-d
        prev_pos[:, 0] %= self.size_x
        prev_pos[:, 1] %= self.size_y
        
        prev_inter = intersection - prev_pos
        prev_inter = self.apply_pbc_vector(prev_inter)

        p_inter2    = np.sum(   (p_inter)**2, axis = 1)
        prev_inter2 = np.sum((prev_inter)**2, axis = 1)
        prev_p2      = np.sum(d**2, axis = 1)

        check = p_inter2 + prev_inter2 + 2 * np.sqrt(p_inter2 * prev_inter2)

        cond = np.abs(check - prev_p2) >= 1E-8
        
        true_intersection[ cond ] = False
        #print(true_intersection)
        #---------------------------------------------------------
        """
        p_to_intersection = intersection - p
        p_to_intersection = self.apply_pbc_vector(p_to_intersection)
        p_to_intersection = np.abs(p_to_intersection)
 
        true_intersection[ p_to_intersection[:, 0] > (Lx / 2) ] = False
        true_intersection[ p_to_intersection[:, 1] > (Ly / 2) ] = False

        true_intersection[ intersection[:, 0] < 0 ] = False
        true_intersection[ intersection[:, 0] > self.size_x ] = False
        
        true_intersection[ intersection[:, 1] < 0 ] = False
        true_intersection[ intersection[:, 1] > self.size_y ] = False
        """
        intersection[~true_intersection] = np.array([1E-8, 1E-8])

        return intersection, true_intersection, vert

    def get_normals_polyon(self, points, displace, edges, Lx, Ly):

        """
        Get the normal vector of a two dimensional two dimension polygon for a point

        points := numpy.ndarray
            coordinates of reflected points
        edges := list
            list of polygon edges
        
        """

        edges = edges + [edges[0]]
        
        number_of_vertices = len(edges) - 1

        final_norm = np.zeros( (points.shape[0], 2), dtype = np.float32)
        final_intersection = np.zeros( (points.shape[0], 2), dtype = np.float32)

        for i in range( len(edges) - 2 + 1 ):

            polygon_point_a = edges[i]
            polygon_point_b = edges[i+1]
    
            intersection, true_intersection, a_to_b = self.get_point_between_points(p = points,
                                                                                    d = displace,
                                                                                   e1 = polygon_point_a,
                                                                                   e2 = polygon_point_b,
                                                                                   Lx = Lx,
                                                                                   Ly = Ly)

            if   a_to_b.sum() > 0.0: norm = polygon_point_a - intersection

            elif a_to_b.sum() < 0.0: norm = polygon_point_b - intersection

            else: raise ValueError("Fuck")
            
            norm = self.apply_pbc_vector(norm)
            norm = np.flip(norm, axis = 1)
            norm[:, 0] *= -1

            norm /= np.linalg.norm(norm, axis = 1).reshape(-1, 1)

            norm[~true_intersection] = np.array([0.0, 0.0])

            final_norm += norm
            final_intersection += intersection

        #assert np.all(np.logical_or( np.abs(len_final - 1.) < 1E-6, np.abs(len_final) < 1E-6)) , "Normals are not normalized!"

        len_norm = np.linalg.norm(final_norm, axis = 1)


        #print((points - displace)[len_norm > 1.])
        #print(points[             len_norm > 1.])
        #print(displace[             len_norm > 1.])
        #print(len_norm[         len_norm > 1.])
        #print(final_norm[         len_norm > 1.])
        #print(final_intersection[ len_norm > 1.])
        

        if not np.all( np.abs( len_norm - 1.) < 1E-8):
            np.save(arr = points, file = 'debug_points.npy')
            np.save(arr = displace, file = 'debug_displace.npy')

            raise ValueError("Normals are not normalized! Saved actual states as debug_points.npy and debug_displace.npy!")

        return final_norm, final_intersection

    
    def hard_boundaries(self):
        
        if not any(self.hard_boundaries_geometry): pass

        else:

            for key, geometry in self.hard_boundaries_geometry.items():

                #Circular hard boundary
                if 'c' in key:
                    
                    #Calculate real distances
                    #--------------------------------------------------------------------------------
                    mid = self.hard_boundaries_geometry[key][0].reshape(1,2)
                    r   = self.hard_boundaries_geometry[key][1]
                    #--------------------------------------------------------------------------------

                    index = self.check_circ_cond(pos = self.w_universe,
                                                 mid = mid,
                                                 r   = r)
                    
                    if not index.size > 0: continue

                    p_prev = self.w_universe[index] - self.displace[index]

                    p_prev[:, 0] %= self.size_x
                    p_prev[:, 1] %= self.size_y
                    
                    test_index = self.check_circ_cond(pos = p_prev,
                                                      mid = mid,
                                                      r   = r)

                    assert not test_index.size > 0, 'Points already in circle'

                    #--------------------------------------------------------------------------------
        
                
                    norm, intersection = self.get_normals_circle(points      = self.w_universe[index],
                                                                 prev_points = p_prev,
                                                                 d           = self.displace[index],
                                                                 mid         = mid,
                                                                 r           = r
                                                                 )
                    d_new = intersection - p_prev
                    d_new = self.apply_pbc_vector(d_new)
                    
                    reflected = self.calc_reflection(d = d_new,
                                                     n = norm 
                                                    )
                    
                    new_pos = intersection + reflected
                    new_pos[:, 0] %= self.size_x
                    new_pos[:, 1] %= self.size_y


                    #----------------------------------------------------------------------------------
                    #Testing
                    real_dist_new = (new_pos - mid)
                    real_dist_new = self.apply_pbc_vector(real_dist_new)
                    real_dist_new = np.linalg.norm(real_dist_new, axis = 1)
                    
                    assert np.all(real_dist_new > r), 'Not all new positions are outside the circle'
                    
                    #TODO Make it applicable for PBC

                    pos_p_prev = p_prev - intersection
                    pos_p_prev = self.apply_pbc_vector(pos_p_prev)
                    pos_ref    = self.apply_pbc_vector(reflected)

                    dot_pp = np.sum(pos_p_prev  * norm, axis = 1)
                    dot_pr = np.sum(pos_ref * norm, axis = 1)

                    dot_pp /= np.linalg.norm(pos_p_prev, axis = 1)
                    dot_pr /= np.linalg.norm(pos_ref   , axis = 1)

                    assert np.allclose(dot_pr, dot_pp), "In angle is not equal out angle"
                    assert np.allclose(dot_pp, dot_pr), "In angle is not equal out angle"
                    #----------------------------------------------------------------------------------

                    self.w_universe[index] = new_pos

                #Square hard boundary
                if 'p' in key:
                    
                    #Calculate real distances
                    #--------------------------------------------------------------------------------
                    edges = self.hard_boundaries_geometry[key][0:4]
                    Lx    = self.hard_boundaries_geometry[key][4]
                    Ly    = self.hard_boundaries_geometry[key][5] 
                    mid   = self.hard_boundaries_geometry[key][6].reshape(1, 2)
                    #--------------------------------------------------------------------------------
                    
                    index = self.check_square_cond(pos = self.w_universe,
                                                   Lx = Lx,
                                                   Ly = Ly,
                                                   mid = mid)


                    if not index.size > 0: continue

                    p_prev = self.w_universe[index] - self.displace[index]

                    p_prev[:, 0] %= self.size_x
                    p_prev[:, 1] %= self.size_y

                    test_index = self.check_square_cond(pos = p_prev,
                                                         Lx = Lx,
                                                         Ly = Ly,
                                                        mid = mid)
                    
                    assert not test_index.size > 0, 'Points already in square'
                    #--------------------------------------------------------------------------------
    
                    norm, intersection = self.get_normals_polyon(points = self.w_universe[ index ],
                                                               displace = self.displace[ index ],
                                                                 edges  = edges,
                                                                     Lx = Lx,
                                                                     Ly = Ly
                                                                 )
                
                    d_new = intersection - p_prev
                    d_new = self.apply_pbc_vector(d_new)

                    reflected = self.calc_reflection(d = d_new,
                                                     n = norm 
                                                     )
                    
                    new_pos = intersection + reflected
                    new_pos[:, 0] %= self.size_x
                    new_pos[:, 1] %= self.size_y
                    #----------------------------------------------------------------------------------
                    #Testing

                    pos_p_prev = p_prev - intersection
                    pos_p_prev = self.apply_pbc_vector(pos_p_prev)
                    
                    pos_ref    = self.apply_pbc_vector(reflected)

                    dot_pp = np.sum(pos_p_prev  * norm, axis = 1)
                    dot_pr = np.sum(pos_ref * norm, axis = 1)

                    dot_pp /= np.linalg.norm(pos_p_prev, axis = 1)
                    dot_pr /= np.linalg.norm(pos_ref   , axis = 1)

                    assert np.allclose(dot_pr, dot_pp), "In angle is not equal out angle"
                    assert np.allclose(dot_pp, dot_pr), "In angle is not equal out angle"
                    #----------------------------------------------------------------------------------
                    
                    self.w_universe[index] = new_pos
                    

    #-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    #Analysis part

    def load_data_unwrap(self):

        try:
            if self.u_storage.shape[0] == self.nsteps // self.nstxout: return 0
        except: pass
        
        self.u_storage  = np.zeros( (0, self.N, self.dim ), dtype = np.float32 )
        self.time_array = np.zeros( (0)                   , dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            self.u_storage  = np.vstack( (self.u_storage, np.load(self.output + f"_unwrap.{chk_number.zfill(5)}.npy") ))

            self.time_array = np.append( self.time_array, np.load(self.output +   f"_time.{chk_number.zfill(5)}.npy")  ) 

        #Validation
        assert self.u_storage.shape[0] == (self.nsteps // self.nstxout) + 1, "Number of frames is not correct!"

        assert np.allclose( np.diff( self.time_array ), self.dt * self.nstxout ), "Time step distance is not as expected!"
        assert np.allclose( self.dt * self.nstxout, np.diff( self.time_array ) ), "Time step distance is not as expected!"

        return 1
    
    def load_data_wrap(self):

        try:
            if self.w_storage.shape[0] == self.nsteps // self.nstxout: return 0
        except: pass
        
        self.w_storage  = np.zeros( (0, self.N, self.dim ), dtype = np.float32 )
        self.time_array = np.zeros( (0)                   , dtype = np.float32 )

        for chk_number in range(1, self.nsteps // self.nstchk + 1 ):

            chk_number = str(chk_number)

            try: 

                self.w_storage  = np.vstack( (self.w_storage, np.load(self.output + f"_wrap.{chk_number.zfill(5)}.npy") ))
                self.time_array = np.append( self.time_array, np.load(self.output +   f"_time.{chk_number.zfill(5)}.npy")  ) 

            except FileNotFoundError as e:
                pass

        #Validation
        #assert self.w_storage.shape[0] == (self.nsteps // self.nstxout) + 1, "Number of frames is not correct!"

        assert np.allclose( np.diff( self.time_array ), self.dt * self.nstxout ), "Time step distance is not as expected!"
        assert np.allclose( self.dt * self.nstxout, np.diff( self.time_array ) ), "Time step distance is not as expected!"

        return 1


    def mean_square_displacement(self, skip):

        print(f"Calculate MSD from Frame 1/tau {1*self.dt * self.nstxout}ns to Frame {self.nsteps // self.nstxout}/tau {(self.nsteps // self.nstxout) * self.dt * self.nstxout} with skip Frame {skip/(self.dt* self.nstxout)}/tau {skip}ns")

        #Load data
        self.load_data_unwrap()

        skip /= (self.dt * self.nstxout)

        lagtimes = np.arange(1, self.nsteps // self.nstxout, int(skip))

        msd = np.zeros( lagtimes.shape[0] , dtype = np.float32)
        sd_per_particle = np.zeros( (lagtimes.shape[0], self.N) , dtype = np.float32)

        for i, lag in tqdm(enumerate(lagtimes), total = lagtimes.shape[0]):

            dr = self.u_storage[:-lag, :, :] - self.u_storage[lag:, :, :]
            sqdist = np.square(dr).sum(axis=-1)

            msd[i] = sqdist.mean()
            sd_per_particle[i] = sqdist.mean(axis = 0)

        tau = np.float32(lagtimes)
        tau *= self.dt * self.nstxout

        return tau, msd, sd_per_particle
    
    def mean_square_displacement_distr(self, tau):
       
        #Load data
        self.load_data_unwrap()

        lag = int(np.round(tau / (self.dt * self.nstxout) ))

        dr = self.u_storage[:-lag, :, :] - self.u_storage[lag:, :, :]
        sqdist = np.square(dr).sum(axis=-1)

        sd_per_particle = sqdist.mean(axis = 0)

        print(f'MSD Distribution at {lag * self.dt * self.nstxout / 1000 / 1000} ms') 

        return sd_per_particle
    
    @staticmethod
    def mean_square_displacement_fit(tau, msd):
        
        #tau -> ns
        #msd -> nm2
        DiffCoeff, Intercept = np.polyfit(x = tau, y = msd, deg = 1)

        #DiffCoeff -> nm2/ns -> um2/ms
        #Intercept -> nm2

        return DiffCoeff, Intercept

    @staticmethod
    def plot_log_histogram_mean_square_displacement_distr(sd_per_particle, label, lo_limit = 1E-4, up_limit = 1.0, nbins = 51, color = 'red'):

        """
        Plot histogram with log-space bins

        Parameters
        ----------

        sd_per_particle := numpy.ndarray
            Squared-Displacement per particle at a single time-lag tau (expected unit: square-micrometer)
        label           := str
            Label for legend
        lo_limit        := float
            Lower limit for log-spaced bins (opt.)
        up_limit        := float
            Upper limit for log-spaced bins (opt.)
        nbins           := int
            Number of log-spaced bins (opt.)
        color           := str
            Matplotlib color
            
        """

        a = plt.hist(sd_per_particle, 
                     density=True, 
                     bins = np.logspace(np.log10(lo_limit),np.log10(up_limit), nbins),
                     histtype = 'step',
                     color = color,
                     label = label
                    )

        #Scale
        plt.xscale('log')

        #Label
        plt.ylabel('Number of Trajectories')
        plt.xlabel(r'MSD / $\mu$m$^2$')

        


