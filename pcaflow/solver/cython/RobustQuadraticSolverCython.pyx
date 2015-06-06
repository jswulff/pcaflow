# cython: profile=False

import numpy as np

cimport cython
cimport numpy as np

cdef extern from "armadillosolver.h":
    cdef cppclass ArmadilloSolver:
        ArmadilloSolver(float*, float*, float*, int, int, int, int, double, double, int, int) except +
        void solve(float*, float*, int)
        void get_flow(float*, float*)
        void get_coefficients(float*)
        void get_weights(float*)


cdef class RobustQuadraticSolverCython:
    """
    RobustQuadraticSolver class, cython implementation.

    """
    cdef ArmadilloSolver *thisptr

    cdef int pc_width
    cdef int pc_height
    cdef int nc
    cdef int debug
    cdef int n_iters
    cdef double sigma
    cdef double lambd
    cdef float[:,:] cm
    cdef int nc_max_per

    #cdef float[:,:] ar_return_flow_u
    #cdef float[:,:] ar_return_flow_v
    #cdef float[:] ar_return_coefficients

    def __cinit__(self,
            float[:,:] flow_bases_u,
            float [:,:] flow_bases_v,
            cov_matrix,
            pc_size,
            params,
            n_iters=3):
        """
        Initialize robust solver.

        """

        self.pc_width = pc_size[0]
        self.pc_height = pc_size[1]
        self.nc = params['NC']
        self.debug = params['debug']
        self.sigma = params['sigma']
        self.lambd = params['lambda']

        self.n_iters=n_iters

        self.nc_max_per = flow_bases_u.shape[0]

        nc_per = self.nc / 2
        cov_w = cov_matrix.shape[0]
        cov_w_per = cov_w / 2

        # if nc_per != cov_w_per:
        #     cov_inds = np.r_[np.arange(nc_per),np.arange(nc_per)+cov_w_per]
        #     cov_matrix_ = cov_matrix[cov_inds,:][:,cov_inds]
        #     self.cm = cov_matrix_
        # else:
        #     self.cm = cov_matrix

        self.cm = cov_matrix

        #self.cm = cov_matrix[:,:self.nc]

        print('NC: ')
        print(self.nc)
        print('NC_MAX_PER: ')
        print(self.nc_max_per)

        self.thisptr = new ArmadilloSolver(
                &flow_bases_u[0,0],
                &flow_bases_v[0,0],
                #&cov_matrix[0,0],
                &self.cm[0,0],
                self.pc_width,
                self.pc_height,
                self.nc,
                self.nc_max_per,
                self.sigma,
                self.lambd,
                self.debug,
                self.n_iters)

#        self.ar_return_flow_u = np.zeros((self.pc_height,self.pc_width),dtype='float32')
#        self.ar_return_flow_v = np.zeros((self.pc_height,self.pc_width),dtype='float32')
#        self.ar_return_coefficients = np.zeros(self.nc,dtype='float32')
#
    def __dealloc__(self):
        del self.thisptr


    def solve(self,
            float[:,:] kp0,
            float[:,:] kp1,
            #kp0,
            #kp1,
            initial_weights=None,
            return_flow=True,
            return_coefficients=False,
            return_weights=False):

        if initial_weights is not None:
            print('[ ** ERROR **] Initial weights not yet implemented.')

        n_kp = kp0.shape[0]

        # This has to be defined dynamically
        ar_return_weights = np.zeros(n_kp,dtype='float32')


        ar_return_flow_u = np.zeros((self.pc_height,self.pc_width),dtype='float32')
        ar_return_flow_v = np.zeros((self.pc_height,self.pc_width),dtype='float32')
        ar_return_coefficients = np.zeros(self.nc,dtype='float32')
        return_coefficients_full = np.zeros(2*self.nc_max_per,dtype='float32')


        cdef float[:,:] _ar_return_flow_u = ar_return_flow_u
        cdef float[:,:] _ar_return_flow_v = ar_return_flow_v
        cdef float[:] _ar_return_coefficients = ar_return_coefficients
        cdef float[:] _ar_return_weights = ar_return_weights

        # Call the C++ code
        self.thisptr.solve(&kp0[0,0],
                &kp1[0,0],
                n_kp)
        #self.solveit(kp0,kp1,n_kp)

        # Return what we want to return
        ret = []
#        if return_flow:
#            self.thisptr.get_flow(&(self.ar_return_flow_u[0,0]),
#                    &(self.ar_return_flow_v[0,0]))
#            ret.append(self.ar_return_flow_u)
#            ret.append(self.ar_return_flow_v)
#
#        if return_coefficients:
#            self.thisptr.get_coefficients(&(self.ar_return_coefficients[0]))
#            ret.append(self.ar_return_coefficients)
#        if return_weights:
#            self.thisptr.get_weights(&_ar_return_weights[0])
#            ret.append(self.ar_return_weights)

        if return_flow:
            self.thisptr.get_flow(&_ar_return_flow_u[0,0],
                    &_ar_return_flow_v[0,0])
            ret.append(ar_return_flow_u)
            ret.append(ar_return_flow_v)

        if return_coefficients:
            self.thisptr.get_coefficients(&_ar_return_coefficients[0])
            return_coefficients_full[:self.nc/2] = ar_return_coefficients[:self.nc/2]
            return_coefficients_full[self.nc_max_per:self.nc_max_per+self.nc/2] = ar_return_coefficients[self.nc/2:]
            #ret.append(ar_return_coefficients)
            ret.append(return_coefficients_full)
        if return_weights:
            self.thisptr.get_weights(&_ar_return_weights[0])
            ret.append(ar_return_weights)


        return ret


#    cdef solveit(self,float[:,:] kp0, float[:,:] kp1, n_kp):
#        self.thisptr.solve(&kp0[0,0],
#                &kp1[0,0],
#                n_kp)
#




