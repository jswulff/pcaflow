import numpy as np

from libcpp.vector cimport vector
#from libc.math import max

cimport cython
cimport numpy as np

cdef extern from "matcher.h":
    cdef cppclass Matcher:
        cppclass parameters:
            parameters()
            int nms_n
            int nms_tau
            int nms_n                  # non-max-suppression: min. distance between maxima (in pixels)
            int nms_tau                # non-max-suppression: interest point peakiness threshold
            int match_binsize          # matching bin width/height (affects efficiency only)
            int match_radius           # matching radius (du/dv in pixels)
            int match_disp_tolerance   # dv tolerance for stereo matches (in pixels)
            int outlier_disp_tolerance # outlier removal: disparity tolerance (in pixels)
            int outlier_flow_tolerance # outlier removal: flow tolerance (in pixels)
            int multi_stage            # 0=disabled,1=multistage matching (denser and faster)
            int half_resolution        # 0=disabled,1=match at half resolution, refine at full resolution
            int refinement             # refinement (0=none,1=pixel,2=subpixel)
            double f                       # calibration (only for match prediction)
            double cu
            double cv
            double base

        cppclass p_match:
            float u1p
            float v1p
            float u1c
            float v1c
            

        Matcher(Matcher.parameters)
        #~Matcher()

        void pushBack(unsigned char*, int*, int)
        void matchFeatures(int)

        vector[Matcher.p_match] getMatches()
        
            
cdef print_doc():
    print('Blahb blah blubb')

cdef class LibvisoMatcher:
    """
    LibvisoMatcher class.

    
    """
    cdef Matcher *thisptr
    cdef Matcher.parameters param_ptr
    cdef int nms_n

    cdef int prune
    cdef int use_buckets
    cdef int buckets_x
    cdef int buckets_y
    cdef int bucket_size
    cdef int w
    cdef int wprev
    cdef int h
    cdef int hprev
    #cdef int [:] bucket_counts_temp
    #cdef bucket_counts_temp_
    cdef int [:] bucket_counts
    #cdef bucket_counts_
    cdef int[:] bucket_stepsize
    cdef int[:] bucket_curstep

    def __cinit__(self,
                  use_prune=False,
                  use_buckets=False,
                  buckets_x=4,
                  buckets_y=4,
                  bucket_size=100,
                  nms_n=5,
                  nms_tau=25,
                  match_binsize=50,
                  match_radius=200,
                  match_disp_tolerance=1,
                  outlier_disp_tolerance=5,
                  outlier_flow_tolerance=5,
                  multi_stage=1,
                  half_resolution=0,
                  refinement=1):
        """
        This is some docstring.
        
        """
        print('Initializing matcher...')
        
        self.param_ptr = Matcher.parameters()

        self.param_ptr.nms_n                  = nms_n;   # non-max-suppression: min. distance between maxima (in pixels)
        self.param_ptr.nms_tau                = nms_tau; # non-max-suppression: interest point peakiness threshold
        self.param_ptr.match_binsize          = match_binsize;  # matching bin width/height (affects efficiency only)
        self.param_ptr.match_radius           = match_radius; # matching radius (du/dv in pixels)
        self.param_ptr.match_disp_tolerance   = match_disp_tolerance;   # du tolerance for stereo matches (in pixels)
        self.param_ptr.outlier_disp_tolerance = outlier_disp_tolerance;   # outlier removal: disparity tolerance (in pixels)
        self.param_ptr.outlier_flow_tolerance = outlier_flow_tolerance;   # outlier removal: flow tolerance (in pixels)
        self.param_ptr.multi_stage            = multi_stage;   # 0=disabled,1=multistage matching (denser and faster)
        self.param_ptr.half_resolution        = half_resolution;   # 0=disabled,1=match at half resolution, refine at full resolution
        self.param_ptr.refinement             = refinement;   # refinement (0=none,1=pixel,2=subpixel)

        self.thisptr = new Matcher(self.param_ptr)

        self.nms_n = self.param_ptr.nms_n

        self.prune = use_prune;
        self.use_buckets = use_buckets;
        self.buckets_x = buckets_x;
        self.buckets_y = buckets_y;
        self.bucket_size = bucket_size;

        self.bucket_counts = np.zeros(buckets_x*buckets_y,dtype='int32')
        #self.bucket_counts = self.bucket_counts_
        self.bucket_stepsize = np.zeros(buckets_x*buckets_y,dtype='int32')
        self.bucket_curstep = np.zeros(buckets_x*buckets_y,dtype='int32')
        #self.bucket_counts_temp = self.bucket_counts_temp_

        self.wprev = 0
        self.hprev = 0



    def __dealloc__(self):
        del self.thisptr
        #free(self.bucket_counts)
        #del self.param_ptr

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def pushBack(self, unsigned char[:,:] image):
        cdef int w = image.shape[1]
        cdef int h = image.shape[0]

        if self.hprev != h or self.wprev != w:
            print('WARNING: Image changed size. ({},{} => {},{}).'.format(self.hprev,self.wprev,h,w))
            print('Re-allocating matcher...')
            del self.thisptr
            self.thisptr = new Matcher(self.param_ptr)
            self.wprev = w
            self.hprev = h
            
        self.w = w
        self.h = h
        dims_ = np.array([w,h,w],dtype='int32')
        cdef int[:] dims = dims_
        self.thisptr.pushBack(&image[0,0], &dims[0], 0)

        
    def getMatches(self):
        if self.prune:
            return self._getMatches_pruned();
        elif self.use_buckets:
            return self._getMatches_bucketed();
        else:
            return self._getMatches_plain()

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def _getMatches_bucketed(self):
        self.thisptr.matchFeatures(0)
        cdef vector[Matcher.p_match] matches = self.thisptr.getMatches()
        cdef int n_matches = matches.size()
        out_1 = np.zeros((n_matches,2))
        out_2 = np.zeros((n_matches,2))

        cdef int i
        cdef int x0,y0
        cdef int bucket
        cdef int n_buckets = self.buckets_x * self.buckets_y

        cdef int bucket_width = self.w / self.buckets_x
        cdef int bucket_height = self.h / self.buckets_y

        for i in range(n_buckets):
            self.bucket_counts[i] = 0
            self.bucket_stepsize[i] = 0
            self.bucket_curstep[i] = 0

        cdef int j = 0

        # Count how many features fall into each bucket
        for i in range(n_matches):
            x0 = <int>matches[i].u1p
            y0 = <int>matches[i].v1p
            bucket = (y0 / bucket_height) * self.buckets_x + (x0 / bucket_width)
            if bucket < n_buckets:
                self.bucket_counts[bucket] += 1

        # For each bucket, compute the step size
        for i in range(n_buckets):
            self.bucket_stepsize[i] = max(1,self.bucket_counts[i] / self.bucket_size)

        for i in range(n_matches):
            x0 = <int>matches[i].u1p
            y0 = <int>matches[i].v1p
            bucket = (y0 / bucket_height) * self.buckets_x + (x0 / bucket_width)

            if bucket >= n_buckets:
                continue

            if self.bucket_curstep[bucket] % self.bucket_stepsize[bucket] == 0:
                out_1[j,0] = x0
                out_1[j,1] = y0
                out_2[j,0] = matches[i].u1c
                out_2[j,1] = matches[i].v1c
                j += 1

            self.bucket_curstep[bucket] += 1

        out_1 = out_1[:j+1,:]
        out_2 = out_2[:j+1,:]

        return out_1, out_2


    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def _getMatches_plain(self):
        self.thisptr.matchFeatures(0)
        cdef vector[Matcher.p_match] matches = self.thisptr.getMatches()
        n_matches = matches.size()
        out_1 = np.zeros((n_matches,2))
        out_2 = np.zeros((n_matches,2))

        for i in range(n_matches):
            out_1[i,0] = matches[i].u1p
            out_1[i,1] = matches[i].v1p
            out_2[i,0] = matches[i].u1c
            out_2[i,1] = matches[i].v1c

        return out_1, out_2

    #@cython.boundscheck(False)
    #@cython.wraparound(False)
    def _getMatches_pruned(self):
        self.thisptr.matchFeatures(0)
        cdef vector[Matcher.p_match] matches = self.thisptr.getMatches()
        n_matches = matches.size()
        out_1 = np.zeros((n_matches,2))
        out_2 = np.zeros((n_matches,2))

        for i in range(n_matches):
            out_1[i,0] = matches[i].u1p
            out_1[i,1] = matches[i].v1p
            out_2[i,0] = matches[i].u1c
            out_2[i,1] = matches[i].v1c

        out_valid = np.ones(n_matches,dtype='bool')

        i=0
        N = 2*self.nms_n

        diff1 = np.abs(np.diff(out_1,axis=0).sum(axis=1))
        diff2 = np.abs(np.diff(out_2,axis=0).sum(axis=1))
        
        while i < n_matches-1:
            if N < diff1[i] or N < diff2[i]:
                i = i + 1
            else:
                out_valid[i+1] = False
                i = i + 2

        out_1 = out_1[out_valid,:]
        out_2 = out_2[out_valid,:]

        return out_1, out_2

    # def getMatches_pruned(self):
    #     self.thisptr.matchFeatures(0)
    #     cdef vector[Matcher.p_match] matches = self.thisptr.getMatches()
    #     n_matches = matches.size()
    #     out_1 = np.zeros((n_matches,2))
    #     out_2 = np.zeros((n_matches,2))

    #     i=0
    #     j=0
        
    #     x1p = 0
    #     y1p = 0
    #     x2p = 0
    #     y2p = 0

    #     while i < n_matches:
    #     #for i in range(n_matches):
    #         x1 = matches[i].u1p
    #         y1 = matches[i].v1p
    #         x2 = matches[i].u1c
    #         y2 = matches[i].v1c

    #         N = 2*self.nms_n
            
    #         if i==0 or N > (abs(x1p-x1) + abs(y1p-y1)) or N > (abs(x2p-x2) + abs(y2p-y2)):
    #             out_1[j,0] = x1
    #             out_1[j,1] = y1
    #             out_2[j,0] = x2
    #             out_2[j,1] = y2
    #             j += 1
    #             i += 1
    #         else:
    #             i += 2
    #         x1p = x1
    #         y1p = y1
    #         x2p = x2
    #         y2p = y2

    #     out_1 = out_1[:j,:]
    #     out_2 = out_2[:j,:]
            

    #     return out_1,out_2
        
     # def process(self, image, int seeds_w, int seeds_h, int nr_levels):
     #     img = (image[:,:,0].astype('uint32') << 16) + (image[:,:,1].astype('uint32') << 8) + (image[:,:,2].astype('uint32'))
     #     self.initialize(img,seeds_w,seeds_h,nr_levels)
     #     self.iterate()
     #     return self.labels

     # property labels:
     #     def __get__(self):
     #         #pass
             
     #         # Return memory view on output
     #         R = np.zeros((self.height, self.width),dtype='uint32')
     #         # print(R.shape)
                          
     #         cdef unsigned int[:,:] out_p = R
     #         cdef unsigned int[:,:] out_c = <unsigned int[:self.height,:self.width]> self.thisptr.labels[self.nlevels-1]
             
     #         out_p[:] = out_c[:]
     #         return R
             
     #     def __set__(self, value):
     #         print('Error. Cannot set.')
         
