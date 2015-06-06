#! /usr/bin/env python2
import sys

default_vals = {
    'sigma': 0.95,
    'lambda': 0.2,
    'NC': 500,     # Number of principal components
    
    'remove_homography': 2,
    
    # Gaussian blur kernel size. 0 to deactivate.
    'image_blur': 0,

    # How many models to use.
    #   1 : Use only a single model (PCA-Flow)
    # > 1 : Multiple model (PCA-Layers)
    'n_models': 6,

    ############################## 
    # Feature section
    ############################## 
    
    # Which features to use. Options are FAST, LIBVISO.
    'features': 'libviso',

    # Relative amount of pruning from border
    'features_prune_border': 0.0, 

    # Use CLAHE preprocessing?
    'features_clahe': 1,
 
    # Limit the features to be extracted (FAST only).
    # Options are:
    # [0,1]: Relative fraction of found features
    #   > 1: Total number of features
    #    -1: Use all features better than the mean
    #    -2: Use all features better than the median
    #
    'features_fast_limit': -1,

    # Use multiscale (LIBVISO only)
    'features_libviso_multiscale': 1,
    'features_libviso_n_scales': 2, # If set less or equal to 0, determine by image size.

    # How many feature to maximally put into one bucket.
    # (LIBVISO only).
    # If -1, bucketing is disabled.
    'features_libviso_bucket_size': 50,

    # How many buckets in x and y direction (LIBVISO only)
    'features_libviso_buckets_x': 8,
    'features_libviso_buckets_y': 4,

    'features_libviso_nms_n': 5, # non-max-suppression: min. distance between maxima (in pixels)
    'features_libviso_nms_tau': 25, # non-max-suppression: interest point peakiness threshold
    'features_libviso_match_binsize': 50, # matching bin width/height (affects efficiency only)
    'features_libviso_match_radius': 200, # matching radius (du/dv in pixels)
    'features_libviso_match_disp_tolerance': 1,  # du tolerance for stereo matches (in pixels)
    'features_libviso_outlier_disp_tolerance': 5,  # outlier removal: disparity tolerance (in pixels)
    'features_libviso_outlier_flow_tolerance': 7,  # outlier removal: flow tolerance (in pixels)
    'features_libviso_multi_stage': 1,  # 0=disabled,1=multistage matching (denser and faster)
    'features_libviso_half_resolution': 0,  # 0=disabled,1=match at half resolution, refine at full resolution
    'features_libviso_refinement': 1,  # refinement (0=none,1=pixel,2=subpixel)


    ##############################
    # Model section
    ##############################
    
    # Weight of the warping error
    # In the paper, this is implicitly set to 1.
    'model_gamma_warp': 1.0,

    # Weight of the color error
    'model_gamma_c': 5.0,

    # Weight of the location error
    'model_gamma_l': 6.5,

    # Pairwise weight
    'model_gamma': 350.0,

    # How many mixtures to use for the color model
    'model_color_n_mixtures': 1,

    # Sigma when rectifying warping costs
    'model_sigma_w': 300.0,

    # How much to weight the member centroid in the EM step
    'model_factor_dist_to_median': 0.008,

    # Location weight of initialization
    'em_init_loc_weight': 0.4,

    # Remove outliers using the full solution.
    # Can be one of:
    #     -1 : just multiply by the weights
    #      0 : disable
    #      
    'em_additional_remove_outliers': 0,


    # Parameters for sublayers
    'SUBLAYER_lambda': 0.02,
    'SUBLAYER_sigma': 0.95,
    'SUBLAYER_NC': 200,


    ###############################
    # MISC SECTION
    ###############################
    'debug': 0,
    }


preset_parameters = {
        'pcalayers_sintel': {
            'image_blur': 0,
            'features_libviso_outlier_flow_tolerance': 5,
            'lambda': 0.2,
            'sigma': 0.95, #0.1, 
            'em_init_loc_weight': 0.4,
            'model_factor_dist_to_median' : 0.008,
            'SUBLAYER_lambda': 0.02, 
            'SUBLAYER_sigma': 0.95,
            'SUBLAYER_NC': 200,
            'n_models': 6,
            'model_gamma': 350.0, #450.0,
            'model_gamma_c': 5.0, #3.0
            'model_gamma_l': 6.5,
            'model_sigma_w': 300.0,
            'remove_homography': 2,
            },


        'pcaflow_sintel': {
            'n_models': 1,
            'image_blur': 5,
            'features_libviso_outlier_flow_tolerance': 6,
            'lambda': 0.2,
            'sigma': 1.35, 
            'remove_homography': 0,
            },

        'pcaflow_kitti': {
            'n_models': 1,
            'image_blur': 5,
            'features_libviso_outlier_flow_tolerance': 7,
            'lambda': 0.4,
            'sigma': 0.55,
            'features_libviso_multiscale': 0,
            'features_prune_border': 0.05,
            'remove_homography': 0,
            },


        'pcalayers_kitti': {
            'image_blur': 5, #3,
            'features_libviso_outlier_flow_tolerance': 9, #8,
            'features_prune_border': 0.05,
            'features_libviso_multiscale': 1,
            'features_libviso_n_scales': 2,
            'n_models': 8, #6,
            'em_init_loc_weight': 0.0, #0.2,
            'model_factor_dist_to_median': 0.0,
            'lambda': 0.8, #0.4,
            'sigma': 0.55, #0.4,
            'SUBLAYER_lambda': 0.4, #0.1,
            'SUBLAYER_sigma': 0.7, #0.5,
            'SUBLAYER_NC': 120,
            'model_gamma_c': 0.5, # 2.0,
            'model_gamma_l': 8.0,
            'model_gamma': 75.0,
            'model_sigma_w': 10.0, #1.5,
            },

        }


 


def get_parameters(params=None,preset=None,do_print=True):
    # Read default values
    p = dict(default_vals)

    if preset is not None:
        assert(preset_parameters.has_key(preset))
        print('*** LOADING PRESET {} ***'.format(preset))
        params_preset = preset_parameters[preset]
        for k,v in params_preset.items():
            assert(p.has_key(k))
            p[k] = v

    # Read values from command line
    # For these, convert parameters to int.
    params_int = ['NC', 'image_blur', 'SUBLAYER_NC', 'n_models']
    for k,v in p.items():
        prm = '-'+k
        if not prm in sys.argv:
            # Value was not found
            continue
        else:
            # Read in value
            v_ = sys.argv[sys.argv.index(prm)+1]
            if k == 'features' or k == 'adapt_size_mismatch':
                pass
            elif k in params_int:
                v_ = int(float(v_))
            else:
                v_ = float(v_)
            p[k] = v_
    
    if params is not None:
        # Read values from given parameters -- those override all previous.
        params_ = dict(params)
        for k,v in params_.items():
            print('Replacing entry {}. Old: {}, new: {}'.format(k,p[k],v))
            p[k] = v

    if do_print:
        print('')
        print('[PARAMETERS MAIN]')
        print('')
        for k2 in sorted(p.keys()):
            v2 = p[k2]
            print('\t{0}: \t{1}'.format(k2,v2))

    return p



def get_sublayer_parameters(params=None):
    p = dict(get_parameters(params),do_print=False)

    for k,v in p.items():
        if k.startswith('SUBLAYER_'):
            k2 = k.strip('SUBLAYER_')
            print('k : {} :: k2 : {}'.format(k,k2))
            assert(p.has_key(k2))
            p[k2] = v

    print('')
    print('[PARAMETERS SUBLAYER]:')
    print('')
    for k in sorted(p.keys()):
        v2 = p[k]
        print('\t{0}: \t{1}'.format(k,v2))


    return p

