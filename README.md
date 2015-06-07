PCA-Flow: Fast, approximate optical flow computation
====================================================

This software package contains two algorithms for the computation of optical flow, as described in Wulff & Black, "Efficient Sparse-to-Dense Optical Flow Estimation using a Learned Basis and Layers" (CVPR 2015).

*PCA-Flow* computes approximate optical flow extremely fast, by making the assumption that optical flow lies on a low-dimensional subspace.

*PCA-Layers* extends this to a layered model to increase accuracy, especially at boundaries.

We hope this software is useful to you.
If you have any questions, comments, or issues, please do not hesitate to [contact us](mailto:jonas.wulff@tuebingen.mpg.de).


Installation
------------

### Requirements

* [OpenCV](http://www.opencv.org) >= 3.0-rc1 with Python bindings
* [OpenBlas](http://www.openblas.net)
* Scientific Python stack with NumPy, SciPy, Scikit-Learn, Cython, and Matplotlib.

The easiest way to get the required scientific Python stack is to install a distribution such as
[Anaconda](https://store.continuum.io/cshop/anaconda/).


#### A note about libviso2

In order to obtain the best results, we recommend using the libviso2 features as described in

    @INPROCEEDINGS{Geiger2011IV,
      author = {Andreas Geiger and Julius Ziegler and Christoph Stiller},
      title = {StereoScan: Dense 3D Reconstruction in Real-time},
      booktitle = {Intelligent Vehicles Symposium (IV)},
      year = {2011}
    } 

These features can be downloaded from http://cvlibs.net/software/libviso.
Simply place the file libviso2.zip in the root directory (where the readme.md is).
The build.sh script will then build the necessary extension.

If you can or do not want to use libviso2, PCA-flow will fall back onto A-AKAZE features (see http://www.robesafe.com/personal/pablo.alcantarilla/kaze.html for details), which are included in OpenCV.


### Installation

If all the requirements are satisfied, run `build.sh` to check dependencies, and build the necessary local libraries.

After that, you can test PCA-Flow with

    python demo.py

The file `demo.py` contains further details on how to use PCA-Flow and PCA-Layers.


Examples
--------

### demo.py
`demo.py` can be run without any parameters, and shows how to use PCA-Flow in a
Python program.

### compute_flow.py
`compute_flow.py [PARAMETER] IMAGE1 IMAGE2 OUTFILE` can be used to compute the flow between two arbitrary frames IMAGE1 and IMAGE2 and write the output flow to OUTFILE.
`PARAMETER` can be a combination of the following settings:

* `-kitti`/`-sintel`: Use optimized parameters for specific dataset.
* `-pcaflow`/`-pcalayers`: Compute flow using PCA-Flow or PCA-Layers.
* If no parameters are given, the settings `-sintel -pcalayers` are used.

Example: `python compute_flow.py -kitti -pcaflow image1.png image2.png output.flo`.

### compute_flow_directory.py
`compute_flow_directory.py [PARAMETER] INDIR OUTDIR` can be used to compute the flow for all files in directory INDIR and write the output flow files to OUTDIR.
`PARAMETER` can be a combination of the following settings:

* `-kitti`/`-sintel`: Use optimized parameters for specific dataset.
* `-pcaflow`/`-pcalayers`: Compute flow using PCA-Flow or PCA-Layers.
* If no parameters are given, the settings `-sintel -pcalayers` are used.

Example: `python compute_flow_directory.py -kitti -pcaflow ~/sintel/training/final/alley_1 ~/sintel_flow/alley_1/`


Citation
--------

If you use PCA-Flow, please cite the following paper: 

    @inproceedings{Wulff:CVPR:2015,
      title = {Efficient Sparse-to-Dense Optical Flow Estimation using a Learned Basis and Layers},
      author = {Wulff, Jonas and Black, Michael J.},
      booktitle = { IEEE Conf. on Computer Vision and Pattern Recognition (CVPR) 2015},
      month = jun,
      year = {2015}
    }


License
-------

See `LICENSE.md` for licensing issues and details.


Contact
-------

If you run into any issues with PCA-Flow, please do not hesitate to contact us at
`jonas.wulff@tuebingen.mpg.de`.


