#include "armadillosolver.h"

ArmadilloSolver::ArmadilloSolver(
        float *p_flow_bases_u,
        float *p_flow_bases_v,
        float *p_cov_matrix,
        int p_pc_width,
        int p_pc_height,
        int p_nc,
        int p_nc_max_per,
        double p_sigma,
        double p_lambda,
        int p_debug,
        int n_iters)
{
    // Copy parameters
    this->pc_width = p_pc_width;
    this->pc_height = p_pc_height;
    this->nc = p_nc;
    this->sigma_sq = p_sigma * p_sigma;
    this->lambda = p_lambda;
    this->debug = p_debug;
    this->n_iters = n_iters;
    
    // Copy pointers to raw data of the flow bases.
    //this->flow_bases_u = flow_bases_u;
    //this->flow_bases_v = flow_bases_v;
    
    // These matrices are of size (width*height) x n_bases.
    this->flow_bases_u_t = fmat(p_flow_bases_u, p_pc_width*p_pc_height, p_nc/2);
    this->flow_bases_v_t = fmat(p_flow_bases_v, p_pc_width*p_pc_height, p_nc/2);

    // If we want less cols, remove them here.
    this->flow_bases_u_t = this->flow_bases_u_t.head_cols(p_nc/2);
    this->flow_bases_v_t = this->flow_bases_v_t.head_cols(p_nc/2);
   
    // Generate covariance matrix
    fmat cov(p_cov_matrix, 2*p_nc_max_per,2*p_nc_max_per);
    // Armadillo stores the matrices as column-major. But cov is symmetric, so this is not a problem.

    if (p_nc_max_per == p_nc/2)
    {
      // We use the full covariance matrix
      // Generate Q as inverse of covariance matrix with some slight regularization.
      mat Q_ = this->lambda * (conv_to<mat>::from(cov) + eye<mat>(p_nc,p_nc)).i();
      this->Q = conv_to<fmat>::from(Q_);
    }
    else {
      if (0) { // Old-style inversion
      	mat Q_ = this->lambda * (conv_to<mat>::from(cov) + eye<mat>(2*p_nc_max_per,2*p_nc_max_per)).i();
        uvec inds(p_nc);

        inds.subvec(0,p_nc/2-1) = linspace<uvec>(0,p_nc/2-1,p_nc/2);
        inds.subvec(p_nc/2,p_nc-1) = linspace<uvec>(0,p_nc/2-1,p_nc/2)+p_nc_max_per;

	    this->Q = conv_to<fmat>::from(Q_(inds,inds));
      }
      else { // New style
	    uvec inds = join_cols(linspace<uvec>(0,p_nc/2-1,p_nc/2),
			      linspace<uvec>(p_nc_max_per, p_nc_max_per+p_nc/2-1,p_nc/2));
			    
        fmat cov2 = cov(inds,inds);
      	mat Q_ = this->lambda * (conv_to<mat>::from(cov2) + eye<mat>(p_nc,p_nc)).i();
        this->Q = conv_to<fmat>::from(Q_);
      }
	
    }

      
      // fvec indices_ = kp0tr.col(1) * this->pc_width + kp0tr.col(0);
      // uvec indices = conv_to<uvec>::from(indices_);

      // if (this->debug)
      //     std::cout << "[DEBUG] \t Filling A ..." << std::endl;

      // fmat Au = this->flow_bases_u_t.rows(indices);
      // fmat Av = this->flow_bases_v_t.rows(indices);

      
    // this->Q = this->lambda * (cov + eye<fmat>(p_nc,p_nc)).i();

    if (this->debug)
    {
        std::cout << "[DEBUG] Initialized ArmadilloSolver." << std::endl;
        std::cout << "[DEBUG] Using " << this->nc << " basis vectors." << std::endl;
    }
}

ArmadilloSolver::~ArmadilloSolver()
{
    // We did not really initialize anything, so we just set our pointers to 0.

    if (this->debug)
        std::cout << "[DEBUG] Destroyed ArmadilloSolver." << std::endl;
}

void ArmadilloSolver::solve(
        float *kp0,
        float *kp1,
        int n_kp
        //float *initial_weights,
        )
{

    this->n_kp = n_kp;

    if (this->debug)
    {
        std::cout << "[DEBUG] Starting ArmadilloSolver, using " << this->nc << " basis vectors." << std::endl;
        std::cout << "[DEBUG] Creating the system. " << std::endl;
    }
    // Create the system
    this->createSystem(kp0,kp1,n_kp);

    if (this->debug)
    {
        std::cout << "[DEBUG] Finished creating the system." << std::endl;
        std::cout << "[DEBUG] Starting solving process." << std::endl;
    }

    this->solve_irls();

    if (this->debug)
    {
        std::cout << "[DEBUG] Finished solving process." << std::endl;
        std::cout << "[DEBUG] Preparing result." << std::endl;
    }

}

void ArmadilloSolver::get_flow(float *p_flow_u, float *p_flow_v)
{
    if (this->debug)
    {
        std::cout << "[DEBUG] Computing result: Flow" << std::endl;
    }

    // Treat the output as vectors.
    fvec wrap_u(p_flow_u, this->pc_width*this->pc_height,false,true);
    fvec wrap_v(p_flow_v, this->pc_width*this->pc_height,false,true);

    if (this->debug)
    {
        std::cout << "[DEBUG] Size of x: " << this->x.n_elem << std::endl;
    }

    wrap_u = this->flow_bases_u_t * this->x.subvec(0,this->nc/2 - 1);
    wrap_v = this->flow_bases_v_t * this->x.subvec(this->nc/2,this->nc - 1);

    if (this->debug)
    {
        std::cout << "[DEBUG] Example output: " << wrap_u(100 * this->pc_width + 100) << std::endl;
        std::cout << "[DEBUG] Same output in float: " << p_flow_u[100*this->pc_width + 100] << std::endl;
    }
}

void ArmadilloSolver::get_coefficients(float *p_coefficients)
{

    if (this->debug)
    {
        std::cout << "[DEBUG] Computing result: Coefficients" << std::endl;
    }

    fvec wrap_c(p_coefficients, this->nc, false, true);
    wrap_c = this->x;
}

void ArmadilloSolver::get_weights(float *p_weights)
{
    if (this->debug)
    {
        std::cout << "[DEBUG] Computing result: Weights" << std::endl;
    }

    fvec wrap_w(p_weights,this->n_kp, false, true);
    wrap_w = this->wsq.subvec(0,this->n_kp-1);
}

void ArmadilloSolver::createSystem(float *kp0, float *kp1, int n_kp /*,int max_bases */)
{
    /*
     * Generate y
     *
     */

    if (this->debug)
        std::cout << "[DEBUG] \t Creating data and vectorizing..." << std::endl;


    // Note that these are transposed, since armadillo uses column-major ordering.
    fmat kp0_(kp0,2,n_kp, false, true);
    fmat kp1_(kp1,2,n_kp, false, true);
    fmat kp0t = kp0_.t();

    fmat uv = (kp1_.t()-kp0t);

    this->y = vectorise(uv); // column-wise

    if (this->debug)
        std::cout << "[DEBUG] \t Creating A ..." << std::endl;

    /*
     * Generate A
     *
     */

    int n_bases_per = this->nc/2;
    //if (max_bases > 0)
    //    n_bases = min(max_bases,this->nc);

    this->A = zeros<fmat>(2*n_kp, 2*n_bases_per);

    fmat kp0tr = floor(kp0t);
    fvec indices_ = kp0tr.col(1) * this->pc_width + kp0tr.col(0);
    uvec indices = conv_to<uvec>::from(indices_);

    if (this->debug)
        std::cout << "[DEBUG] \t Filling A ..." << std::endl;

    fmat Au = this->flow_bases_u_t.rows(indices);
    fmat Av = this->flow_bases_v_t.rows(indices);
    this->A.submat(0,0,n_kp-1,n_bases_per-1) = Au;
    this->A.submat(n_kp,n_bases_per,2*n_kp-1,2*n_bases_per-1) = Av;

    if (this->debug)
        std::cout << "[DEBUG] \t Done." << std::endl;
}


void ArmadilloSolver::solve_irls()
{
    // Weight vector
    fvec wsq = ones<fmat>(y.n_elem);
    //fvec wsq_ = ones<fmat>(y.n_elem);

    //fvec err_u = ones<fmat>(y.n_elem/2);
    //fvec err_v = ones<fmat>(y.n_elem/2);

    fmat A_(A.n_rows, A.n_cols);
    //fvec y_(y.n_elem);
    fvec x_(A.n_cols);

    fmat T1(A.n_cols, A.n_cols);
    fvec T2(A.n_cols);

    if (this->debug)
    {
        std::cout << "[DEBUG] Solver dimensions: " << std::endl;
        std::cout << "[DEBUG] \t T1: " << T1.n_rows << " x " << T1.n_cols << std::endl;
        std::cout << "[DEBUG] \t T2: " << T2.n_rows << " x " << T2.n_cols << std::endl;
        std::cout << "[DEBUG] \t A: " << A.n_rows << " x " << A.n_cols << std::endl;
        std::cout << "[DEBUG] \t y: " << y.n_elem << std::endl;
        std::cout << "[DEBUG] \t x: " << x_.n_elem << std::endl;
        std::cout << "[DEBUG] \t Q: " << Q.n_rows << " x " << Q.n_cols << std::endl;
    }

    int iters = this->n_iters;

    for (int iter=0; iter < iters; iter++) {
        A_ = A;

        // Re-weight rows of matrix and result vector
        A_.each_col() %= wsq;
        //y_ = y % wsq;

        T1 = A_.t() * A_ + Q;
        T2 = A_.t() * (y % wsq);
        x_ = arma::solve(T1,T2);

        // Computing wsq is a little different, since we want to 
        // weight both x and y direction of the pixel by the L2 error.
        wsq = square(A * x_ - y) * 1.0/this->sigma_sq;
        wsq = repmat(wsq.subvec(0,n_kp-1) + wsq.subvec(n_kp,2*n_kp-1),2,1);

        // Cauchy error
        // Note that the error would be 1.0/(1+f**2), but since we do not take the square
        // root above, we can omit the squaring here.
        wsq.transform( [](float f) {return 1.0/(1.0+f); } );
        wsq = sqrt(wsq);
    }

    this->x = x_;
    this->wsq = square(wsq);
}




