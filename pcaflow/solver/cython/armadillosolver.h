#include <armadillo>
#include <iostream>

#include <vector>

using namespace arma;

class ArmadilloSolver
{
    public:
        ArmadilloSolver(
                float *flow_bases_u,
                float *flow_bases_v,
                float *cov_matrix,
                int pc_width,
                int pc_height,
                int nc,         // Starting parameter block
		int nc_max_per,
                double sigma,
                double lambda,
                int debug=0,
                int n_iters=3);
        ~ArmadilloSolver();

        void solve(
                float *kp0,
                float *kp1,
                int n_kp           // Number of keypoints
                //float *initial_weights=0,
                );

        void get_flow(float *p_flow_u, float *p_flow_v);
        void get_coefficients(float *p_coefficients);
        void get_weights(float *p_weights);

    private:
        void createSystem(float *kp0, float *kp1, int n_kp);
        void solve_irls();

        fmat A;
        fvec y;
        fvec x;
        fvec wsq;

        fmat T1;
        fmat T2;

        fmat Q;

        int n_kp;

        // Parameters
        //float *flow_bases_u;
        //float *flow_bases_v;
        fmat flow_bases_u_t;
        fmat flow_bases_v_t;

        int pc_width;
        int pc_height;
        int nc;
        double sigma_sq;
        double lambda;
        int debug;

        int n_iters;

};




