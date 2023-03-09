#include "spectral.hpp"
#include <thread>
#include <vector>

int main()
{
    double Re = 2000;
    double nu = 1.0/Re;
    double Lx = 2*M_PI;
    double Ly = 2*M_PI;
    double Nx = 256;
    double Ny = Nx;
    double dt = Lx/Nx/4.0;
    double tmax = 50.0;
    int thread_num = std::thread::hardware_concurrency();
    std::cout << "thread: " << thread_num << std::endl;
    int num_iter = static_cast<int>(tmax/dt);
    int span_observe = static_cast<int>(tmax/100/dt);


    //######################
    //  lamb-oseen vortex
    //######################
    double vortex_size = 0.2;
    std::vector<double> gammas = {1, 1};
    std::vector<double> loc_x = {M_PI - 0.5, M_PI + 0.5}; 
    std::vector<double> loc_y = {M_PI,M_PI};
    std::vector<double> radii = {vortex_size, vortex_size};
    auto lamb_oseen = [&](double x, double y)
    {
        double omega = 0.0;

        for(int i = 0; i < 2; ++i)
        {
            double x0 = loc_x[i];
            double y0 = loc_y[i];
            double radius2 = (x - x0)*(x - x0) + (y - y0)*(y - y0);
            double vortex_size2 = radii[i]*radii[i];

            omega += gammas[i]/M_PI/vortex_size2 * std::exp(-radius2/vortex_size2);
        }
        
        return omega;
    };

    Calculater calc_manager(num_iter, span_observe, dt, nu, lamb_oseen, Lx, Ly, "result", Nx,Ny,thread_num);
    calc_manager.run();
}