#ifndef SPECTRAL_HPP
#define SPECTRAL_HPP

#include<fftw3.h>
#include<vector>
#include<functional>
#include<string> 
#include <thread> 
#include <cmath>
#include <iostream>
#include <fstream>
#include "path.hpp"

class Calculater 
{
public:
    Calculater(int num_iter, int span_observe, double dt, double nu, 
               std::function<double(double, double)> init_omega,
               double Lx, double Ly, const std::string& filedir, 
               int nX, int nY, int num_threads);

    ~Calculater()
    {
        fftw_cleanup_threads();
    }

    void record(int iter);

    void debug_report();

    void run();

private:
    // Setting of experiment
    int rank = 2;
    int num_iter;
    int span_observe;
    double dt;
    std::string filedir;
    std::string filename;
    std::string result_dir;

    // VTE property
    double Lx;
    double Ly;
    double nu;
    int num_unknown = 1; // In VTE, omega is the unique unknown value.
    fftw_complex* omega_hat;
    fftw_complex* omega_hat_report;
    fftw_complex* linear_hat;

    // for runge-kutta
    fftw_complex* k1;
    fftw_complex* k2;
    fftw_complex* k3;
    fftw_complex* k4;

    // Data for FFTW3
    int nX;
    int nY;
    int num_threads;
    double* omega_data;
    double* domega_dx_data_pad;
    double* domega_dy_data_pad;
    double* dpsi_dx_data_pad;
    double* dpsi_dy_data_pad;
    double* nonlinear;

    fftw_complex* domega_dx_data_hat_pad;
    fftw_complex* domega_dy_data_hat_pad;
    fftw_complex* dpsi_dx_data_hat_pad;
    fftw_complex* dpsi_dy_data_hat_pad;
    fftw_complex* nonlinear_hat;

    // Plan for FFTW3 
    fftw_plan plan_omega_r2c;
    fftw_plan plan_domega_dx_c2r_pad;
    fftw_plan plan_domega_dy_c2r_pad;
    fftw_plan plan_dpsi_dx_c2r_pad;
    fftw_plan plan_dpsi_dy_c2r_pad;
    fftw_plan plan_nonlinear_r2c_pad;
    fftw_plan plan_omega_hat_c2r;

};

void map_parallel_2d_complex(fftw_complex* data, std::function<void(int, int, fftw_complex)> func,
                     int nX, int nY, int num_thread)
{
    auto job = [&](int threadID)
    {
        for(int i = threadID; i < nX; i += num_thread)
        {
            for(int j = 0; j < nY; ++j)
            {
                func(i, j, data[i * nY + j]);
            }
        }
    };

    std::vector<std::thread> jobs(num_thread);

    for(int k = 0; k < num_thread; ++k)
    {
        jobs[k] = std::thread(job, k);
    }

    for(auto& task : jobs)
    {
        task.join();
    }
}


void map_parallel_2d_double(double* data, std::function<void(int, int, double&)> func,
                     int nX, int nY, int num_thread)
{
    auto job = [&](int threadID)
    {
        for(int i = threadID; i < nX; i += num_thread)
        {
            for(int j = 0; j < nY; ++j)
            {
                func(i, j, data[i * nY + j]);
            }
        }
    };

    std::vector<std::thread> jobs(num_thread);

    for(int k = 0; k < num_thread; ++k)
    {
        jobs[k] = std::thread(job, k);
    }

    for(auto& task : jobs)
    {
        task.join();
    }
}

void parallel_2d(std::function<void(int, int)> func, int nX, int nY, int num_thread)
{
    auto job = [&](int threadID)
    {
        for(int i = threadID; i < nX; i += num_thread)
        {
            for(int j = 0; j < nY; ++j)
            {
                func(i, j);
            }
        }
    };

    std::vector<std::thread> jobs(num_thread);

    for(int k = 0; k < num_thread; ++k)
    {
        jobs[k] = std::thread(job, k);
    }

    for(auto& task : jobs)
    {
        task.join();
    }
}

// This is adhoc routine for runge-kutta.
// I added the argument for specifying phase(= 0,1,2,3).
// Because Runge-kutta Algorithm consists of four phases.
void parallel_2d_for_runge_kutta(std::function<void(int, int, int)> func, int nX, int nY, int num_thread, int phase)
{
    auto job = [&](int threadID)
    {
        for(int i = threadID; i < nX; i += num_thread)
        {
            for(int j = 0; j < nY; ++j)
            {
                func(i, j, phase);
            }
        }
    };

    std::vector<std::thread> jobs(num_thread);

    for(int k = 0; k < num_thread; ++k)
    {
        jobs[k] = std::thread(job, k);
    }

    for(auto& task : jobs)
    {
        task.join();
    }
}


#endif