#include "spectral.hpp"
#include <chrono>
#include <sys/stat.h>

Calculater::Calculater(int num_iter, int span_observe, double dt, double nu, 
            std::function<double(double, double)> init_omega,
            double Lx, double Ly, const std::string& filedir, 
            int nX, int nY, int num_threads):
            num_iter(num_iter), span_observe(span_observe), dt(dt), nu(nu),
            Lx(Lx), Ly(Ly), filedir(filedir), nX(nX), nY(nY), num_threads(num_threads)
{
    int total_real = nX * nY;
    int total_complex = nX * (nY/2 + 1);
    int total_real_pad = static_cast<int>(nX*3/2 * nY*3/2);
    int total_complex_pad = nX*3/2 * (nY*3/4 + 1);
    int nYp = nY * 3 / 2;
    int nXp = nX * 3 / 2;

    std::cout << "[SEQUENCE]: fftw allocation";
    omega_data = fftw_alloc_real(total_real);
    omega_hat = fftw_alloc_complex(total_complex);
    omega_hat_report = fftw_alloc_complex(total_complex);

    domega_dx_data_hat_pad = fftw_alloc_complex(total_complex_pad);
    domega_dx_data_pad = fftw_alloc_real(total_real_pad);

    domega_dy_data_hat_pad = fftw_alloc_complex(total_complex_pad);
    domega_dy_data_pad = fftw_alloc_real(total_real_pad);

    dpsi_dx_data_hat_pad = fftw_alloc_complex(total_complex_pad);
    dpsi_dx_data_pad = fftw_alloc_real(total_real_pad);

    dpsi_dy_data_hat_pad = fftw_alloc_complex(total_complex_pad);
    dpsi_dy_data_pad = fftw_alloc_real(total_real_pad);

    nonlinear = fftw_alloc_real(total_real_pad);
    nonlinear_hat = fftw_alloc_complex(total_complex_pad);

    linear_hat = fftw_alloc_complex(total_complex_pad);

    k1 = fftw_alloc_complex(total_complex);
    k2 = fftw_alloc_complex(total_complex);
    k3 = fftw_alloc_complex(total_complex);
    k4 = fftw_alloc_complex(total_complex);
    std::cout << " -> finish" << std::endl;

    // prepare threads
    std::cout << "[SEQUENCE]: fftw prepare thread"; 
    fftw_init_threads();
    fftw_plan_with_nthreads(num_threads);
    std::cout << " -> finish" << std::endl;

    // make plan
    std::cout << "[SEQUENCE]: fftw make plan";
    int ns[2] = {nX, nY};
    int ns_pad[2] = {3 * nX / 2, 3 * nY / 2};
    plan_omega_r2c = fftw_plan_dft_r2c(rank, ns, omega_data, omega_hat, FFTW_ESTIMATE);
    plan_domega_dx_c2r_pad = fftw_plan_dft_c2r(rank, ns_pad, domega_dx_data_hat_pad, domega_dx_data_pad, FFTW_ESTIMATE);
    plan_domega_dy_c2r_pad = fftw_plan_dft_c2r(rank, ns_pad, domega_dy_data_hat_pad, domega_dy_data_pad, FFTW_ESTIMATE);
    plan_dpsi_dx_c2r_pad = fftw_plan_dft_c2r(  rank, ns_pad, dpsi_dx_data_hat_pad,   dpsi_dx_data_pad,   FFTW_ESTIMATE);
    plan_dpsi_dy_c2r_pad = fftw_plan_dft_c2r(  rank, ns_pad, dpsi_dy_data_hat_pad,   dpsi_dy_data_pad,   FFTW_ESTIMATE);
    plan_nonlinear_r2c_pad = fftw_plan_dft_r2c(rank, ns_pad, nonlinear, nonlinear_hat, FFTW_ESTIMATE);
    plan_omega_hat_c2r = fftw_plan_dft_c2r(rank, ns, omega_hat_report, omega_data, FFTW_ESTIMATE);
    std::cout << " -> finish!" << std::endl;


    // calculate initial omega in wave space
    std::cout << "[SEQUENCE]: prepare initial omega";
    for(int i = 0; i < nX; ++i)
    {
        for(int j = 0; j < nY; ++j)
        {
            double xi = Lx * i / nX;
            double yj = Ly * j / nY;
            omega_data[i * nY + j] = init_omega(xi, yj);
        }
    }
    fftw_execute(plan_omega_r2c);
    // normalize (because fftw doesn't normalize transformation)
    auto normalize_forward = [nX,nY](int i, int j, fftw_complex factor) 
    {
        factor[0] *= 1.0 / (nX * nY);
        factor[1] *= 1.0 / (nX * nY);
    };
    map_parallel_2d_complex(omega_hat, normalize_forward, nX, nY/2 + 1, num_threads);
    std::cout << " -> finish" << std::endl;


    // prepare log file
    std::string proj_dir = PROJECT_DIR;
    result_dir = proj_dir + "/" + filedir;
    mkdir(result_dir.c_str(), 0777); // linux

    std::cout << "[SEQUENCE]: save situation";
    std::ofstream situation_file;
    situation_file.open(result_dir + "/situation.csv", std::ios::out);
    situation_file << ",height,width,dimX,dimY,iter_num,dt,time_span,nu" << std::endl; 
    situation_file << "," << Ly << ",";
    situation_file << Lx << ",";
    situation_file << nX << ",";
    situation_file << nY << ",";
    situation_file << num_iter << ",";
    situation_file << dt << ",";
    situation_file << span_observe << ",";
    situation_file << nu << std::endl;
    situation_file.close();
    std::cout << " -> finish" << std::endl;

    // log initial value;
    record(0);
};

void progress_bar(unsigned int finish_num, unsigned int progress_num, int bar_width, 
                  const std::chrono::nanoseconds& iter_time)
{
    int bar = bar_width * progress_num / finish_num;
    std::string arrow = "\r[";
    for(int i = 0; i < bar; ++i)
    {
        arrow += "=";
    }
    arrow += ">";
    for(int i = bar; i < bar_width - 1; ++i)
    {
        arrow += " ";
    }
    arrow += "]";
    arrow += " : " + std::to_string(progress_num) + "/" + std::to_string(finish_num);

    std::chrono::nanoseconds residual_time = iter_time * (finish_num - progress_num);

    std::chrono::hours res_hour = std::chrono::duration_cast<std::chrono::hours>(residual_time);
    residual_time -= res_hour;

    std::chrono::minutes res_min = std::chrono::duration_cast<std::chrono::minutes>(residual_time);
    residual_time -= res_min;

    std::chrono::seconds res_second = std::chrono::duration_cast<std::chrono::seconds>(residual_time);

    std::printf("%s || Remain --> (%ld h: %ld m: %ld s)", arrow.c_str(), 
                res_hour.count(), res_min.count(), res_second.count());
}


void Calculater::run()
{
    // note: index apart from 0-padding
    auto pad2unpad_x = [&](int index)
    {
        if(0 <= index && index <= nX/2)
        {
            return index;
        }
        else if(nX < index && index < 3*nX/2)
        {
            return index - nX/2;
        }
        else
        {
            std::cerr << "[ERROR]:160: out-of-range" << std::endl;
            return -1;
        }
    };

    auto pad2unpad_y = [&](int index)
    {
        if(0 <= index && index <= nY/2)
        {
            return index;
        }
        else if(nY < index && index < 3*nY/2)
        {
            return index - nY/2;
        }
        else
        {
            std::cerr << "[ERROR]:176: out-of-range" << std::endl;
            return -1;
        }
    };
    
    auto pad2wave_x = [&](int index)
    {
        if(0 <= index && index <= nX*3/4)
        {
            return index;
        }
        else if(nX*3/4 < index && index < nX*3/2)
        {
            return index - nX*3/2;
        }
        else
        {
            std::cerr << "[ERROR]:194: out-of-range" << std::endl;
            return -1;
        }
    };

    auto pad2wave_y = [&](int index)
    {
        if(0 <= index && index <= nY*3/4)
        {
            return index;
        }
        else if(nY*3/4 < index && index < nY*3/2)
        {
            return index - nY*3/2;
        }
        else
        {
            std::cerr << "[ERROR]:211: out-of-range" << std::endl;
            return -1;
        }
    };

    auto unpad2wave_x = [&](int index)
    {
        if(0 <= index && index <= nX/2)
        {

            return index;
        }
        else if(nX/2 < index && index < nX)
        {
            return index - nX;
        }
        else
        {
            std::cerr << "[ERROR]:229: out-of-range" << std::endl;
            return -1;
        }
    };

    auto unpad2wave_y = [&](int index)
    {
        if(0 <= index && index <= nY/2)
        {
            return index;
        }
        else if(nY/2 < index && index < nY)
        {
            return index - nY;
        }
        else
        {
            std::cerr << "[ERROR]:246: out-of-range" << std::endl;
            return -1;
        }
    };

    auto unpad2pad_x = [&](int index)
    {
        if(0 <= index && index <= nX/2)
        {
            return index;
        }
        else if(nX/2 < index && index < nX)
        {
            return index + nX/2;
        }
        else
        {
            std::cerr << "[ERROR]:246: out-of-range" << std::endl;
            return -1;
        }
    };

    auto unpad2pad_y = [&](int index)
    {
        if(0 <= index && index <= nY/2)
        {
            return index;
        }
        else if(nY/2 < index && index < nY)
        {
            return index + nY/2;
        }
        else
        {
            std::cerr << "[ERROR]:280: out-of-range" << std::endl;
            return -1;
        }
    };

    // set the input of fftw IDFT for nonlinear term (wave --> real)
    auto prepare_idft_and_set_linear = [&](int p, int q, int phase)
    {
        int pad_index = p * (nY*3/4 + 1) + q;
        if((nX/2 < p && p <= nX) || (nY/2 < q && q <= nY))
        {
            domega_dx_data_hat_pad[pad_index][0] = 0.0; // re
            domega_dx_data_hat_pad[pad_index][1] = 0.0; // im

            domega_dy_data_hat_pad[pad_index][0] = 0.0; // re
            domega_dy_data_hat_pad[pad_index][1] = 0.0; // im

            dpsi_dx_data_hat_pad[pad_index][0] = 0.0; // re
            dpsi_dx_data_hat_pad[pad_index][1] = 0.0; // im

            dpsi_dy_data_hat_pad[pad_index][0] = 0.0; // re
            dpsi_dy_data_hat_pad[pad_index][1] = 0.0; // im
            
            linear_hat[pad_index][0] = 0.0;
            linear_hat[pad_index][1] = 0.0;
            return;
        }

        int unpad_index_x = pad2unpad_x(p);
        int unpad_index_y = pad2unpad_y(q);
        int unpad_index = unpad_index_x * (nY/2 + 1) + unpad_index_y;
        int wave_x = pad2wave_x(p);
        int wave_y = pad2wave_y(q);

        double k_dash[2] = {2.0*M_PI/Lx*wave_x, 2.0*M_PI/Ly*wave_y};
        double k_square = k_dash[0]*k_dash[0] + k_dash[1]*k_dash[1];

        double omega_hat_re = omega_hat[unpad_index][0];
        double omega_hat_im = omega_hat[unpad_index][1];

        double omega_hat_phase_re, omega_hat_phase_im;
        if(phase == 0) // k1
        {
            omega_hat_phase_re = omega_hat_re;
            omega_hat_phase_im = omega_hat_im;
        }
        else if(phase == 1) // k2
        {
            omega_hat_phase_re = omega_hat_re + 0.5*dt*k1[unpad_index][0];
            omega_hat_phase_im = omega_hat_im + 0.5*dt*k1[unpad_index][1];
        }
        else if(phase == 2) // k3
        {
            omega_hat_phase_re = omega_hat_re + 0.5*dt*k2[unpad_index][0];
            omega_hat_phase_im = omega_hat_im + 0.5*dt*k2[unpad_index][1];
        }
        else
        {
            omega_hat_phase_re = omega_hat_re + dt*k3[unpad_index][0];
            omega_hat_phase_im = omega_hat_im + dt*k3[unpad_index][1];
        }


        fftw_complex psi_hat; 
        if(k_square < 1e-8)
        {
            psi_hat[0] = 0.0;
            psi_hat[1] = 0.0;
        }
        else
        {
            psi_hat[0] = omega_hat_phase_re/k_square;
            psi_hat[1] = omega_hat_phase_im/k_square;
        }


        domega_dx_data_hat_pad[pad_index][0] = -k_dash[0] * omega_hat_phase_im; // re
        domega_dx_data_hat_pad[pad_index][1] = k_dash[0] * omega_hat_phase_re; // im

        domega_dy_data_hat_pad[pad_index][0] = -k_dash[1] * omega_hat_phase_im; // re
        domega_dy_data_hat_pad[pad_index][1] = k_dash[1] * omega_hat_phase_re; // im

        dpsi_dx_data_hat_pad[pad_index][0] = -k_dash[0] * psi_hat[1]; // re
        dpsi_dx_data_hat_pad[pad_index][1] = k_dash[0] * psi_hat[0]; // im

        dpsi_dy_data_hat_pad[pad_index][0] = -k_dash[1] * psi_hat[1]; // re
        dpsi_dy_data_hat_pad[pad_index][1] = k_dash[1] * psi_hat[0]; // im

        linear_hat[pad_index][0] = -nu * k_square * omega_hat_phase_re; // re
        linear_hat[pad_index][1] = -nu * k_square * omega_hat_phase_im; // im

        return;
    };

    auto calc_nonlinear = [&](int p, int q, double& factor)
    {
        int pad_index = p * nY*3/2 + q;

        factor =  - dpsi_dy_data_pad[pad_index]*domega_dx_data_pad[pad_index] 
                  + dpsi_dx_data_pad[pad_index]*domega_dy_data_pad[pad_index];
    };
    
    auto normalize_pad = [&](int p, int q, fftw_complex factor)
    {
        double ratio = 1.0/(nX*3/2 * nY*3/2);
        factor[0] *= ratio;
        factor[1] *= ratio;
    };

    auto calc_rhs = [&](int p, int q, int phase)
    {
        int unpad_index = p * (nY/2 + 1) + q;

        int pad_index_x = unpad2pad_x(p);
        int pad_index_y = unpad2pad_y(q);
        int pad_index = pad_index_x * (nY*3/4 + 1) + pad_index_y;

        if(phase == 0) // k1
        {
            k1[unpad_index][0] = nonlinear_hat[pad_index][0] + linear_hat[pad_index][0];
            k1[unpad_index][1] = nonlinear_hat[pad_index][1] + linear_hat[pad_index][1];
        }
        else if(phase == 1) // k2
        {
            k2[unpad_index][0] = nonlinear_hat[pad_index][0] + linear_hat[pad_index][0];
            k2[unpad_index][1] = nonlinear_hat[pad_index][1] + linear_hat[pad_index][1];
        }
        else if(phase == 2) // k3
        {
            k3[unpad_index][0] = nonlinear_hat[pad_index][0] + linear_hat[pad_index][0];
            k3[unpad_index][1] = nonlinear_hat[pad_index][1] + linear_hat[pad_index][1];
        }
        else // k4
        {
            k4[unpad_index][0] = nonlinear_hat[pad_index][0] + linear_hat[pad_index][0];
            k4[unpad_index][1] = nonlinear_hat[pad_index][1] + linear_hat[pad_index][1];
        }
    };

    auto runge_kutta_update = [&](int p, int q, fftw_complex factor)
    {
        int unpad_index = p * (nY/2 + 1) + q;
        factor[0] = omega_hat[unpad_index][0] + dt/6.0 * (k1[unpad_index][0] + 2.0*(k2[unpad_index][0] + k3[unpad_index][0]) + k4[unpad_index][0]);
        factor[1] = omega_hat[unpad_index][1] + dt/6.0 * (k1[unpad_index][1] + 2.0*(k2[unpad_index][1] + k3[unpad_index][1]) + k4[unpad_index][1]);
    };

    // Integration with Spectral Method!
    std::cout << "[SEQUENCE]: Run spectral method" << std::endl;
    for(int iter = 1; iter <= num_iter; ++iter)
    {
        double elapse = iter * dt;
        std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

        for(int phase = 0; phase < 4; ++phase)
        {
            parallel_2d_for_runge_kutta(prepare_idft_and_set_linear, nX*3/2, nY*3/4 + 1, num_threads, phase);
            fftw_execute(plan_dpsi_dx_c2r_pad);
            fftw_execute(plan_dpsi_dy_c2r_pad);
            fftw_execute(plan_domega_dx_c2r_pad);
            fftw_execute(plan_domega_dy_c2r_pad);
            map_parallel_2d_double(nonlinear, calc_nonlinear, nX*3/2, nY*3/2, num_threads);
            fftw_execute(plan_nonlinear_r2c_pad);
            map_parallel_2d_complex(nonlinear_hat, normalize_pad, nX*3/2, nY*3/4 + 1, num_threads);

            // calculate k_(phase + 1)
            parallel_2d_for_runge_kutta(calc_rhs, nX, nY/2 + 1, num_threads, phase);
        }
        //renew omega_hat
        map_parallel_2d_complex(omega_hat, runge_kutta_update, nX, nY/2 + 1, num_threads);
        

        bool do_record = (iter % span_observe == 0);
        if(do_record) record(iter/span_observe);

        std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
        std::chrono::nanoseconds iter_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        progress_bar(num_iter, iter, 60, iter_time);
    }
    std::cout << std::endl;

}

void Calculater::record(int iter)
{
    std::ofstream snapshot_file;
    snapshot_file.open(result_dir + "/omega" + std::to_string(iter) + ".csv", std::ios::out); 

    for(int i = 0; i < nX; ++i)
    {
        for(int j = 0; j < nY/2 + 1; ++j)
        {
            int index = i * (nY/2 + 1) + j;
            omega_hat_report[index][0] = omega_hat[index][0];
            omega_hat_report[index][1] = omega_hat[index][1];
        }
    }

    // transform omega_hat to omega (wave->real)
    if(iter != 0) fftw_execute(plan_omega_hat_c2r);

    // footer
    for(int i = 0; i < nY; ++i)
    {
        snapshot_file << "," << Ly * i / nY;
    }
    snapshot_file << std::endl;
    
    for(int p = 0; p < nX; ++p)
    {
        snapshot_file << p * Lx / nX;
        for(int q = 0; q < nY; ++q)
        {
            snapshot_file << "," << omega_data[p * nY + q];
        }
        snapshot_file << std::endl;
    }

    snapshot_file.close();
}

void Calculater::debug_report()
{
    std::cout << "==========DEBUG===========" <<  std::endl;
    for(int p = 0; p < nX; ++p)
    {
        for(int q = 0; q < nY/2 + 1; ++q)
        {
            std::cout << "(" << omega_hat[p * (nY/2 + 1) + q][0] << ", " << omega_hat[p * (nY/2 + 1) +q][1] << "), ";
        }
        std::cout << std::endl;
    }
    std::cout << "===========================" << std::endl;
}