#include "PID.h"
#include <cmath>
#include <iostream>
#include <limits>


using namespace std;

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp, double Ki, double Kd) {
    pid_koefs = {Kp, Kd, Ki};
    pid_errors = {0.0, 0.0, 0.0};

    use_twiddle = true;
    already_added = false;
    already_subtracted = false;
    dp = {0.1*pid_koefs[0], 0.1*pid_koefs[1], 0.1*pid_koefs[2]};
    total_error = 0;
    best_error = std::numeric_limits<double>::max();
    step_number = 1;
    pid_element_index = 2;
    update_error_after_steps   = 100;
    update_params_steps  = 1500;

}

void PID::UpdateError(double cte) {
    
    if (step_number == 1) {
        pid_errors[1] = 0.0;
    } else {
        pid_errors[1] = cte - pid_errors[0];
    }
    pid_errors[0] = cte;
    pid_errors[2] += cte;


    cte_prev_ = cte_;
    cte_ = cte;


    // update total error only if we're past number of settle steps
    if (step_number % (update_error_after_steps   + update_params_steps ) > update_error_after_steps  ){
        total_error += pow(cte,2);
    }

    if (step_number % (update_error_after_steps   + update_params_steps ) == 0 && use_twiddle){
        if (total_error < best_error) {
            best_error = total_error;
            if (step_number !=  update_error_after_steps   + update_params_steps ) {
                dp[pid_element_index] *= 1.1;
            }
            pid_element_index = (pid_element_index + 1) % 3;
            already_added = false;
            already_subtracted = false;
        }
        if (!already_added && !already_subtracted) {
            pid_koefs[pid_element_index] += dp[pid_element_index];
            already_added = true;
        } else if (already_added && !already_subtracted) {
            pid_koefs[pid_element_index] -= 2 * dp[pid_element_index];
            already_subtracted = true;
        } else {
            pid_koefs[pid_element_index] += dp[pid_element_index];
            dp[pid_element_index] *= 0.9;
            pid_element_index = (pid_element_index + 1) % 3;
            already_added = false;
            already_subtracted = false;
        }
        total_error = 0;

        cout << "twiddle parameters" << "P: " << pid_koefs[0] << ", I: " << pid_koefs[2] << ", D: " << pid_koefs[1] << endl;
    }
    step_number++;
}

double PID::TotalError() {
     return pid_koefs[0] * pid_errors[0] + pid_koefs[1] * pid_errors[1]
                                       + pid_koefs[2] * pid_errors[2];
}

