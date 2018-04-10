#ifndef PID_H
#define PID_H
#include <vector>

class PID {
public:
  /*
  * Errors
  */
  std::vector<double> pid_errors;


  /*
  * Coefficients
  */ 
  std::vector<double> pid_koefs;


    std::vector<double> dp;
    double total_error;
    double best_error;
    int step_number;
    int pid_element_index;
    int update_error_after_steps;
    int update_params_steps;

    double cte_prev_;

    double cte_;

    bool already_added;
    bool already_subtracted;
    bool use_twiddle;



  /*
  * Constructor
  */
  PID();

  /*
  * Destructor.
  */
  virtual ~PID();

  /*
  * Initialize PID.
  */
  void Init(double Kp, double Ki, double Kd);

  /*
  * Update the PID error variables given cross track error.
  */
  void UpdateError(double cte);

  /*
  * Calculate the total PID error.
  */
  double TotalError();

};

#endif /* PID_H */
