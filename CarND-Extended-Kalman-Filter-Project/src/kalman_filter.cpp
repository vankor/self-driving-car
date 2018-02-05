#include "kalman_filter.h"
#define PI 3.14
#define THLD 0.0001

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init( VectorXd &x_in,
                         MatrixXd &P_in,
                         MatrixXd &F_in,
                         MatrixXd &H_in,
                         MatrixXd &R_in,
                         MatrixXd &Q_in )
{
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
    I_ = Eigen::MatrixXd::Identity(4,4);
}

void KalmanFilter::Predict()
{
    // predict the state
    x_ = F_*x_;
    MatrixXd Ft = F_.transpose();
    P_ = F_*P_*Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z)
{
    // Update equations
    VectorXd y = z - H_*x_;
    KalmanUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z)
{
    // avoid dividing by zero.
    if( x_(0) == 0. && x_(1) == 0. )
        return;

    float rho = sqrt(x_(0)*x_(0) + x_(1)*x_(1));
    float phi = atan2(x_(1), x_(0));
    float rho_dot;
    if (fabs(rho) < THLD) {
        rho_dot = 0;
    } else {
        rho_dot = (x_(0)*x_(2) + x_(1)*x_(3))/rho;
    }

    VectorXd hofx(3);
    hofx << rho, phi, rho_dot;

    // Update the state using Extended Kalman Filter equations
    VectorXd y = z - hofx;
    if( y[1] > PI ){
        y[1] -= 2.f*PI;
    }

    if( y[1] < -PI ) {
        y[1] += 2.f * PI;
    }

    KalmanUpdate(y);
}

void KalmanFilter::KalmanUpdate(const VectorXd &y){
    MatrixXd Ht = H_.transpose();
    MatrixXd S = H_ * P_ * Ht + R_;
    MatrixXd Si = S.inverse();
    MatrixXd PHt = P_ * Ht;
    MatrixXd K = PHt * Si;

    // Compute new state
    x_ = x_ + ( K*y );
    P_ = ( I_ - K*H_ )*P_;
}

