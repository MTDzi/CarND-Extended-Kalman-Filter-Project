#include <math.h>
#include "kalman_filter.h"


using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;

#include <iostream>



// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.


KalmanFilter::KalmanFilter() {}


KalmanFilter::~KalmanFilter() {}


void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
    x_ = x_in;
    P_ = P_in;
    F_ = F_in;
    H_ = H_in;
    R_ = R_in;
    Q_ = Q_in;
}


void KalmanFilter::Predict() {
    x_ = F_ * x_; // + u;
    P_ = F_ * P_ * F_.transpose() + Q_;
}


void KalmanFilter::Update(const VectorXd &z) {
    // LIDAR

    y_ = z - H_ * x_;
    MeasurementUpdate();
}


void KalmanFilter::UpdateEKF(const VectorXd &z) {
    // RADAR

    // To calculate y_ (difference between predicted state and measurement), we first need to transform the
    // prediction in Cartesian coordinates, into prediction in polar coordinates
    double px = x_(0);
    double py = x_(1);
    double vx = x_(2);
    double vy = x_(3);

    double rho = sqrt(px*px + py*py);
    double phi;
    double rho_dot;
    if (fabs(rho) < 0.0001) {
        phi = 0;
        rho_dot = 0;
    } else {
        phi = atan2(py, px);
        rho_dot = (px*vx + py*vy) / rho;
    }

    Vector3d z_pred = {rho, phi, rho_dot};
    y_ = z - z_pred;

    if(y_(1) < -M_PI)
        y_(1) += 2*M_PI;
    else if(y_(1) > M_PI)
        y_(1) -= 2*M_PI;

    MeasurementUpdate();
}

void KalmanFilter::MeasurementUpdate() {
    // COMMON FOR BOTH

    MatrixXd S = H_ * P_ * H_.transpose() + R_;
    K_ = P_ * H_.transpose() * S.inverse();

    x_ += K_ * y_;
    int size = x_.size();
    I_ = MatrixXd::Identity(size, size);
    P_ = (I_ - K_ * H_) * P_;
}
