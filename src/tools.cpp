#include <iostream>
#include "tools.h"


using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;


Tools::Tools() {}


Tools::~Tools() {}


VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {

    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    VectorXd diff;
    VectorXd squared_diff;

    // Check the validity of the following inputs:
    if(estimations.size() != ground_truth.size() || estimations.size() == 0){
        cout << "Invalid estimation or ground_truth data" << endl;
        return rmse;
    }

    // Accumulate squared residuals
    for(int i=0; i<estimations.size(); ++i) {
        diff = estimations[i] - ground_truth[i];
        squared_diff = diff.array() * diff.array();
        rmse += squared_diff;
    }

    // Calculate the mean
    rmse /= estimations.size();

    // Calculate the squared root
    rmse = rmse.array().sqrt();

    // Return the result
    return rmse;
}


MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
    MatrixXd Hj(3,4);
    Hj << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;

    //recover state parameters
    float px = x_state(0);
    float py = x_state(1);
    float vx = x_state(2);
    float vy = x_state(3);

    // Frequently occurring denominators
    float p = sqrt(px*px + py*py);
    float p2 = p * p;
    float p3 = p2 * p;

    //check division by zero
    if(fabs(p) < 0.0001) {
        cout << "CalculateJacobian() - Error - Division by zero" << endl;
        return Hj;
    }

    //compute the Jacobian matrix
    Hj << px/p,                py/p,                0,    0,
          -py/p2,              px/p2,               0,    0,
          py*(vx*py-vy*px)/p3, px*(vy*px-vx*py)/p3, px/p, py/p;

    return Hj;
}
