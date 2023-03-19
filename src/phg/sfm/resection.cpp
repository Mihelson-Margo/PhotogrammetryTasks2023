#include "resection.h"

#include <Eigen/SVD>
#include <iostream>
#include "sfm_utils.h"
#include "defines.h"

namespace {

    // Сделать из первого минора 3х3 матрицу вращения, скомпенсировать масштаб у компоненты сдвига
    matrix34d canonicalizeP(const matrix34d &P)
    {
        matrix3d RR = P.get_minor<3, 3>(0, 0);
        vector3d tt;
        tt[0] = P(0, 3);
        tt[1] = P(1, 3);
        tt[2] = P(2, 3);

        if (cv::determinant(RR) < 0) {
            RR *= -1;
            tt *= -1;
        }

        double sc = 0;
        for (int i = 0; i < 9; i++) {
            sc += RR.val[i] * RR.val[i];
        }
        sc = std::sqrt(3 / sc);

        Eigen::MatrixXd RRe;
        copy(RR, RRe);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(RRe, Eigen::ComputeFullU | Eigen::ComputeFullV);
        RRe = svd.matrixU() * svd.matrixV().transpose();
        copy(RRe, RR);

        tt *= sc;

        matrix34d result;
        for (int i = 0; i < 9; ++i) {
            result(i / 3, i % 3) = RR(i / 3, i % 3);
        }
        result(0, 3) = tt(0);
        result(1, 3) = tt(1);
        result(2, 3) = tt(2);

        return result;
    }

    // (см. Hartley & Zisserman p.178)
    cv::Matx34d estimateCameraMatrixDLT(const cv::Vec3d *Xs, const cv::Vec3d *xs, int count)
    {
        // TODO
        using mat = Eigen::MatrixXd;
        using vec = Eigen::VectorXd;

        int a_rows = 2 * count;
        int a_cols = 12;

        mat A(a_rows, a_cols);

        for (int i_pair = 0; i_pair < count; ++i_pair) {

            double x = xs[i_pair][0];
            double y = xs[i_pair][1];
            double z = xs[i_pair][2];
            cv::Vec4d v(Xs[i_pair][0], Xs[i_pair][1], Xs[i_pair][2], 1);

            for (int i = 0; i < 4; i++) {
                A(2*i_pair, i) = - z * v[i];
                A(2*i_pair, 4+i) = 0;
                A(2*i_pair, 8+i) = x * v[i];
            }

            for (int i = 0; i < 4; i++) {
                A(2*i_pair+1, i) = 0;
                A(2*i_pair+1, 4+i) = -z * v[i];
                A(2*i_pair+1, 8+i) = y * v[i];
            }
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

        Eigen::VectorXd null_space(12);
        for (int i = 0; i < 12; i++) null_space[i] = svda.matrixV()(i, 11);

        //std::cout << "Should be 0: " << A * null_space << "\n";

        matrix34d result;
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 4; j++) {
                result(i, j) = null_space[4*i + j];
            }
        }
        return canonicalizeP(result);
    }


    // По трехмерным точкам и их проекциям на изображении определяем положение камеры
    cv::Matx34d estimateCameraMatrixRANSAC(const phg::Calibration &calib, const std::vector<cv::Vec3d> &X, const std::vector<cv::Vec2d> &x)
    {
        // TODO
        if (X.size() != x.size()) {
            throw std::runtime_error("estimateCameraMatrixRANSAC: X.size() != x.size()");
        }

        const int n_points = X.size();

        // https://en.wikipedia.org/wiki/Random_sample_consensus#Parameters
        // будет отличаться от случая с гомографией
        const int n_trials = 100000; //TODO;

        const double threshold_px = 3;

        const int n_samples = 6;
        uint64_t seed = 1;

        int best_support = 0;
        cv::Matx34d best_P;

        std::vector<int> sample;
        for (int i_trial = 0; i_trial < n_trials; ++i_trial) {
            phg::randomSample(sample, n_points, n_samples, &seed);

            cv::Vec3d ms0[n_samples];
            cv::Vec3d ms1[n_samples];
            for (int i = 0; i < n_samples; ++i) {
                ms0[i] = X[sample[i]];
                ms1[i] = calib.unproject(x[sample[i]]);
            }

            cv::Matx34d P = estimateCameraMatrixDLT(ms0, ms1, n_samples);

            int support = 0;
            for (int i = 0; i < n_points; ++i) {
                //TODO спроецировать 3Д точку в пиксель с использованием P и calib;
                cv::Vec3d q = calib.project(
                        P * cv::Vec4d(X[i][0], X[i][1], X[i][2], 1)
                );
                cv::Vec2d px(q[0]/q[2], q[1]/q[2]);

                /*
                if (i == sample[0]) {
                    std::cout << "P = " << P << "\n";
                    std::cout << "calib = " << calib.K() << "\n";
                    std::cout << "X = " << X[i] << "\n";
                    std::cout << "ms[0] = " << ms0[0] << "\n";
                    std::cout << "q = " << q << "\n";
                    std::cout << "px = " << px << "\n";
                    std::cout << "x[i] = " << x[i] << "\n";
                } */
                if (cv::norm(px - x[i]) < threshold_px) {
                    ++support;
                }
            }

            if (support > best_support) {
                best_support = support;
                best_P = P;

                std::cout << "estimateCameraMatrixRANSAC : support: " << best_support << "/" << n_points << std::endl;

                if (best_support == n_points) {
                    break;
                }
            }
        }

        std::cout << "estimateCameraMatrixRANSAC : best support: " << best_support << "/" << n_points << std::endl;

        if (best_support == 0) {
            throw std::runtime_error("estimateCameraMatrixRANSAC : failed to estimate camera matrix");
        }

        return best_P;
    }


}

cv::Matx34d phg::findCameraMatrix(const Calibration &calib, const std::vector <cv::Vec3d> &X, const std::vector <cv::Vec2d> &x) {
    return estimateCameraMatrixRANSAC(calib, X, x);
}
