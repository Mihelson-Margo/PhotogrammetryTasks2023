#include "triangulation.h"

#include "defines.h"

#include <iostream>
#include <Eigen/SVD>

// По положениям камер и ключевых точкам определяем точку в трехмерном пространстве
// Задача эквивалентна поиску точки пересечения двух (или более) лучей
// Используем DLT метод, составляем систему уравнений. Система похожа на систему для гомографии, там пары уравнений получались из выражений вида x (cross) Hx = 0, а здесь будет x (cross) PX = 0
// (см. Hartley & Zisserman p.312)
cv::Vec4d phg::triangulatePoint(const cv::Matx34d *Ps, const cv::Vec3d *ms, int count)
{
    // TODO
    // составление однородной системы + SVD
    // без подвохов

    int a_rows = 2 * count;
    int a_cols = 3;

    Eigen::MatrixXd A(a_rows, a_cols);
    Eigen::VectorXd b(a_rows);

    for (int i_pair = 0; i_pair < count; ++i_pair) {

        double x = ms[i_pair][0];
        double y = ms[i_pair][1];
        double z = ms[i_pair][2];

        for (int i = 0; i < 3; i++) {
            A(2*i_pair, i) = x * Ps[i_pair](2, i) - z * Ps[i_pair](0, i);
            A(2*i_pair+1, i) = y * Ps[i_pair](2, i) - z * Ps[i_pair](1, i);
        }
        b(2*i_pair) = - x * Ps[i_pair](2, 3) + z * Ps[i_pair](0, 3);
        b(2*i_pair + 1) = - y * Ps[i_pair](2, 3) + z * Ps[i_pair](1, 3);
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svda(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Eigen::MatrixXd s_inv(a_cols, a_rows);
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 3; j++) s_inv(i, j) = 0;
    }
    for (int i = 0; i < 3; i++) s_inv(i, i) = 1/svda.singularValues()[i];

    auto res = svda.matrixV() * s_inv * svda.matrixU().transpose() * b;

    cv::Vec4d res_cv;
    for (int i = 0; i < 3; i++) res_cv[i] = res[i];
    res_cv[3] = 1;
    return res_cv;
}
