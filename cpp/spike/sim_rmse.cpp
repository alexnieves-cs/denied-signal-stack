// SPDX-License-Identifier: GPL-3.0-or-later
// sim_rmse: run the OpenVINS simulator + MSCKF (GT init, same loop as run_simulation) and report
// position RMSE, drift %, orientation/position NEES and per-epoch MSCKF/SLAM update counts.
// Usage: sim_rmse <estimator_config.yaml> [per_epoch.csv]
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>

#include "core/VioManager.h"
#include "sim/Simulator.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "utils/print.h"
#include "utils/quat_ops.h"
#include "utils/sensor_data.h"

using namespace ov_msckf;

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <estimator_config.yaml> [per_epoch.csv]\n", argv[0]);
    return 2;
  }
  auto parser = std::make_shared<ov_core::YamlParser>(argv[1]);
  std::string verbosity = "WARNING";
  parser->parse_config("verbosity", verbosity);
  ov_core::Printer::setPrintLevel(verbosity);

  VioManagerOptions params;
  params.print_and_load(parser);
  params.print_and_load_simulation(parser);
  params.num_opencv_threads = 0;
  params.use_multi_threading_pubs = false;
  params.use_multi_threading_subs = false;
  auto sim = std::make_shared<Simulator>(params);
  auto sys = std::make_shared<VioManager>(params);
  if (!parser->successful()) {
    std::fprintf(stderr, "unable to parse all parameters\n");
    return 1;
  }

  double next_imu_time = sim->current_timestamp() + 1.0 / params.sim_freq_imu;
  Eigen::Matrix<double, 17, 1> imustate;
  if (!sim->get_state(next_imu_time, imustate)) {
    std::fprintf(stderr, "could not get initial state\n");
    return 1;
  }
  const double dt_ci = sim->get_true_parameters().calib_camimu_dt;
  imustate(0, 0) -= dt_ci;
  sys->initialize_with_gt(imustate);

  std::ofstream csv;
  if (argc > 2) {
    csv.open(argv[2]);
    csv << "t,px,py,pz,gx,gy,gz,err,nees_ori,nees_pos,n_msckf,n_slam\n";
  }

  double buffer_timecam = -1;
  std::vector<int> buffer_camids;
  std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> buffer_feats;

  size_t n = 0, n_zero_msckf = 0, max_zero_run = 0, zero_run = 0, zero_first10 = 0;
  double se = 0, nees_o = 0, nees_p = 0, path = 0, max_err = 0, last_err = 0;
  bool have_prev = false, diverged = false;
  Eigen::Vector3d prev_gt;
  double t0 = -1;

  while (sim->ok()) {
    ov_core::ImuData imu;
    if (sim->get_next_imu(imu.timestamp, imu.wm, imu.am)) sys->feed_measurement_imu(imu);

    double time_cam;
    std::vector<int> camids;
    std::vector<std::vector<std::pair<size_t, Eigen::VectorXf>>> feats;
    if (!sim->get_next_cam(time_cam, camids, feats)) continue;
    if (buffer_timecam != -1) {
      sys->feed_measurement_simulation(buffer_timecam, buffer_camids, buffer_feats);
      auto state = sys->get_state();
      Eigen::Matrix<double, 17, 1> gt;
      if (sim->get_state(state->_timestamp + dt_ci, gt)) {
        Eigen::Vector4d q_true = gt.block(1, 0, 4, 1);
        Eigen::Vector3d p_true = gt.block(5, 0, 3, 1);
        Eigen::Vector4d q_est = state->_imu->quat();
        Eigen::Vector3d p_est = state->_imu->pos();
        // JPL error state: q_true = dq (x) q_est, dq ~ [theta/2, 1]
        Eigen::Vector4d dq = ov_core::quat_multiply(q_true, ov_core::Inv(q_est));
        Eigen::Vector3d dth = 2.0 * dq.block(0, 0, 3, 1);
        Eigen::Vector3d dp = p_true - p_est;
        std::vector<std::shared_ptr<ov_type::Type>> order = {state->_imu->q(), state->_imu->p()};
        Eigen::MatrixXd P = StateHelper::get_marginal_covariance(state, order);
        double no = dth.transpose() * P.block(0, 0, 3, 3).inverse() * dth;
        double np = dp.transpose() * P.block(3, 3, 3, 3).inverse() * dp;
        size_t nm = sys->get_good_features_MSCKF().size();
        size_t ns = sys->get_features_SLAM().size();
        if (t0 < 0) t0 = state->_timestamp;
        if (have_prev) path += (p_true - prev_gt).norm();
        prev_gt = p_true;
        have_prev = true;
        double err = dp.norm();
        if (!std::isfinite(err) || (path > 50 && err > 0.05 * path)) diverged = true;
        if (nm + ns == 0) {
          n_zero_msckf++;
          zero_run++;
          if (state->_timestamp - t0 < 10.0) zero_first10++;
        } else {
          zero_run = 0;
        }
        max_zero_run = std::max(max_zero_run, zero_run);
        se += err * err;
        nees_o += no;
        nees_p += np;
        max_err = std::max(max_err, err);
        last_err = err;
        n++;
        if (csv.is_open())
          csv << std::fixed << state->_timestamp << "," << p_est(0) << "," << p_est(1) << "," << p_est(2) << "," << p_true(0) << ","
              << p_true(1) << "," << p_true(2) << "," << err << "," << no << "," << np << "," << nm << "," << ns << "\n";
      }
    }
    buffer_timecam = time_cam;
    buffer_camids = camids;
    buffer_feats = feats;
  }
  if (n == 0) {
    std::printf("RESULT epochs=0\n");
    return 1;
  }
  double rmse = std::sqrt(se / n);
  const double cam_hz = params.sim_freq_cam;
  std::printf("RESULT epochs=%zu path_m=%.1f rmse_m=%.4f max_err_m=%.3f final_err_m=%.3f drift_pct=%.3f "
              "nees_ori=%.3f nees_pos=%.3f zero_update_epochs=%zu max_zero_run_s=%.2f zero_update_first10s=%zu diverged=%d\n",
              n, path, rmse, max_err, last_err, path > 0 ? 100.0 * rmse / path : 0.0, nees_o / n, nees_p / n, n_zero_msckf,
              max_zero_run / cam_hz, zero_first10, diverged ? 1 : 0);
  return std::isfinite(rmse) ? 0 : 1;
}
