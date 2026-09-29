// SPDX-License-Identifier: GPL-3.0-or-later
// ovserver: OpenVINS MSCKF behind a strict length-prefixed request/response protocol on stdin/stdout.
// See ../PROTOCOL.md. OpenVINS prints to stdout, so fd 1 is redirected to stderr at startup and the
// protocol is written to a private duplicate of the original stdout.
#include <unistd.h>

#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "core/VioManager.h"
#include "core/VioManagerOptions.h"
#include "state/State.h"
#include "state/StateHelper.h"
#include "track/TrackBase.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

namespace {

enum MsgType : uint8_t {
  MSG_INIT = 1,
  MSG_IMU = 2,
  MSG_CAM = 3,
  MSG_INIT_GT = 4,
  MSG_REANCHOR = 5,
  MSG_SHUTDOWN = 6,
  MSG_CAM_STEREO = 7,
  MSG_PING = 8,
  // responses
  RSP_ACK = 0x81,
  RSP_STATE = 0x83,
};

enum Status : uint8_t {
  ST_OK = 0,
  ST_NOT_INITIALIZED = 1,
  ST_ERROR = 2,
  ST_NEED_IMU = 3,
  ST_BAD_REQUEST = 4,
};

constexpr uint32_t kMaxPayload = 64u << 20;  // 64 MiB
constexpr size_t kMaxTrackList = 200;

int g_out = -1;

bool read_exact(int fd, void *buf, size_t n) {
  auto *p = static_cast<uint8_t *>(buf);
  while (n > 0) {
    ssize_t r = ::read(fd, p, n);
    if (r == 0) return false;
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

bool write_exact(int fd, const void *buf, size_t n) {
  auto *p = static_cast<const uint8_t *>(buf);
  while (n > 0) {
    ssize_t r = ::write(fd, p, n);
    if (r < 0) {
      if (errno == EINTR) continue;
      return false;
    }
    p += r;
    n -= static_cast<size_t>(r);
  }
  return true;
}

struct Writer {
  std::vector<uint8_t> b;
  template <typename T> void put(T v) {
    const auto *p = reinterpret_cast<const uint8_t *>(&v);
    b.insert(b.end(), p, p + sizeof(T));
  }
};

struct Reader {
  const uint8_t *p;
  size_t n, i = 0;
  Reader(const std::vector<uint8_t> &v) : p(v.data()), n(v.size()) {}
  bool has(size_t k) const { return i + k <= n; }
  template <typename T> T get() {
    T v;
    std::memcpy(&v, p + i, sizeof(T));
    i += sizeof(T);
    return v;
  }
};

bool send(uint8_t type, const std::vector<uint8_t> &payload) {
  uint32_t len = static_cast<uint32_t>(payload.size());
  uint8_t hdr[5];
  std::memcpy(hdr, &len, 4);
  hdr[4] = type;
  return write_exact(g_out, hdr, 5) && write_exact(g_out, payload.data(), payload.size());
}

bool ack(Status s, const std::string &msg = "") {
  Writer w;
  w.put<uint8_t>(s);
  w.b.insert(w.b.end(), msg.begin(), msg.end());
  return send(RSP_ACK, w.b);
}

// Subclass only to read the tracker's protected state (frontend health).
class ServerVio : public ov_msckf::VioManager {
public:
  explicit ServerVio(ov_msckf::VioManagerOptions &p) : VioManager(p) {}
  bool has_state() const { return is_initialized_vio; }  // initialized() also demands a first update
  // initialize_with_gt() leaves the tracker at init_max_features; the dynamic initializer would raise it.
  void full_tracking(const ov_msckf::VioManagerOptions &p) {
    trackFEATS->set_num_features(static_cast<int>(std::floor((double)p.num_pts / (double)p.state_options.num_cameras)));
  }
  void tracks(int cam, std::vector<std::tuple<uint32_t, float, float>> &out) {
    auto ids = trackFEATS->get_last_ids();
    auto obs = trackFEATS->get_last_obs();
    auto it_i = ids.find(cam);
    auto it_o = obs.find(cam);
    if (it_i == ids.end() || it_o == obs.end()) return;
    for (size_t k = 0; k < it_i->second.size() && k < it_o->second.size(); k++)
      out.emplace_back(static_cast<uint32_t>(it_i->second[k]), it_o->second[k].pt.x, it_o->second[k].pt.y);
  }
};

struct Server {
  std::unique_ptr<ov_msckf::VioManagerOptions> params;
  std::shared_ptr<ServerVio> vio;
  double last_imu_t = -1;
  bool gt_initialized = false;
  std::deque<ov_core::ImuData> imu_hist;  // replayed into a fresh estimator on REANCHOR

  void feed_imu(const ov_core::ImuData &m) {
    vio->feed_measurement_imu(m);
    last_imu_t = m.timestamp;
    imu_hist.push_back(m);
    while (!imu_hist.empty() && imu_hist.front().timestamp < m.timestamp - 3.0) imu_hist.pop_front();
  }

  bool ready() const { return vio != nullptr; }

  Status init(const std::string &path, std::string &err) {
    if (access(path.c_str(), R_OK) != 0) {
      err = "config not readable: " + path;
      return ST_BAD_REQUEST;
    }
    auto parser = std::make_shared<ov_core::YamlParser>(path);
    std::string verbosity = "WARNING";
    parser->parse_config("verbosity", verbosity);
    ov_core::Printer::setPrintLevel(verbosity);
    params = std::make_unique<ov_msckf::VioManagerOptions>();
    params->print_and_load(parser);
    params->use_multi_threading_pubs = false;  // lockstep: every CAM is fully processed before we reply
    params->use_multi_threading_subs = false;
    if (!parser->successful()) {
      err = "unable to parse all parameters";
      params.reset();
      return ST_BAD_REQUEST;
    }
    vio = std::make_shared<ServerVio>(*params);
    imu_hist.clear();
    last_imu_t = -1;
    gt_initialized = false;
    return ST_OK;
  }

  static bool finite(const std::vector<double> &v) {
    for (double x : v)
      if (!std::isfinite(x)) return false;
    return true;
  }

  // imustate = [t, q_GtoI(JPL xyzw), p_IinG, v_IinG, bg, ba]
  Status init_gt(const Eigen::Matrix<double, 17, 1> &x, const Eigen::MatrixXd *cov) {
    if (cov) {
      // Re-anchor: a fresh estimator (drops all clones/features) seeded with the given marginals.
      vio = std::make_shared<ServerVio>(*params);
      for (const auto &m : imu_hist) vio->feed_measurement_imu(m);
    }
    Eigen::Matrix<double, 17, 1> s = x;
    s.block(1, 0, 4, 1).normalize();
    vio->initialize_with_gt(s);
    vio->full_tracking(*params);
    if (cov) {
      auto state = vio->get_state();
      std::vector<std::shared_ptr<ov_type::Type>> order = {state->_imu};
      Eigen::MatrixXd P = 0.5 * (*cov + cov->transpose());
      ov_msckf::StateHelper::set_initial_covariance(state, P, order);
    }
    gt_initialized = true;
    return ST_OK;
  }

  bool reply_state(Status st) {
    Writer w;
    w.put<uint8_t>(st);
    std::vector<std::tuple<uint32_t, float, float>> tracks;
    uint32_t n_msckf = 0, n_slam = 0;
    if (vio && (st == ST_OK || st == ST_NOT_INITIALIZED) && vio->has_state()) {
      auto state = vio->get_state();
      w.put<double>(state->_timestamp);
      Eigen::Vector4d q = state->_imu->quat();
      Eigen::Vector3d p = state->_imu->pos(), v = state->_imu->vel();
      for (int i = 0; i < 4; i++) w.put<double>(q(i));
      for (int i = 0; i < 3; i++) w.put<double>(p(i));
      for (int i = 0; i < 3; i++) w.put<double>(v(i));
      std::vector<std::shared_ptr<ov_type::Type>> order = {state->_imu->p(), state->_imu->q()};
      Eigen::MatrixXd P = ov_msckf::StateHelper::get_marginal_covariance(state, order);
      for (int r = 0; r < 6; r++)
        for (int c = 0; c < 6; c++) w.put<double>(P(r, c));
      n_msckf = static_cast<uint32_t>(vio->get_good_features_MSCKF().size());
      n_slam = static_cast<uint32_t>(vio->get_features_SLAM().size());
    } else {
      w.put<double>(std::nan(""));
      for (int i = 0; i < 4 + 3 + 3 + 36; i++) w.put<double>(0.0);
    }
    if (vio) vio->tracks(0, tracks);
    w.put<uint32_t>(static_cast<uint32_t>(tracks.size()));
    w.put<uint32_t>(n_msckf);
    w.put<uint32_t>(n_slam);
    uint16_t nl = static_cast<uint16_t>(std::min(tracks.size(), kMaxTrackList));
    w.put<uint16_t>(nl);
    for (size_t k = 0; k < nl; k++) {
      w.put<uint32_t>(std::get<0>(tracks[k]));
      w.put<float>(std::get<1>(tracks[k]));
      w.put<float>(std::get<2>(tracks[k]));
    }
    return send(RSP_STATE, w.b);
  }

  // Parse one (cam_id, w, h, pixels) block, validated against the loaded intrinsics.
  bool parse_image(Reader &r, int &cam, cv::Mat &img, std::string &err) {
    if (!r.has(1 + 4 + 4)) {
      err = "short image header";
      return false;
    }
    cam = r.get<uint8_t>();
    uint32_t w = r.get<uint32_t>(), h = r.get<uint32_t>();
    auto it = params->camera_intrinsics.find(static_cast<size_t>(cam));
    if (it == params->camera_intrinsics.end() || cam >= params->state_options.num_cameras) {
      err = "unknown cam_id " + std::to_string(cam);
      return false;
    }
    int ew = it->second->w(), eh = it->second->h();
    if (params->downsample_cameras) {
      ew *= 2;
      eh *= 2;
    }
    if (static_cast<int>(w) != ew || static_cast<int>(h) != eh) {
      err = "image size " + std::to_string(w) + "x" + std::to_string(h) + " != " + std::to_string(ew) + "x" + std::to_string(eh);
      return false;
    }
    size_t npix = static_cast<size_t>(w) * h;
    if (!r.has(npix)) {
      err = "short pixel payload";
      return false;
    }
    img = cv::Mat(static_cast<int>(h), static_cast<int>(w), CV_8UC1);
    std::memcpy(img.data, r.p + r.i, npix);
    r.i += npix;
    return true;
  }

  bool handle_cam(const std::vector<uint8_t> &payload, int ncams) {
    if (!ready()) return reply_state(ST_BAD_REQUEST);
    Reader r(payload);
    if (!r.has(8)) return reply_state(ST_BAD_REQUEST);
    ov_core::CameraData msg;
    msg.timestamp = r.get<double>();
    if (!std::isfinite(msg.timestamp)) return reply_state(ST_BAD_REQUEST);
    for (int k = 0; k < ncams; k++) {
      int cam;
      cv::Mat img;
      std::string err;
      if (!parse_image(r, cam, img, err)) {
        PRINT_WARNING("[ovserver] rejected CAM: %s\n", err.c_str());
        return reply_state(ST_BAD_REQUEST);
      }
      msg.sensor_ids.push_back(cam);
      msg.images.push_back(img);
      if (params->use_mask && params->masks.count(static_cast<size_t>(cam)) && params->masks.at(cam).size() == img.size())
        msg.masks.push_back(params->masks.at(cam));
      else
        msg.masks.push_back(cv::Mat::zeros(img.rows, img.cols, CV_8UC1));
    }
    if (r.i != r.n) return reply_state(ST_BAD_REQUEST);
    if (ncams == 2 && msg.sensor_ids[0] == msg.sensor_ids[1]) return reply_state(ST_BAD_REQUEST);
    // OpenVINS propagates to the camera time: it needs IMU data at or past it.
    if (last_imu_t < msg.timestamp + params->calib_camimu_dt) return reply_state(ST_NEED_IMU);
    try {
      vio->feed_measurement_camera(msg);
    } catch (const std::exception &e) {
      PRINT_ERROR("[ovserver] exception in update: %s\n", e.what());
      return reply_state(ST_ERROR);
    }
    return reply_state(vio->has_state() ? ST_OK : ST_NOT_INITIALIZED);
  }
};

}  // namespace

int main(int, char **) {
  // Protocol on a private copy of stdout; everything OpenVINS prints goes to stderr.
  g_out = dup(STDOUT_FILENO);
  if (g_out < 0 || dup2(STDERR_FILENO, STDOUT_FILENO) < 0) {
    std::perror("ovserver: dup");
    return 1;
  }
  ov_core::Printer::setPrintLevel("WARNING");
  Server srv;
  std::vector<uint8_t> payload;

  while (true) {
    uint8_t hdr[5];
    if (!read_exact(STDIN_FILENO, hdr, 5)) return 0;  // EOF: client went away
    uint32_t len;
    std::memcpy(&len, hdr, 4);
    uint8_t type = hdr[4];
    if (len > kMaxPayload) {
      // Cannot trust the stream any more: discard the claimed bytes in chunks, then report.
      std::vector<uint8_t> sink(1 << 16);
      uint64_t left = len;
      while (left > 0) {
        size_t k = static_cast<size_t>(std::min<uint64_t>(left, sink.size()));
        if (!read_exact(STDIN_FILENO, sink.data(), k)) return 0;
        left -= k;
      }
      ack(ST_BAD_REQUEST, "payload too large");
      continue;
    }
    payload.resize(len);
    if (len > 0 && !read_exact(STDIN_FILENO, payload.data(), len)) return 0;

    bool ok = true;
    switch (type) {
    case MSG_PING:
      ok = ack(ST_OK, "ovserver 1");
      break;
    case MSG_INIT: {
      std::string path(payload.begin(), payload.end()), err;
      Status st = srv.init(path, err);
      ok = ack(st, err);
      break;
    }
    case MSG_IMU: {
      if (!srv.ready() || len != 7 * 8) {
        ok = ack(ST_BAD_REQUEST, "IMU needs INIT and 56 bytes");
        break;
      }
      std::vector<double> v(7);
      std::memcpy(v.data(), payload.data(), 56);
      if (!Server::finite(v) || (srv.last_imu_t >= 0 && v[0] <= srv.last_imu_t)) {
        ok = ack(ST_BAD_REQUEST, "non-finite or non-increasing IMU");
        break;
      }
      ov_core::ImuData m;
      m.timestamp = v[0];
      m.wm << v[1], v[2], v[3];
      m.am << v[4], v[5], v[6];
      srv.feed_imu(m);
      ok = ack(ST_OK);
      break;
    }
    case MSG_CAM:
      ok = srv.handle_cam(payload, 1);
      break;
    case MSG_CAM_STEREO:
      ok = srv.handle_cam(payload, 2);
      break;
    case MSG_INIT_GT:
    case MSG_REANCHOR: {
      size_t want = (type == MSG_INIT_GT) ? 17 * 8 : (17 + 225) * 8;
      if (!srv.ready() || len != want) {
        ok = ack(ST_BAD_REQUEST, "INIT_GT/REANCHOR needs INIT and exact size");
        break;
      }
      std::vector<double> v(len / 8);
      std::memcpy(v.data(), payload.data(), len);
      if (!Server::finite(v)) {
        ok = ack(ST_BAD_REQUEST, "non-finite state");
        break;
      }
      Eigen::Matrix<double, 17, 1> x;
      for (int i = 0; i < 17; i++) x(i) = v[i];
      if (type == MSG_INIT_GT) {
        ok = ack(srv.init_gt(x, nullptr));
      } else {
        Eigen::MatrixXd P(15, 15);
        for (int r = 0; r < 15; r++)
          for (int c = 0; c < 15; c++) P(r, c) = v[17 + r * 15 + c];
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(0.5 * (P + P.transpose()));
        if (es.eigenvalues().minCoeff() <= 0) {
          ok = ack(ST_BAD_REQUEST, "REANCHOR covariance not positive definite");
          break;
        }
        ok = ack(srv.init_gt(x, &P));
      }
      break;
    }
    case MSG_SHUTDOWN:
      ack(ST_OK);
      return 0;
    default:
      ok = ack(ST_BAD_REQUEST, "unknown message type " + std::to_string(type));
    }
    if (!ok) return 0;  // client closed the pipe
  }
}
