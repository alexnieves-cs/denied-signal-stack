// SPDX-License-Identifier: GPL-3.0-or-later
// klt_smoke: TrackKLT on 20 synthetic CV_8UC1 frames with two independently shifted texture regions
// (parallax, so the F-matrix RANSAC is not degenerate), then cv::imread of a PNG we wrote.
// Usage: klt_smoke <out_dir>
#include <cstdio>
#include <memory>
#include <random>
#include <set>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "cam/CamRadtan.h"
#include "track/TrackKLT.h"
#include "utils/print.h"
#include "utils/sensor_data.h"

int main(int argc, char **argv) {
  std::string out = argc > 1 ? argv[1] : ".";
  ov_core::Printer::setPrintLevel("WARNING");
  const int W = 640, H = 480;
  auto cam = std::make_shared<ov_core::CamRadtan>(W, H);
  Eigen::Matrix<double, 8, 1> calib;
  calib << 400, 400, W / 2.0, H / 2.0, 0, 0, 0, 0;
  cam->set_value(calib);
  std::unordered_map<size_t, std::shared_ptr<ov_core::CamBase>> cams = {{0, cam}};
  ov_core::TrackKLT tracker(cams, 200, 0, false, ov_core::TrackBase::HISTOGRAM, 15, 5, 5, 10);

  // Two big random textures, blurred so they have trackable corners.
  std::mt19937 rng(42);
  auto make_tex = [&](int w, int h) {
    cv::Mat t(h, w, CV_8UC1);
    std::uniform_int_distribution<int> d(0, 255);
    for (int r = 0; r < h; r++)
      for (int c = 0; c < w; c++) t.at<uchar>(r, c) = (uchar)d(rng);
    cv::GaussianBlur(t, t, cv::Size(0, 0), 2.0);
    cv::normalize(t, t, 0, 255, cv::NORM_MINMAX);
    return t;
  };
  cv::Mat far = make_tex(W + 200, H + 200), near = make_tex(W / 2 + 200, H + 200);

  size_t min_tracks = 1e9, last_tracks = 0;
  std::set<size_t> prev_ids;
  for (int k = 0; k < 20; k++) {
    cv::Mat img(H, W, CV_8UC1);
    int sf = k * 2, sn = k * 5;  // far region moves 2 px/frame, near region 5 px/frame
    far(cv::Rect(sf, 50, W, H)).copyTo(img);
    near(cv::Rect(sn, 50, W / 2, H)).copyTo(img(cv::Rect(W / 2, 0, W / 2, H)));
    ov_core::CameraData msg;
    msg.timestamp = 0.05 * k;
    msg.sensor_ids = {0};
    msg.images = {img};
    msg.masks = {cv::Mat::zeros(H, W, CV_8UC1)};
    tracker.feed_new_camera(msg);
    // tracked = ids in this frame that were also present in the previous frame (KLT continuity)
    auto last_ids = tracker.get_last_ids();  // returned by value
    std::set<size_t> now(last_ids[0].begin(), last_ids[0].end());
    size_t tracked = 0;
    for (size_t id : now) tracked += prev_ids.count(id);
    prev_ids = now;
    if (k > 0) min_tracks = std::min(min_tracks, tracked);
    last_tracks = tracked;
    std::printf("frame %d ids=%zu tracked=%zu\n", k, last_ids[0].size(), tracked);
    if (k == 0) cv::imwrite(out + "/klt_smoke_frame0.png", img);
  }
  cv::Mat back = cv::imread(out + "/klt_smoke_frame0.png", cv::IMREAD_GRAYSCALE);
  bool imread_ok = !back.empty() && back.cols == W && back.rows == H;
  std::printf("RESULT min_tracked_frames1to19=%zu last_tracked=%zu imread_ok=%d\n", min_tracks, last_tracks, imread_ok ? 1 : 0);
  return (min_tracks >= 100 && imread_ok) ? 0 : 1;
}
