# SPEC

## What it is
A software-only positioning stack that gives a drone continuous 6-DoF pose (position plus orientation) with quantified uncertainty when GPS is unavailable, jammed, or spoofed. It fuses a camera, IMU, barometer, and magnetometer with pre-loaded satellite map imagery, runs an integrity monitor that watches every input source for degradation or attack, and fails over gracefully without operator action. The whole thing is proven inside your 3D simulator with a live metrics dashboard before it ever touches hardware. No custom hardware, no RF transmission, nothing you need a license to build.

## The problem it solves
GPS is a faint, unauthenticated broadcast from space. Jamming is routine in conflict zones, spoofing incidents rose sharply through 2025, and a drone that trusts a lied-to GPS flies itself into the ground. The expensive answer is a $10,000 fiber-optic inertial system. This project is the commodity-sensor answer: a $399-class compute module with a camera and IMU that keeps the drone located anyway.

## Components, in detail
**Sensor abstraction layer.** Timestamped, calibrated inputs: camera (mono or stereo), IMU (accel plus gyro at 200 Hz or better), barometer, magnetometer. Handles time synchronization and extrinsic calibration (where each sensor sits relative to the others). Unglamorous, and the place real projects silently die, so it gets its own module and its own tests.

**VIO frontend.** Detects and tracks visual features frame to frame (corner features plus optical-flow tracking), preintegrates IMU measurements between frames, and selects keyframes. Output is visual residuals and inertial constraints, not poses yet.

**VIO estimator core.** A filter-based MSCKF estimator (the OpenVINS approach: efficient, well understood) maintaining pose, velocity, IMU biases, and feature positions in one state vector. This is integrated open source, not written from scratch.

**Aiding layer.** Three absolute references that bound the VIO drift:
- Satellite image matcher: registers the downward camera view against pre-loaded georeferenced map tiles, producing absolute position fixes. Student-scale work has demonstrated ~2.8 m median error with this approach.
- Optical flow plus rangefinder: damps velocity drift and holds altitude, cheap and robust.
- Barometer and magnetometer: altitude and heading priors.

**Fusion backend.** A factor graph (or EKF) that blends the VIO relative motion with the absolute aiding fixes, propagating covariance honestly through the whole chain. This is what turns two estimates into one trusted estimate with an error ellipse you can defend.

**Integrity monitor.** The moat. Cross-checks every independent estimate against the others (VIO vs GPS vs map matcher), gates measurements by covariance, and runs degradation detectors: feature-count collapse, motion-blur metric, illumination shock. Outputs a health vector per source and a failover decision. Its job is to make the stack honest about its own uncertainty.

**GPS attack interface.** GPS measurements enter only through a gate owned by the integrity monitor. When spoofing is flagged, GPS is rejected and the stack continues on VIO plus aiding with no discontinuity the flight controller can see.

**Output interface.** Pose plus covariance at a fixed rate, formatted as MAVLink VISION_POSITION_ESTIMATE, so a real PX4 or ArduPilot could consume it later without changes.

**Sim harness.** Synthetic pinhole-camera rendering from your 3D scene, IMU/baro/mag models with configurable noise, GPS with jamming and spoof injection, and a ground-truth trajectory recorder. Deterministic seeds make every run reproducible.

**Dashboard.** Live 3D drone view, estimated vs ground-truth trajectory plot, error-over-time charts, a per-scenario scoreboard, and an attack timeline showing when the spoof started, when it was detected, and when failover happened.

## Features (what it does, from the user's chair)
- Holds position knowledge through total GPS loss, with drift bounded by map aiding on long routes.
- Detects GPS spoofing by cross-consistency and reports detection latency as a metric.
- Fails over from GPS to denied mode automatically, mid-flight, with no operator input.
- Announces its own degradation: if vision collapses in a dark forest, it says so and leans on inertial plus baro rather than hallucinating.
- Replays any run deterministically from a seed for debugging and demos.
- Runs a scripted scenario suite: urban canyon, forest, twilight, aggressive maneuvers, mid-flight GPS cutover, drag-off spoof attack.
- Live camera mode: point a webcam at your desk, watch feature tracks and the estimated trajectory draw itself.

## Data flow, one frame's journey
A timestamped image and IMU sample arrive at the sensor layer. The frontend tracks features and preintegrates the IMU. The estimator updates the state vector. Periodically, the aiding layer registers a frame against the satellite map and the fusion backend blends that absolute fix in. The integrity monitor scores every source, gates or rejects outliers, and publishes the pose plus covariance. The dashboard renders the drone, the trajectory, and the error plots live.

## Evaluation: how you know it works
Metrics: ATE RMSE in meters, relative pose error, drift as percent of distance traveled, map-aiding relocalization success rate, spoof detection latency, time to recover after GPS loss, CPU load on the target hardware. Targets: under 1% drift unaided on EuRoC-class data, map-aided error in the low single meters, spoof flagged within seconds of onset. Benchmarks: EuRoC and TUM-VI datasets first, sim scenarios second, real hardware only as a stretch goal.

## Roadmap
- Phase 0: repo, sensor models, eval harness proven on a dead-reckoning baseline.
- Phase 1: VIO core integrated and hitting the drift target.
- Phase 2: satellite-image aiding for bounded long-range error.
- Phase 3: integrity monitor, spoof interface, and the dashboard.
- Phase 4 (optional): Raspberry Pi plus camera field test, or a learned depth-cue upgrade.

## What it is not
Not a weapon or a targeting system. Not signal-level RF work; spoof detection stays at measurement consistency, which keeps it clean. Not a flight controller; it publishes pose estimates and lets PX4 do the flying. Not real-time on a Pi in v1; the Pi is a stretch goal, not a promise.

## Definition of done
The demo. One scripted sim run: the drone flies a 2 km route, GPS is jammed at t=60 s, spoofed at t=120 s, vision degrades through a forest segment, and the dashboard shows continuous pose, detection flags, failover events, and final error numbers on one screen. That recording is the resume artifact, the interview walkthrough, and the seed of the startup pitch all at once.
