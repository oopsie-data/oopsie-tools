# Oopsie Tools

Tools for collecting, annotating, inspecting, and converting robotic manipulation rollout data.

This repository currently provides around:

- HDF5 episode recording (`EpisodeRecorder`)
- Web + CLI annotation workflows
- In-the-loop annotation during policy rollout

---

For detailed explanations on how to use our tooling and contribute to the project, please visit [our website](https://oopsie-data.com/).

## Repository structure

The main tooling for data gathering and annotation is located in `oopsie-tools`.

We provide example scripts for automatically collecting and annotating evaluation data while running policy inference in examples. Currently we support the evaluation scripts supported by `openpi` and Trossen robotics `act_plus_plus` repository. If you want to run a different evaluation script, check out the detailed instructions on integrating 