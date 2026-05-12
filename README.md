# Oopsie Tools

Tools for collecting, annotating, inspecting, and converting robotic manipulation rollout data.

This repository currently provides around:

- HDF5 episode recording (`EpisodeRecorder`)
- Web annotation workflows
- In-the-loop annotation during policy rollout

---

For detailed explanations on how to use our tooling and contribute to the project, please visit [our website](https://oopsie-data.com/).

For an overview of the steps necessary to integrate the tooling into your workflow and to contribute data to the official Oopsie Data repositories, check out [our quickstart guide](https://oopsie-data.com/quickstart).
You can also use the information in AI_CONTEXT.md to guide a coding agent through the setup.

## Repository structure

The main tooling for data gathering and annotation is located in `oopsie-tools`.

We provide example scripts for automatically collecting and annotating evaluation data while running policy inference in examples. Currently we support the evaluation scripts supported by `openpi` and Trossen robotics `act_plus_plus` repository. If you want to run a different evaluation script, check out the detailed instructions on integrating our tools into standard robot evaluation pipelines.

## Issue/PR for support requests

We are very aware that changing eval code and recording data while doing experiments can be a big hassle and cause friction. We are therefore happy to help you integrate the recording tool into your setup. Please let us know what robot platform and policy you are evaluating and where to find a sample of your evaluation & inference code.

## Contributing

You can use our toolset any time you like to record and annotate robot rollouts. To contribute your data to the official Oopsie Dataset, please follow the [sign-up instructions](https://oopsie-data.com/contributing/)!

If you run into any issues, please do not hesitate to contact the team via mail or raise an issue here.
