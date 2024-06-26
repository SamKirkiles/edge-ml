# ML at the edge on Cloudflare Workers

<p align="center">
  <img src="https://i.imgur.com/qzszKIP.jpeg" width="350" title="hover text">
</p>


I've been interested in Cloudflare's new Workers AI platform for a few months now. The service puts GPUs at the edge. This means Cloudflare physically installs GPUs into its data centers all over the world, getting the models as physically close to its users as possible to minimize latency. The service speeds up inference time by pre-loading the weights of popular models such as Llama and Mistral on the GPU and provides an API endpoint to Cloudflare users. Currently, however, there is no way to deploy your own GPU-based custom ML model on Cloudflare Workers AI. The goal of this project: Can we put our own (very small) custom ML models at the edge using Cloudflare Workers?

To run your own version, follow the tutorial linked below! There is a fair bit of setup which needs to be done on your Cloudflare account which the tutorial walks you through.

[Read more about it here](https://medium.com/@sam.kirkiles/ml-at-the-edge-deploying-custom-ml-models-on-cloudflare-workers-49d1d8fef9cd)
