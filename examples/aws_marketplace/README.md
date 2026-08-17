# AWS Marketplace

[`aws_train_quickstart.ipynb`](aws_train_quickstart.ipynb) runs a Causal GPT-RL
training job end to end: fetch a dataset into your own S3, launch the job with
the Algorithm ARN from your subscription, load a policy bundle while the job is
still running, and open the finished model with the public runtime.

Subscribe first:
<https://aws.amazon.com/marketplace/pp/prodview-is6jt3bcwkq5c>

The notebook is standalone — it installs what it needs and imports nothing from
this repository, so it runs the same whether you cloned this repo or opened the
notebook on its own in SageMaker.

What it assumes you already have: an active subscription, a SageMaker execution
role, and three S3 prefixes in the region you launch in. It brings its own
dataset, so you do not need one to get through it.

For what the job accepts and what it gives back, see
[`training/docs/aws/`](../../training/docs/aws/README.md).
