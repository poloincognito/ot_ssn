# %%
import ott
import yaml
import os
import jax.numpy as jnp
import matplotlib.pyplot as plt

from ml_collections import ConfigDict
from cellot.utils.loaders import load_data

# --config ./configs/tasks/4i.yaml --config ./configs/models/cellot.yaml --config.data.target cisplatin


# %%
def get_dataset(dataset_name, batchsize):

    # Specify the path to your YAML file
    config_path = "datasets/configs/4i.yaml"
    model_path = "datasets/configs/cellot.yaml"

    # Load the YAML file
    with open(config_path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    with open(model_path, "r") as yaml_file:
        model = yaml.safe_load(yaml_file)

    # Complete the configuration
    config["data"]["target"] = dataset_name
    config["dataloader"]["batch_size"] = batchsize

    config.update(model)

    config = ConfigDict(config)

    # Load the data
    data = load_data(config)

    return data


# %%
dataset_name = "cisplatin"

os.chdir("..")
data = get_dataset(dataset_name, 128)


# %%
def compute_ott_estim_error(data):
    # Parse the data
    X_train = jnp.array(next(iter(data["train"]["source"])))
    Y_train = jnp.array(next(iter(data["train"]["target"])))
    X_test = jnp.array(next(iter(data["test"]["source"])))
    Y_test = jnp.array(next(iter(data["test"]["target"])))

    # Solve via Sinkhorn (train)
    geom = ott.geometry.pointcloud.PointCloud(
        X_train, Y_train
    )  # Define an euclidean geometry
    problem = ott.problems.linear.linear_problem.LinearProblem(
        geom
    )  # Define your problem
    solver = ott.solvers.linear.sinkhorn.Sinkhorn()  # Select the Sinkhorn solver
    out = solver(problem)
    ot_train_estim = out.primal_cost

    # Solve via Sinkhorn (test)
    geom = ott.geometry.pointcloud.PointCloud(
        X_test, Y_test
    )  # Define an euclidean geometry
    problem = ott.problems.linear.linear_problem.LinearProblem(
        geom
    )  # Define your problem
    solver = ott.solvers.linear.sinkhorn.Sinkhorn()  # Select the Sinkhorn solver
    out = solver(problem)
    ot_test_estim = out.primal_cost
    return abs(float(ot_train_estim - ot_test_estim))


# %%
dataset_names = [
    "cisplatin",
    "cisplatin_oparalib",
    "decitabine",
    "trametinib_panobinostat",
    "sorafenib",
    "palbociclib",
]
batchsizes = [32, 64, 128, 256, 512, 1024]

# Compute errors
errors = {}
for dataset_name in dataset_names:
    print(f"Dataset: {dataset_name}")
    _errors = []
    for batchsize in batchsizes:
        print(f"Batchsize: {batchsize}")
        data = get_dataset(dataset_name, batchsize)
        error = compute_ott_estim_error(data)
        _errors.append(error)
    errors[dataset_name] = _errors

# %%
# # Plot errors
# plt.plot(batchsize, _errors)
# plt.xscale("log")
# plt.xlabel("Batchsize")
# plt.ylabel("Error")
# plt.title("Error in OT estimation\n{}".format(dataset_name))
