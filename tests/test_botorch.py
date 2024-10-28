import pytest

import numpy as np
import pandas as pd
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

import gumbi as gmb
from gumbi.utils.misc import s
from gumbi.regression.botorch import GP

cars = sns.load_dataset('mpg').dropna().astype({'weight':float, 'model_year':float})

def test_single_input_single_output():
    ds = gmb.DataSet(cars,
                    outputs=['mpg', 'acceleration'],
                    log_vars=['mpg', 'acceleration', 'weight', 'horsepower', 'displacement'])
    
    gp = GP(ds)
    gp.fit(outputs=['mpg'], continuous_dims=['horsepower'], continuous_kernel='RBF', seed=0, ARD=True)

    X = gp.prepare_grid()
    y = gp.predict_grid();

    pp = gmb.ParrayPlotter(X, y)
    pp.plot()
    sns.scatterplot(data=cars, x='horsepower', y='mpg', color=sns.cubehelix_palette()[-1], alpha=0.5);
    plt.close('all')

def test_single_input_single_output_proposals():
    ds = gmb.DataSet(cars,
                    outputs=['mpg', 'acceleration'],
                    log_vars=['mpg', 'acceleration', 'weight', 'horsepower', 'displacement'])
    
    ds2 = gmb.DataSet(
        cars.sample(5, random_state=1),
        outputs=["mpg", "acceleration"],
        log_vars=["mpg", "acceleration", "weight", "horsepower", "displacement"],
    )

    gp = GP(ds2)

    gp.fit(
        outputs=["mpg"],
        continuous_dims=["horsepower"],
        continuous_kernel="RBF",
        seed=0,
        ARD=True,
    )

    X = gp.prepare_grid(
        limits=gmb.parray(
            horsepower=[ds.wide.horsepower.min(), ds.wide.horsepower.max()], stdzr=gp.stdzr
        )
    )
    y = gp.predict_grid()

    # sns.scatterplot(data=cars, x='horsepower', y='mpg', color=sns.cubehelix_palette()[-1], alpha=0.05);
    # pp = gmb.ParrayPlotter(X, y)
    # pp.plot()
    # sns.scatterplot(
    #     data=ds2.wide, x="horsepower", y="mpg", color=sns.cubehelix_palette()[-1]
    # )

    bounds = np.array([[X.z.values().min(0)+0.1, X.z.values().max(0)-0.1]]).T
    # bounds = torch.from_numpy(bounds).to(gp.device)

    candidates, _ = gp.propose(maximize=True, q=5, bounds=bounds)

    # for x in candidates:
    #     plt.axvline(x.values(), color='C0', linestyle='--', alpha=0.5)
    
    


@pytest.mark.parametrize("multitask_kernel", ["Independent", "Hadamard", "Kronecker"])
def test_single_input_multi_output(multitask_kernel):
    ds = gmb.DataSet(cars,
                    outputs=['mpg', 'acceleration'],
                    log_vars=['mpg', 'acceleration', 'weight', 'horsepower', 'displacement'])
    gp = GP(ds)
    gp.fit(outputs=['mpg', 'acceleration'], continuous_dims=['horsepower'], multitask_kernel=multitask_kernel)

    X = gp.prepare_grid()
    Y = gp.predict_grid()

    axs = plt.subplots(2,1, figsize=(6, 8))[1]
    for ax, output in zip(axs, gp.outputs):
        y = Y.get(output)

        gmb.ParrayPlotter(X, y).plot(ax=ax)

        sns.scatterplot(data=cars, x='horsepower', y=output, color=sns.cubehelix_palette()[-1], alpha=0.5, ax=ax);
        
    ## Visualize learned correlations
        
    # X = gp.parray(horsepower=[50, 100, 150, 200])
    # plt.figure()
    # gp.predict_points(X)
    # y = gp.predictions

    # samples_df = pd.concat(
    #     [
    #         pd.DataFrame(point.dist.rvs(10000, random_state=j).as_dict()).assign(
    #             horsepower=hp.values()
    #         )
    #         for j, (point, hp) in enumerate(zip(y, X))
    #     ],
    #     ignore_index=True,
    # )

    # sns.pairplot(
    #     samples_df, hue="horsepower", kind="kde", corner=True, plot_kws={"levels": 1}
    # )
        
    plt.close('all')


@pytest.mark.parametrize("multitask_kernel, ard",
                         product(["Independent", "Hadamard", "Kronecker"], [False, True]),
                         ids=map(', '.join, product(["Independent", "Hadamard", "Kronecker"], ["ARD", "no ARD"]))
                         )
def test_multi_input_multi_output(multitask_kernel, ard):

    ds = gmb.DataSet(cars,
                    outputs=['mpg', 'acceleration'],
                    log_vars=['mpg', 'acceleration', 'weight', 'horsepower', 'displacement'])
    gp = GP(ds)

    gp.fit(outputs=['mpg', 'acceleration'], continuous_dims=['horsepower', 'weight'], multitask_kernel=multitask_kernel, ard=ard);

    XY = gp.prepare_grid()
    Z = gp.predict_grid(with_noise=False)
        
    X = XY["horsepower"]
    Y = XY["weight"]

    axs = plt.subplots(2,1, figsize=(6, 8))[1]
    for ax, output in zip(axs, gp.outputs):
        z = Z.get(output)
        μ = z.μ
        σ = z.dist.std()
        norm = plt.Normalize(μ.min(), μ.max())

        plt.sca(ax)
        pp = gmb.ParrayPlotter(X, Y, z)
        pp(plt.contourf, levels=20, cmap="pink", norm=norm)
        pp.colorbar(ax=ax)

        cs = ax.contour(
            X.values(), Y.values(), σ, levels=4, colors="0.2", linestyles="--"
        )
        ax.clabel(cs)

        sns.scatterplot(
            data=cars,
            x="horsepower",
            y="weight",
            hue=output,
            palette="pink",
            hue_norm=norm,
            ax=ax,
        )
        ax.legend().remove()
        
    plt.close('all')
    
    
def test_categorical_continuous_input():
    ds = gmb.DataSet(cars,
                    outputs=['mpg', 'acceleration'],
                    log_vars=['mpg', 'acceleration', 'weight', 'horsepower', 'displacement'])
    gp = GP(ds)
    gp.fit(outputs=['mpg'], categorical_dims=['origin'], continuous_dims=['horsepower']);

    X = gp.prepare_grid()
    axs = plt.subplots(1,3, figsize=(18, 4))[1]
    for i, (ax, origin) in enumerate(zip(axs, cars.origin.unique())):
        y = gp.predict_grid(categorical_levels={'origin': origin}, with_noise=False)

        gmb.ParrayPlotter(X, y).plot(ax=ax, palette=sns.light_palette(f'C{i}'))

        sns.scatterplot(data=cars[cars.origin==origin], x='horsepower', y='mpg', color=f'C{i}', alpha=0.5, ax=ax);
        ax.set_title(origin)
        ax.set_ylim([5, 50])


@pytest.mark.parametrize("multitask_kernel, sequential",
                         product(["Independent", "Hadamard", "Kronecker"], [False, True]),
                         ids=map(', '.join, product(["Independent", "Hadamard", "Kronecker"], ["batch", "sequential"]))
                         )
def test_single_input_multi_output_proposals(multitask_kernel, sequential):

    q = 4

    ds4 = gmb.DataSet(
        cars.sample(5, random_state=0),
        outputs=["mpg", "horsepower", "acceleration", "displacement", "model_year"],
        log_vars=["mpg", "acceleration", "weight", "horsepower", "displacement"],
    )

    gp = GP(ds4)
    gp.fit(
        outputs=["mpg", "horsepower"],
        continuous_dims=["weight"],
        continuous_kernel="RBF",
        seed=0,
        ARD=True,
        multitask_kernel=multitask_kernel,
    )

    candidates, _ = gp.propose(q=q, sequential=sequential)
    preds = gp.predict_points(candidates)

    X = gp.prepare_grid()
    Y = gp.predict_grid()

    axs = plt.subplots(1, 3, figsize=(9, 3))[1]
    for ax, output in zip(axs, gp.outputs):
        y = Y.get(output)

        sns.scatterplot(data=cars, x="weight", y=output, color="k", alpha=0.05, ax=ax)

        gmb.ParrayPlotter(X, y).plot(ax=ax, palette=sns.light_palette("C0"))

        for weight, pred in zip(candidates, preds):
            y = pred.get(output).μ
            y_err = np.abs(np.array([pred.get(output).dist.interval(0.95)]) - y).T
            ax.errorbar(weight, y, y_err, fmt="o", color="C1", ms=10, mec="w")

        sns.scatterplot(
            data=ds4.wide, x="weight", y=output, color="C0", ax=ax, zorder=10
        )

    ax = axs[2]
    sns.scatterplot(
        data=cars,
        x="mpg",
        y="horsepower",
        ax=ax,
        color="k",
        alpha=0.05,
        label="Full Dataset",
    )
    ax.plot(Y.get("mpg").μ, Y.get("horsepower").μ, label="Prediction", color="C0")

    x = preds.get("mpg").μ
    y = preds.get("horsepower").μ
    x_err = np.abs(np.array(preds.get("mpg").dist.interval(0.95)) - x)
    y_err = np.abs(np.array(preds.get("horsepower").dist.interval(0.95)) - y)
    ax.errorbar(
        x, y, y_err, x_err, fmt="o", color="C1", ms=10, label="Proposals", mec="w"
    )

    sns.scatterplot(
        data=ds4.wide,
        x="mpg",
        y="horsepower",
        ax=ax,
        label="Observations",
        color="C0",
        zorder=10,
    )
    ax.legend()

    axs[0].set_title("MPG vs Weight")
    axs[1].set_title("Horsepower vs Weight")
    axs[2].set_title("Pareto Front")
    plt.suptitle(f"{multitask_kernel} Kernel: " + ("Sequential" if sequential else "Batch"))
    plt.tight_layout()
    
    plt.close('all')
    

derivative_params = (
    list(product([1, 2, 3], [1], [None])) + list(product([1, 2, 3], [2, 3], ["Independent", "Kronecker", "Hadamard"]))
)
derivative_test_ids = [
    f"{n_out} output{s(n_out)}, {n_in} input{s(n_in)}"
    + (f", {multitask_kernel} kernel" if multitask_kernel is not None else "")
    for n_in, n_out, multitask_kernel in derivative_params
]


@pytest.mark.parametrize("n_in, n_out, multitask_kernel", derivative_params, ids=derivative_test_ids)
def test_derivatives(n_in, n_out, multitask_kernel, verbose=True):
    res = 5
    if verbose:
        msg = f"Testing {n_out} output{s(n_out)} and {n_in} input{s(n_in)}"
        if multitask_kernel is not None:
            msg += f" with {multitask_kernel} multitask kernel"
        print(msg)
    inputs = np.meshgrid(*[np.linspace(0, 1, res) for _ in range(n_in)])
    outputs = [np.sum(np.stack(inputs, axis=-1), axis=-1) for _ in range(n_out)]

    ds_test = gmb.DataSet(
        pd.DataFrame(
            {f"input_{i}": inputs[i].flatten() for i in range(n_in)}
            | {f"output_{j}": outputs[j].flatten() for j in range(n_out)}
        ),
        outputs=[f"output_{j}" for j in range(n_out)],
    )

    gp_test = GP(ds_test)

    gp_test.fit(
        outputs=[f"output_{j}" for j in range(n_out)],
        continuous_dims=[f"input_{i}" for i in range(n_in)],
        # linear_dims=[f"input_{i}" for i in range(n_in)],
        continuous_kernel="RBF",
        seed=0,
        ARD=True,
        multitask_kernel=multitask_kernel,
    )

    X = gp_test.prepare_grid(
        limits=gp_test.parray(
            **{
                f"input_{i}": (
                    inputs[i].min() + np.ptp(inputs[i]) / 4,
                    inputs[i].max() - np.ptp(inputs[i]) / 4,
                )
                for i in range(n_in)
            },
            stdzd=False,
        ),
        resolution=res * 2,
    )
    y = gp_test.predict_grid()

    dydX_norm = gp_test.predict_grid_grad(norm=True)
    extremes = f"{dydX_norm.values().min()}, {dydX_norm.values().max()}"
    assert len(dydX_norm.names) == n_out
    assert np.allclose(dydX_norm.values(), np.sqrt(n_in), atol=0.1), extremes

    dydX = gp_test.predict_grid_grad(norm=False)
    extremes = f"{dydX.values().min()}, {dydX.values().max()}"
    assert len(dydX.names) == n_out * n_in
    assert np.allclose(dydX.values(), 1, atol=0.1), extremes
