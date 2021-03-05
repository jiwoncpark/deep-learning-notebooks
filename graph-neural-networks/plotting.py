import corner
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_corner(pred, truth, plot_prior=True, training_samples=None, fig=None,
                posterior_label='BNN posterior', posterior_color='#d6616b'):
    """Plot the posterior samples against the training distribution

    Parameters
    ----------
    training_samples : np.array of shape `[n_training_examples, 3]`
        training labels
    pred : np.array of shape `[n_samples, 3]`
        samples from predictive distribution for a validation example

    """
    Y_dim = training_samples.shape[-1]
    labels = ['Label {:d}'.format(i) for i in range(Y_dim)]
    legend_elements = []
    # Plot training distribution (implicit prior)
    if plot_prior:
        fig = corner.corner(training_samples,
                            labels=labels,
                            color='tab:gray',
                            smooth=1.2,
                            alpha=0.5,
                            fill_contours=True,
                            plot_datapoints=False,
                            label_kwargs={'fontsize': 30},
                            plot_contours=True,
                            show_titles=False,
                            plot_density=False,
                            levels=[0.68, 0.95],
                            contour_kwargs=dict(linestyles='--'),
                            quiet=True,
                            fig=fig,
                            #range=[[-20, 20], [-2, 2], [4, 20]],
                            use_math_text=True,
                            hist_kwargs=dict(density=True,),
                            hist2d_kwargs=dict(pcolor_kwargs=dict(alpha=0.1)))
        legend_elements.append(Patch(facecolor='tab:gray',
                                     edgecolor='tab:gray',
                                     label=r'Training distribution'))
    # Plot prediction samples for given validation example
    _ = corner.corner(pred,
                      color=posterior_color,
                      smooth=1.0,
                      alpha=0.5,
                      labels=labels,
                      label_kwargs={'fontsize': 30},
                      fill_contours=True,
                      plot_datapoints=False,
                      plot_contours=True,
                      show_titles=True,
                      levels=[0.68, 0.95],
                      truths=truth,
                      truth_color='k',
                      contour_kwargs=dict(linestyles='solid',
                                          colors=posterior_color),
                      quiet=True,
                      title_fmt=".1g",
                      fig=fig,
                      #range=[[-20, 20], [-2, 2], [4, 20]],
                      title_kwargs={'fontsize': 18},
                      use_math_text=True,
                      hist_kwargs=dict(density=True,
                                       linewidth=2,
                                       histtype='step'),
                      hist2d_kwargs=dict(alpha=0.5))
    legend_elements.append(Patch(facecolor=posterior_color,
                                 edgecolor='k',
                                 alpha=1.0,
                                 label=posterior_label))
    fig.subplots_adjust(right=1.5, top=1.5)
    # fig.legend(handles=legend_elements, fontsize=20, loc=[-0.7, 2.5])
    for ax in fig.get_axes():
        ax.tick_params(axis='both', labelsize=20)
    return fig, legend_elements
