import numpy as np
import itertools
import matplotlib.pyplot as plt
import utils.consts as consts
import matplotlib
import pandas as pd
import os
import matplotlib.pyplot as plt
def plot_auroc(fpr, tpr, roc_auc):
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


def plot_grid(k_grid, auc_knn_all_train, auc_knn_all_val):
    plt.plot(k_grid, auc_knn_all_train, '-o', label="train", linewidth=2)
    plt.plot(k_grid, auc_knn_all_val, '-*b', label="val", linewidth=2)
    plt.xlabel('k', fontsize=14)
    plt.ylabel('roc_auc', fontsize=14)
    plt.legend()
    plt.show()


def plot_mean_std_metric(df_metrics, metric, lims, metric_name='', title_figure='mean_std_plot', flag_save_figure=False):

    df_metrics = df_metrics[df_metrics['metric'] == metric]

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))

    ax.plot(df_metrics['mean'], df_metrics['model'], ls='', marker='o', color='#8000FF')

    ax.hlines(df_metrics['model'], df_metrics['mean'] - df_metrics['std'], df_metrics['mean'] + df_metrics['std'],
              label='', lw=2, color='#8000FF', ls='-')

    ax.grid(axis='x', ls='-')
    ax.grid(axis='y', ls=':', lw=1, alpha=0.5)
    ax.set(
        xlabel=metric_name,
        xlim=lims,
        title=title_figure
    )

    fig.tight_layout()

    if flag_save_figure:
        fig.savefig(title_figure, dpi=300, bbox_inches='tight')
    else:
        plt.show()


def plot_confusion_matrix(cm, target_names, id, title='Confusion matrix', cmap=None, normalize=False, flag_save_figure=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    if flag_save_figure:
        plt.savefig(consts.PATH_PROJECT_FIGURES + title + '_' + str(id) + ".pdf")
    else:
        plt.show()


def plot_CGM(u):
    x = u.index[0:len(u)]

    l1 = [101.5] * len(u)
    l2 = [141.5] * len(u)
    l3 = [180] * len(u)
    l4 = [234] * len(u)

    fig = plt.figure(figsize=(20, 20))
    fig.clf()
    ax = fig.subplots(1, 1)
    ax.plot(x, u.values,'.-',ms=3)
    ax.fill_between(x, 39, 101.5,
                    color='lightcoral', alpha=0.2)
    ax.plot(x, l1, 'red')
    ax.fill_between(x, 101.5, 141.5,
                    color='darksalmon', alpha=0.25)
    ax.plot(x, l2, 'red')
    ax.fill_between(x, 141.5, 180,
                    color='lightyellow', alpha=0.4)
    ax.plot(x, l3, 'red')
    ax.fill_between(x, 180, 234,
                    color='lightgreen', alpha=0.2)
    ax.plot(x, l4, 'red')

    ax.fill_between(x, 234 ,401,
                    color='lightcyan', alpha=0.5)
    plt.xlabel('Date of the Measurement',fontname='serif', fontsize=40)
    plt.ylabel('Glucose level',fontname='serif', fontsize=40)
    plt.xticks(fontname='serif', fontsize=35)
    plt.yticks(fontname='serif', fontsize=35)
    plt.grid(axis='x', ls='-', lw=1, alpha=0.9)
    plt.grid(axis='y', ls=':', lw=1, alpha=0.9)
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.show()
def plot_heatmap_selected_features_fes(df_selected_features,name , flag_save_figure=True):
    df_selected_features.index = df_selected_features['names']
    # df_selected_features = df_selected_features.drop(columns=['std', 'score_sum', 'Unnamed: 0','names'])
    df_selected_features = df_selected_features.drop(columns=['names'])
    # df_selected_features.columns=['Pt.1','Pt.2','Pt.3','Pt.4','Pt.5']
    df_heatmap= df_selected_features
    # fig, ax = plt.subplots(1, 3, figsize=(8, 6))
    if len (df_heatmap) > 15:
        df_heatmap=df_heatmap[:15]
    # im, _ = plot_heatmap(df_heatmap.values, df_heatmap.index.values, df_heatmap.columns.values,
    #                   vmin=0.4, vmax=df_heatmap.values.max(), cmap="magma_r", cbarlabel="AUCROC")

    im, _ = plot_heatmap(df_heatmap.values, df_heatmap.index.values, df_heatmap.columns.values,
                         vmin=0.3, vmax=0.9, cmap="magma_r", cbarlabel="VALUE")
    # texts = annotate_heatmap(im, valfmt="{x:d}", size=11)
    # annotate_heatmap(im, valfmt="{x:d}", size=11, threshold=20)

    if flag_save_figure:
        plt.savefig(str(os.path.join(consts.PATH_PROJECT_REPORTS, 'heatmap_{}.png'.format(name))),
                    bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_heatmap(data, row_labels, col_labels, ax=None, cbar_kw=None, cbarlabel="", fontsize=11, **kwargs):

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom", fontsize=fontsize)

    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(fontsize)

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=fontsize)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=fontsize)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_xticklabels(), rotation=90)

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(True)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts
def plot_errorbar_metric(df,variables_plotted,errors,x_names,size,title,label_x,label_y,marker,color,path=''):

    fig = plt.figure(figsize=(size[0], size[1]))
    for j,e in enumerate(variables_plotted):
        y=df[e].astype(float).values
        err=df[errors[j]].astype(float).values
        x=x_names
        plt.errorbar(x, y, err,label=e,marker=marker[j], ms=8, fillstyle='none',color=color[j])
        plt.fill_between(x, y - err, y + err,
                         color=color[j], alpha=0.25)




    plt.grid(axis='x', ls='-', lw=3, alpha=0.9)
    plt.grid(axis='y', ls=':', lw=5, alpha=0.9)
    # plt.title(title,fontname='serif',fontsize = 35)
    plt.xticks(fontname='serif',fontsize = 25)
    plt.yticks(fontname='serif',fontsize=25)
    plt.ylim(0.6,0.9)
    plt.xlabel(label_x,fontname='serif',fontsize = 30)
    plt.ylabel(label_y,fontname='serif',fontsize = 30)
    # plt.legend(loc ='upper left')
    plt.tight_layout()
    fig.autofmt_xdate(rotation=45)

    if path != '':
        plt.savefig(path)
    plt.show()
def point_plot(df,names=['IA','IA-FS-PRE','IA-FS-POST'],colors=['seagreen','coral','royalblue'],label_x='SUBSETS',label_y='AUCROC',markers=['o','s','v']):

    fig = plt.figure(figsize=(25, 10))
    fig.clf()
    ax = fig.subplots(1, 1)
    offset=-0.15
    for e in range(len(names)):
        x = np.asarray(range(1,8))
        y=df[names[e]]
        err=df[names[e]+'_STD']
        ax.errorbar(x-offset, df[names[e]], df[names[e]+'_STD'], linestyle='None',color=colors[e], ms=15, marker=markers[e],capsize=20,label=names[e])
        offset= offset+0.15
    label=['S' + str(e) for e in x]
    plt.grid(axis='x', ls='-', lw=2, alpha=0.9)
    plt.grid(axis='y', ls=':', lw=2, alpha=0.9)
    # plt.title(title,fontname='serif',fontsize = 35)
    plt.xticks(ticks=x,labels=label,fontname='serif', fontsize=25)
    plt.yticks(fontname='serif', fontsize=25)
    plt.ylim(0.5, 0.9)
    plt.xlabel(label_x, fontname='serif', fontsize=30)
    plt.ylabel(label_y, fontname='serif', fontsize=30)
    plt.legend(fontsize="20",loc ='upper right',)
    plt.tight_layout()
    fig.autofmt_xdate(rotation=45)
    fig.tight_layout()
    fig.show()

def lollipop_plot (x,y,color,name):
    fig = plt.figure(figsize=(40, 20))
    plt.hlines(y=x, xmin = 0, xmax =y, color=color)
    plt.plot(y, x, "o",ms=15,color =color)
    plt.xlabel('Difference in the number of ocurrences in case vs control',fontname='serif', fontsize=30)
    plt.ylabel(name,fontname='serif', fontsize=30)
    plt.xticks(fontname='serif', fontsize=25)
    plt.yticks(fontname='serif', fontsize=25)
    plt.grid(axis='x', ls='-', lw=2, alpha=0.9)
    plt.grid(axis='y', ls=':', lw=2, alpha=0.9)
    fig.tight_layout()
    fig.show()
