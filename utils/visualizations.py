import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def plot_confusion_matrix(actual_values, predicted_values, act_val_axis_label, pred_val_axis_label, save_to, map_title = None, show_grid = True, unique_id_count = None):
    confusion_matrix = metrics.confusion_matrix(actual_values, predicted_values)

    plt.xlabel(act_val_axis_label)
    plt.ylabel(pred_val_axis_label)

    plt.imshow(confusion_matrix, interpolation='none')
    plt.colorbar()
    
    plt.title(map_title)
    plt.grid(show_grid)

    if unique_id_count is not None:
        ticks=np.linspace(0, unique_id_count - 1,num=unique_id_count)
        plt.xticks(ticks,fontsize=5)
        plt.yticks(ticks,fontsize=5)

    plt.savefig(save_to)
    plt.close()
