import seaborn as sns
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)
import matplotlib.pyplot as plt
import numpy as np


class MetricsCalculator:
    def __init__(self, labels, predictions, probabilities, detection_counts):
        self.labels = labels
        self.predictions = predictions
        self.probabilities = probabilities
        self.detection_counts = detection_counts

    def compute_precision(self):
        return precision_score(self.labels, self.predictions)

    def compute_recall(self):
        return recall_score(self.labels, self.predictions)

    def compute_accuracy(self):
        return accuracy_score(self.labels, self.predictions)

    def compute_f1(self):
        return f1_score(self.labels, self.predictions)

    def compute_auprc(self):
        precision, recall, _ = precision_recall_curve(self.labels, self.probabilities)
        return auc(recall, precision)

    def compute_confusion_matrix(self):
        return confusion_matrix(self.labels, self.predictions)

    def compute_mcc(self):
        return matthews_corrcoef(self.labels, self.predictions)

    def plot_true_positives_false_positives(self, ax):
        labels_array = np.array(self.labels)
        predictions_array = np.array(self.predictions)
        probabilities_array = np.array(self.probabilities)
        # Masks for all categories
        mask_tp = (predictions_array == 1) & (labels_array == 1)
        mask_fp = (predictions_array == 1) & (labels_array == 0)
        # Probabilities for each category
        probs_tp = probabilities_array[mask_tp]
        probs_fp = probabilities_array[mask_fp]

        # Plotting
        sns.histplot(
            probs_tp,
            kde=True,
            color="darkred",
            label="True Positives",
            ax=ax,
            stat="density",
            bins=20,
            alpha=0.7,
        )

        sns.histplot(
            probs_fp,
            kde=True,
            color="salmon",
            label="False Positives",
            ax=ax,
            stat="density",
            bins=20,
            alpha=0.7,
        )

        ax.set_xlabel("Predicted Probabilities")
        ax.set_ylabel("Density")
        ax.set_title("True/False Positive Prob. Distribution")
        ax.legend()
        ax.set_yscale("log")

    def plot_true_negatives_false_negatives(self, ax):
        labels_array = np.array(self.labels)
        predictions_array = np.array(self.predictions)
        probabilities_array = np.array(self.probabilities)

        # Masks for all categories
        mask_tn = (predictions_array == 0) & (labels_array == 0)
        mask_fn = (predictions_array == 0) & (labels_array == 1)

        # Probabilities for each category
        probs_tn = probabilities_array[mask_tn]
        probs_fn = probabilities_array[mask_fn]

        # Exclude fictitious probability values (1.0) for false and true negatives
        probs_fn = probs_fn[probs_fn != 1.0]
        probs_tn = probs_tn[probs_tn != 1.0]

        # Plotting
        sns.histplot(
            probs_tn,
            kde=True,
            color="darkgreen",
            label="True Negatives",
            ax=ax,
            stat="density",
            bins=20,
            alpha=0.7,
        )

        sns.histplot(
            probs_fn,
            kde=True,
            color="lightgreen",
            label="False Negatives",
            ax=ax,
            stat="density",
            bins=20,
            alpha=0.7,
        )

        ax.set_xlabel("Predicted Probabilities")
        ax.set_ylabel("Density")
        ax.set_title("True/False Negative Prob. Distribution")
        ax.legend()
        ax.set_yscale("log")

    def prepare_plots(self):
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))

        # Plot FP and FN distributions using the dedicated method
        self.plot_true_positives_false_positives(axs[0,0])
        self.plot_true_negatives_false_negatives(axs[1, 0])

        common_x_min = min(self.probabilities)
        common_x_max = max(self.probabilities)
        # Align the x-axis for the two plots
        axs[0,0].set_xlim(common_x_min, common_x_max)
        axs[1,0].set_xlim(common_x_min, common_x_max)

        # Filter out probabilities equal to 1
        prob_filtered = [prob for prob in self.probabilities if prob != 1]
        # Calculate the histogram of filtered probabilities to get frequencies, use this data to find max frequency
        freq_filtered, _ = np.histogram(prob_filtered, bins=50)
        max_freq_filtered = np.max(freq_filtered)
        buffer_margin = max(10, int(0.1 * max_freq_filtered))
        y_axis_cap = max_freq_filtered + buffer_margin

        # axs[0, 1].hist(
        #     self.probabilities,
        #     bins=50,
        #     label="Probabilities Distribution",
        #     alpha=0.6,
        #     color="b",
        # )
        # axs[0, 1].set_title("Probability Distribution")
        # axs[0, 1].set_xlabel("Probability score")
        # axs[0, 1].set_ylabel("Frequency")
        # axs[0, 1].legend(loc="best")
        # axs[0, 1].set_ylim(0, y_axis_cap)

        # Detection Count Distribution
        axs[0, 1].hist(
            self.detection_counts,
            bins=len(set(self.detection_counts)),
            label="Detection Counts Distribution",
            alpha=0.6,
            color="g",
        )
        axs[0, 1].set_title("Detection Count Distribution")
        axs[0, 1].set_xlabel("Detection Count")
        axs[0, 1].set_ylabel("Frequency")
        axs[0, 1].legend(loc="best")

        # Confusion Matrix
        cm = confusion_matrix(self.labels, self.predictions)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Predicted 0", "Predicted 1"],
            yticklabels=["True 0", "True 1"],
            ax=axs[1, 1],
        )
        axs[1, 1].set_xlabel("Predicted labels")
        axs[1, 1].set_ylabel("True labels")
        axs[1, 1].set_title("Confusion Matrix")

        plt.tight_layout()
        return fig

    def save_and_show_plots(self, filename, show=False):
        fig = self.prepare_plots()
        fig.savefig(filename)
        if show:
            fig.show()
