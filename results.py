import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_rows", None)


df0 = pd.read_csv(
    "tuning_results/general_augment/metrics_general_augment.csv",
    index_col=[0],
)
# df1 = pd.read_csv("tuning_results/focus1/metrics_focus1.csv", index_col=[0])
# df2 = pd.read_csv("tuning_results/focus2/metrics_focus2.csv", index_col=[0])
# df3 = pd.read_csv("tuning_results/focus3/metrics_focus3.csv", index_col=[0])
# df4 = pd.read_csv("tuning_results/focus4/metrics_focus4.csv", index_col=[0])
# df5 = pd.read_csv("tuning_results/focus5/metrics_focus5.csv", index_col=[0])

# concatenated_df = pd.concat([df0, df1, df2, df3])
concatenated_df = pd.concat([df0])
concatenated_df.set_index("Threshold", inplace=True)
concatenated_df.sort_index(inplace=True)
df = concatenated_df.loc[~concatenated_df.index.duplicated(keep="first")]
print(df)


plt.figure(figsize=(10, 6))
plt.plot(df.index, df["F1"], color="black", label="F1")
plt.plot(df.index, df["Precision"], color="green", label="Precision")
plt.plot(df.index, df["Recall"], color="red", label="Recall")
plt.xlabel("Threshold")
plt.ylabel("Value")
plt.title("F1, Precision, and Recall by Threshold")
plt.legend()
plt.grid(True)
plt.savefig("tuning_plot_augment.png")  # Save the plot as a PNG image
plt.close()  # Close the plotting window to free up memory
