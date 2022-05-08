from rlcode.constants import project_dir
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RUNS = 10
STEPS = 20000

result_frame_save_path = project_dir.joinpath("results").joinpath(
    f"q2_11_result_{RUNS}_{STEPS}.csv"
)
result_frame = pd.read_csv(result_frame_save_path, header=0)
result_frame = (
    result_frame.groupby(["parameter", "parameter_value"]).mean().reset_index()
)
print(result_frame)

plot = sns.lineplot(data=result_frame, x="parameter_value", y="reward", hue="parameter")
plot.set(xscale="log")
plt.show()
