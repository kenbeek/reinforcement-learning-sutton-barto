import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from rlcode.constants import output_dir, project_dir

RUNS = 10
STEPS = 200000

result_frame_save_path = project_dir.joinpath("results").joinpath(
    f"q2_11_result_{RUNS}_{STEPS}.csv"
)
result_frame = pd.read_csv(result_frame_save_path, header=0)
result_frame = (
    result_frame.groupby(["parameter", "parameter_value"]).mean().reset_index()
)
print(result_frame)

fig, ax = plt.subplots(figsize=(10, 7))
sns.lineplot(data=result_frame, x="parameter_value", y="reward", hue="parameter")
plt.xscale("log")
plt.legend()
plt.title(
    f"Parameter study for different types of agents. {RUNS} runs over {STEPS} steps."
)
plt.show()


save_file = output_dir.joinpath(f"parameter_study{RUNS}_{STEPS}.png")
fig.savefig(save_file)
