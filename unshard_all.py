import yaml
import os
import subprocess
import tempfile

# Path to your base YAML file
TEMPLATE_YAML = "configs/unshard.yml"

# Iterations to run unshard on
# ITERATIONS = [10000, 20000, 40000, 60000, 80000]
# ITERATIONS = list(range(5000, 75001, 5000))
# ITERATIONS = list(range(50000, 75001, 5000))

# ITERATIONS = [75000]

ITERATIONS = list(range(5000, 25001, 5000))

with open(TEMPLATE_YAML, "r") as f:
    base_config = yaml.safe_load(f)

for it in ITERATIONS:
    config = base_config.copy()
    # update iteration
    config["load_args"]["iteration"] = it

    # update unsharded_path
    unsharded_base = base_config["unsharded_path"]
    unsharded_dir = os.path.dirname(unsharded_base)
    filename = f"diff_causal_weight_v1__{it}"
    config["unsharded_path"] = os.path.join(unsharded_dir, filename)

    # create a temporary yaml file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as tmp:
        yaml.dump(config, tmp, sort_keys=False)
        tmp_path = tmp.name

    print(f"▶️ Running unshard for iteration {it}...")
    subprocess.run(["bash", "scripts/common/unshard.sh", tmp_path], check=True)

    # remove temp file if you don’t want to keep it
    os.remove(tmp_path)

print("✅ All iterations processed!")
