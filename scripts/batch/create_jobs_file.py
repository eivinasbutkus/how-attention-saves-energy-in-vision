
from itertools import product
import hydra
from hydra.core.hydra_config import HydraConfig

import what_where as ww


@hydra.main(version_base=None, config_name='config_vcs_flexible', config_path=str(ww.utils.CONFIG_DIR))
def create_vcs_jobs_file(cfg):
    config_name = HydraConfig.get().job.config_name
    experiment_name = cfg.experiment.name

    instances = list(range(cfg.experiment.n_instances))

    if cfg.experiment.name == "journal_fixed":
        energy_costs = ww.utils.get_energy_costs(cfg)
    else:
        energy_costs = [None]  # in flexible experiments, energy cost is not fixed

    models = cfg.analysis.models

    num_workers = cfg.train.dataloader.num_workers

    jobs_file = ww.utils.ROOT_DIR / "scripts" / "batch" / f"jobs_{experiment_name}.txt"
    jobs_file.parent.mkdir(parents=True, exist_ok=True)

    combinations = list(product(instances, models, energy_costs))

    with open(jobs_file, "w") as f:
        for instance, model, energy_cost in combinations:
            cmd = "python scripts/train.py"

            cmd += f" --config-name={config_name}"

            cmd += f" model={model}"
            cmd += f" train.instance={instance}"
            cmd += f" train.dataloader.num_workers={num_workers}"

            if energy_cost is not None:
                # fixing the energy cost
                cmd += f" train.energy.cost.min={energy_cost}"
                cmd += f" train.energy.cost.max={energy_cost}"

            cmd += f" seed={instance}"
            cmd += f";"

            print(cmd)
            f.write(cmd + "\n")

    print(f"{experiment_name} jobs: ", len(combinations))



if __name__ == "__main__":
    create_vcs_jobs_file()