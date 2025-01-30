import json
import os

from tqdm.notebook import tqdm


def separate_meta_files(base_path):
    """Create a meta.json file for all experiments in the base_path directory."""
    experiments = sorted([e for e in os.listdir(base_path)
                          if os.path.isdir(f"{base_path}/{e}")])

    for e in tqdm(experiments):
        files = os.listdir(f"{base_path}/{e}")

        if not os.path.exists(f"{base_path}/{e}/meta.json"):
            first_meta = None
            for i in files:
                if not i.endswith(".json"):
                    continue
                else:
                    with open(f"{base_path}/{e}/{i}") as f:
                        first = json.load(f)
                    first_meta = first["meta"]
                    first_meta["time_start"] = first["time_start"]
                    first_meta["time_end"] = first["time_end"]
                    break

            if first_meta is None:
                print(f"Experiment {e} has no files.")
                continue
            else:
                print(f"Creating {e}/meta.json.")
                with open(f"{base_path}/{e}/meta.json", "w") as f:
                    json.dump(fp = f, obj = first_meta)



def load_experiments(base_path, slc=None,
                     filter_on_meta=None, filter_on_instance=None,
                     map=None, reduce_instances=None, reduce_experiments=None,
                     tqdm_internal = True):
    """Load experimental data from the base_path directory, allowing to filter on meta.json (experiment properties), as well as filter individual instances, map them, reduce instances of an experiment, and reduce all experiments."""

    experiments = sorted([e for e in os.listdir(base_path)
                          if os.path.isdir(f"{base_path}/{e}")])
    data = {}

    if slc is None:
        slc = slice(None, None, None)

    for e in tqdm(experiments[slc]):
        instances = []
        files = os.listdir(f"{base_path}/{e}")

        try:
            with open(f"{base_path}/{e}/meta.json") as f:
                meta = json.load(f)
        except FileNotFoundError:
            print(f"Meta file {base_path}/{e}/meta.json does not exist. Skipping experiment {e}. Run `separate_meta_files({base_path})`.")
            continue
        except json.JSONDecodeError:
            print(f"Cannot JSON-decode {base_path}/{e}/meta.json.")


        if filter_on_meta is not None and not filter_on_meta(meta):
            continue


        for i in tqdm(files, leave=False, disable=not tqdm_internal):
            if not i.endswith(".json"):
                print(f"Unexpected file {base_path}/{e}/{i}. Skipping.")
                continue

            if i == "meta.json":
                continue

            try:
                with open(f"{base_path}/{e}/{i}") as f:
                    inst = json.load(f)
            except json.JSONDecodeError as err:
                print(f"Error when loading {e}/{i}. Skipping. Error follows:")
                print(err)
                continue

            if filter_on_instance is not None and not filter_on_instance(inst):
                continue
            else:
                if map is not None:
                    instances.append(map(inst))
                else:
                    instances.append(inst)


        if reduce_instances is not None:
            data[e] = reduce_instances(instances)
        else:
            data[e] = instances

    print(f"Loaded {len(data)} out of {len(experiments)} available experiments.")

    if reduce_experiments is not None:
        return reduce_experiments(data)
    else:
        return data
