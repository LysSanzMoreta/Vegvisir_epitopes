from PyInstaller.utils.hooks import collect_submodules, collect_data_files


# Collect all submodules from ray and ray.tune
hiddenimports = collect_submodules('ray') + collect_submodules('ray.tune')

# Optionally collect any data files that Ray Tune needs at runtime
datas = collect_data_files('ray',include_py_files=True) + collect_data_files('ray.tune',include_py_files=True)