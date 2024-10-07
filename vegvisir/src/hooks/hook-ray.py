from PyInstaller.utils.hooks import collect_submodules, collect_data_files


# Collect all submodules from ray
hiddenimports = collect_submodules('ray') + ['setproctitle']

# collect data files
datas = collect_data_files('ray',include_py_files=True)