from PyInstaller.utils.hooks import collect_submodules, collect_data_files

# collect all submodules from scipy
hiddenimports = collect_submodules('scipy')

# collect scipy data files
datas = collect_data_files('scipy')