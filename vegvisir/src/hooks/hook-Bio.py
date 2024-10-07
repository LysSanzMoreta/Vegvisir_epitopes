from PyInstaller.utils.hooks import collect_submodules, collect_data_files
import os

# Collect additional data files (if any)
#datas = collect_data_files(package="biopython",subdir='Bio.Align.substitution_matrices.data', includes=['*.txt', '*.csv','*'])
# datas = collect_data_files('Bio.Align.substitution_matrices.data', includes=['*.txt', '*.csv','*'])
#
# data_dir = os.path.join(os.path.dirname(__file__), '_internal/Bio/Align/substitution_matrices/data')
# if os.path.exists(data_dir):
#     datas += [(data_dir, 'Bio/Align/substitution_matrices/data')]
#

# collect biopython submodules
hiddenimports = collect_submodules('Bio')

# collect data files
datas = collect_data_files('Bio')