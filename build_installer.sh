micromamba activate vegvisir
pyi-makespec --paths vegvisir/src --additional-hooks-dir=vegvisir/src/hooks Vegvisir_GUI.py --add-data vegvisir/src/vegvisir/data/anchor_info_content:vegvisir/src/vegvisir/data/anchor_info_content --add-data vegvisir/src/vegvisir/data/common_files:vegvisir/src/vegvisir/data/common_files --add-data vegvisir/src/vegvisir/data/benchmark_datasets:vegvisir/src/vegvisir/data/benchmark_datasets
#--add-data '/home/dragon/micromamba/envs/vegvisir/lib/python3.9/site-packages/Bio/Align/substitution_matrices:dist/Vegvisir_GUI/_internal/Bio/Align/substitution_matrices' #re-writes the .spec file, so careful
pyinstaller Vegvisir_GUI.spec --log-level=DEBUG -y --clean #clean to overwrite the environment, otherwise it uses the cached one
./dist/Vegvisir_GUI/Vegvisir_GUI

#https://stackoverflow.com/questions/74125426/local-directory-while-using-pyinstaller

#Help: https://stackoverflow.com/questions/49085970/no-such-file-or-directory-error-using-pyinstaller-and-scrapy


#Hooks: https://github.com/pyinstaller/pyinstaller-hooks-contrib/tree/master/_pyinstaller_hooks_contrib