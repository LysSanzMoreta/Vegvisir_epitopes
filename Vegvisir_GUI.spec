# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Vegvisir_GUI.py'],
    pathex=['vegvisir/src'],
    binaries=[],
    datas=[('vegvisir/src/vegvisir/data/anchor_info_content', 'vegvisir/src/vegvisir/data/anchor_info_content'), ('vegvisir/src/vegvisir/data/common_files', 'vegvisir/src/vegvisir/data/common_files'), ('vegvisir/src/vegvisir/data/benchmark_datasets', 'vegvisir/src/vegvisir/data/benchmark_datasets')],
    hiddenimports=[],
    hookspath=['vegvisir/src/hooks'],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Vegvisir_GUI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Vegvisir_GUI',
)
