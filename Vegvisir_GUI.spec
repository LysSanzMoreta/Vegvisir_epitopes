# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['Vegvisir_GUI.py'],
    pathex=[],
    binaries=[],
    datas=[('vegvisir/src','vegvisir/src')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

mpl_font_path = os.path.join('vegvisir', 'src', 'vegvisir','data')
a.datas = [entry for entry in a.datas if not entry[0].startswith(mpl_font_path)]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Vegvisir_GUI.exe',
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
