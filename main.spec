# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=["helperFunctions", "."],
    binaries=[],
    datas=[
        ("icons", "icons"),        
        ("c:/vips-dev-8.16/bin", "vips"),                
        ("E:/Alienware_March 22/current work/00-new code May_22/dimReduction_v2/chemSimplifier3/Lib/site-packages/llvmlite/*", "llvmlite"),        
        ("E:/Alienware_March 22/current work/00-new code May_22/dimReduction_v2/chemSimplifier3/Lib/site-packages/llvmlite/binding/*", "llvmlite/binding"),
        ],
    hiddenimports=['numba.core.runtime', 'numba.core.registry'],
    hookspath=[],
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
    name='Chemistry Simplifier v1',
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
    icon="icons/colour_wheel.ico"
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Chemistry Simplifier v1',
)
