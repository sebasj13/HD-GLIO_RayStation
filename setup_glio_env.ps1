# setup_glio_env.ps1 - build the conda environment for the HD-GLIO pipeline.
#
# Creates the conda env "glio" (python 3.10) with:
#   - PyTorch 2.4.1 (CUDA 12.4 wheel; needs an NVIDIA driver supporting CUDA 12.4).
#     IMPORTANT: installed BEFORE hd-bet, because hd-bet pulls torchvision
#     whose PyPI wheel pins the newest torch and would replace the CUDA
#     build with a CPU-only one.
#   - hd_glio      (HD-GLIO from the third_party/HD-GLIO git submodule;
#                   pulls nnU-Net v1 1.7.1)
#   - hd-bet 2.x   (brain extraction; pulls nnunetv2 - which requires
#                   numpy>=1.24, so numpy is NOT pinned here; verified that
#                   nnunet 1.7.1/batchgenerators/medpy contain no removed
#                   numpy aliases, so numpy 2.x works for the whole stack)
#   - pydicom, SimpleITK, nibabel, scikit-image, matplotlib (DICOM tooling)
#
# Model weights are NOT auto-downloaded here: behind a TLS-intercepting proxy
# the Python (certifi) downloads fail with SSLError. See README "Model weights"
# for the curl.exe --ssl-no-revoke workaround:
#   HD-GLIO -> %USERPROFILE%\hd_glio_params   (zenodo 4014850)
#   HD-BET  -> %USERPROFILE%\hd-bet_params\release_2.0.0  (zenodo 14445620)
#
# Run:  powershell -ExecutionPolicy Bypass -File .\setup_glio_env.ps1

$ErrorActionPreference = "Continue"

# The nnU-Net v1 sdist fails to build a wheel under the default %TEMP% because
# pip's build path + nnunet's deep module tree exceed the Windows MAX_PATH
# limit (260 chars). Point pip's temp dir to a short path instead.
$shortTmp = "C:\tmp"
New-Item -ItemType Directory -Force $shortTmp | Out-Null
$env:TEMP = $shortTmp
$env:TMP = $shortTmp
$env:TMPDIR = $shortTmp

Write-Host "=== [1/7] Creating conda env 'glio' (python 3.10) ==="
$envExists = $false
conda env list | Out-String -Stream | ForEach-Object { if ($_ -match "^\s*glio\s") { $envExists = $true } }
if ($envExists) {
    Write-Host "env 'glio' already exists - skipping creation"
} else {
    conda create -n glio python=3.10 -y
    if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: conda create failed"; exit 1 }
}

Write-Host "=== [2/7] Installing PyTorch + torchvision (CUDA 12.4 wheels) ==="
# torchvision must be the matching +cu124 build installed from the pytorch
# index: nnunetv2 (pulled in by hd-bet) imports timm, timm imports
# torchvision.ops.misc. A later PyPI torchvision would drag a newer CPU-only
# torch over the CUDA build - the exact pin prevents that.
conda run --live-stream -n glio python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: torch install failed"; exit 1 }

Write-Host "=== [3/7] Installing hd_glio (HD-GLIO submodule + nnU-Net v1) ==="
$hdglio = Join-Path $PSScriptRoot "third_party\HD-GLIO"
if (-not (Test-Path (Join-Path $hdglio "setup.py"))) {
    git -C $PSScriptRoot submodule update --init --recursive
    if (-not (Test-Path (Join-Path $hdglio "setup.py"))) {
        Write-Host "FATAL: third_party\HD-GLIO is empty - clone with --recursive or run 'git submodule update --init'"
        exit 1
    }
}
conda run --live-stream -n glio python -m pip install "$hdglio"
if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: hd_glio install failed"; exit 1 }

Write-Host "=== [4/7] Installing hd-bet (brain extraction, HD-BET 2.x) ==="
conda run --live-stream -n glio python -m pip install hd-bet
if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: hd-bet install failed"; exit 1 }

Write-Host "=== [5/7] Installing DICOM tooling ==="
conda run --live-stream -n glio python -m pip install pydicom SimpleITK nibabel scikit-image matplotlib
if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: tooling install failed"; exit 1 }

Write-Host "=== [6/7] Verification ==="
conda run --live-stream -n glio python -c "import torch; print('torch', torch.__version__, 'cuda:', torch.cuda.is_available())"
conda run --live-stream -n glio python -c "import numpy, nnunet, hd_glio, HD_BET, pydicom, SimpleITK, nibabel, skimage, matplotlib; from nnunet.inference.predict import predict_cases; print('nnunet', nnunet.__version__, '| numpy', numpy.__version__, '| SimpleITK', SimpleITK.__version__)"
if ($LASTEXITCODE -ne 0) { Write-Host "FATAL: verification failed"; exit 1 }

Write-Host ""
Write-Host "Environment ready. Next steps:"
Write-Host "  1. Model weights: if the Python download fails behind a proxy, use curl:"
Write-Host "      curl.exe -L --ssl-no-revoke -o `"$env:TEMP\hd_glio_v2_params.zip`" `"https://zenodo.org/record/4014850/files/hd_glio_v2_params.zip?download=1`""
Write-Host "      curl.exe -L --ssl-no-revoke -o `"$env:TEMP\release_v1.5.0.zip`" `"https://zenodo.org/records/14445620/files/release_v1.5.0.zip?download=1`""
Write-Host "     Extract the zip contents into:"
Write-Host "       %USERPROFILE%\hd_glio_params\               (fold_0, plans.pkl, version=2)"
Write-Host "       %USERPROFILE%\hd-bet_params\release_2.0.0\  (fold_all, plans.json, ...)"
Write-Host "  2. Site config: copy config.example.json to config.json and set job_root"
Write-Host "  3. Start the watcher:  powershell -ExecutionPolicy Bypass -File .\start_watcher.ps1"
