<#
.SYNOPSIS
  Qwen3.8-27B + DFlash on an AMD Radeon AI PRO R9700 (Windows 11 + WSL2).

.EXAMPLE
  .\qwen38.ps1 install        # one-time: WSL Ubuntu, Docker, ROCDXG, image, models, kernels
  .\qwen38.ps1 start          # OpenAI-compatible server on http://localhost:8080/v1
  .\qwen38.ps1 start -Long    # leave 1.0 GiB for Windows instead of 1.5 -> ~260k context
  .\qwen38.ps1 bench          # prefill / decode / draft-acceptance benchmark
  .\qwen38.ps1 status | logs | stop
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('install', 'start', 'stop', 'status', 'logs', 'bench', 'help')]
    [string]$Command = 'help',
    [string]$Distro,
    [switch]$Long
)
# No $ErrorActionPreference = 'Stop': Windows PowerShell 5.1 turns harmless native stderr
# (e.g. "Can't initialize amdsmi" under WSL) into terminating errors. Exit codes decide instead.

if ($Command -eq 'help') { Get-Help $PSCommandPath -Examples; return }

if (-not $Distro) {
    # wsl.exe -l prints UTF-16; strip the NULs before matching.
    $installed = (wsl.exe -l -q 2>$null) -replace "`0", '' | Where-Object { $_ }
    $Distro = @('Ubuntu-24.04', 'Ubuntu') | Where-Object { $installed -contains $_ } | Select-Object -First 1
}
if (-not $Distro) {
    if ($Command -ne 'install') { throw 'No Ubuntu WSL distro found. Run: .\qwen38.ps1 install' }
    Write-Host 'Installing WSL2 + Ubuntu 24.04. Create the Linux user when asked, then run .\qwen38.ps1 install again.'
    wsl.exe --install -d Ubuntu-24.04
    return
}

$repo = (wsl.exe -d $Distro -u root -- wslpath -a ($PSScriptRoot -replace '\\', '/')).Trim()
$sh = "$repo/scripts/qwen38.sh"

switch ($Command) {
    'install' { wsl.exe -d $Distro -u root -- bash $sh setup }
    'start'   { if ($Long) { wsl.exe -d $Distro -u root -- bash $sh start --long } else { wsl.exe -d $Distro -u root -- bash $sh start } }
    'bench'   { wsl.exe -d $Distro -u root -- python3 "$repo/bench/bench.py" }
    default   { wsl.exe -d $Distro -u root -- bash $sh $Command }
}
if ($LASTEXITCODE) { exit $LASTEXITCODE }
