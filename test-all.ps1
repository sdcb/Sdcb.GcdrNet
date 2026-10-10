# Runs the test suite across the ISA matrix. Usage: ./test-all.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$proj = 'tests/Sdcb.GcdrNet.Tests'
$run = { param($name, $env)
    Write-Host "=== $name ===" -ForegroundColor Cyan
    $env.GetEnumerator() | ForEach-Object { Set-Item "env:$($_.Key)" $_.Value }
    try { dotnet test $proj -c Release --nologo }
    finally { $env.GetEnumerator() | ForEach-Object { Remove-Item "env:$($_.Key)" -EA SilentlyContinue } }
    if ($LASTEXITCODE -ne 0) { Write-Error "$name failed"; exit 1 }
}

# AVX-512 -> AVX2 -> Vector -> Scalar. On machines without AVX-512 the first two runs
# are both the AVX2 tier (the AVX-512 kernel tests skip).
& $run 'default (AVX-512 if supported, else AVX2)' @{}
& $run 'no-AVX512 / AVX2 (DOTNET_EnableAVX512=0)' @{ DOTNET_EnableAVX512 = '0' }
& $run 'no-AVX512F / AVX2 (DOTNET_EnableAVX512F=0)' @{ DOTNET_EnableAVX512F = '0' }
& $run 'vector-128 (DOTNET_EnableAVX2=0)' @{ DOTNET_EnableAVX2 = '0' }
& $run 'sse-no-fma (DOTNET_EnableAVX=0)' @{ DOTNET_EnableAVX = '0' }
& $run 'scalar (DOTNET_EnableHWIntrinsic=0)' @{ DOTNET_EnableHWIntrinsic = '0' }

Write-Host 'All ISA tiers passed.' -ForegroundColor Green
