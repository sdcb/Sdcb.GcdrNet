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

& $run 'default (AVX2/AVX-512)' @{}
& $run 'vector (GCDRNET_ISA=vector)' @{ GCDRNET_ISA = 'vector' }
& $run 'scalar (GCDRNET_ISA=scalar)' @{ GCDRNET_ISA = 'scalar' }
& $run 'no-AVX (DOTNET_EnableAVX=0)' @{ DOTNET_EnableAVX = '0' }
& $run 'no-AVX512 (DOTNET_EnableAVX512=0)' @{ DOTNET_EnableAVX512 = '0' }

Write-Host 'All ISA tiers passed.' -ForegroundColor Green
