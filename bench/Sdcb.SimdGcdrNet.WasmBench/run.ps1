# One command for the browser: [dotnet publish (LLVM AOT)] -> server.mjs -> headless Edge/Chrome ->
# relay the console -> save the JSON result under artifacts/ -> exit with the bench's code.
# Needs `dotnet workload install wasm-tools-net10` and Node.js.
#
#   .\run.ps1                                         # --corpus cpu, all 49 timed images
#   .\run.ps1 -Bench "--corpus cpu" -Count 10 -NoBuild
#   .\run.ps1 -Bench "--bench 1152 1600 3" -Env GCDRNET_PROF=1
#   .\run.ps1 -Bench "--corpus cpu --vector"          # Vector<float> tier, PackedSimd off (A/B)
#   .\run.ps1 -Bench "--tiers 384 512"                # PackedSimd vs Vector<float> in one process
#   .\run.ps1 -Bench "--check small"                  # PyTorch golden
#   .\run.ps1 -Env GCDRNET_THREADS=4 -Open            # visible browser window
param(
    [string]$Bench = "--corpus cpu",
    [int]$Count = 0,
    [string]$Out = "",
    # runtime environment for .NET, e.g. -Env GCDRNET_THREADS=8,GCDRNET_PROF=1
    [string[]]$Env = @(),
    [switch]$Open,
    [switch]$NoBuild,
    [switch]$NoAot
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot

if (-not (Get-Command node -ErrorAction SilentlyContinue)) { throw "node not found on PATH (Node.js 18+ is required)" }

if (-not $NoBuild) {
    $aot = if ($NoAot) { "false" } else { "true" }
    Write-Host "[WasmBench] dotnet publish (RunAOTCompilation=$aot)..." -ForegroundColor Cyan
    dotnet publish $here -c Release --nologo -v:q "-p:RunAOTCompilation=$aot" | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "publish failed" }
}

$argv = $Bench
if ($Count -gt 0) { $argv += " --count $Count" }
if (-not $Out) {
    $tag = ($Bench -replace '^--', '' -replace '[^\w]+', '-').Trim('-')
    $Out = "wasm/$tag-$(Get-Date -Format 'MMdd-HHmmss').json"
}
$nodeArgs = @("$here\server.mjs", "--args", $argv, "--out", $Out)
if ($Open) { $nodeArgs += "--open" }
foreach ($pair in ($Env -join ',').Split(',', [StringSplitOptions]::RemoveEmptyEntries)) { $nodeArgs += @("--env", $pair.Trim()) }

& node @nodeArgs
exit $LASTEXITCODE
