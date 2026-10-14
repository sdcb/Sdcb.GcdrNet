# One CI benchmark replica: builds the benchmark, heats the VM with a discarded short run, then
# runs every case on this runner -> results/<platform>-<id>-r<replica>.json.
#   -Cases 'id,backend,ENV=VALUE;id,backend,'   (ENV part optional, ';'-separated cases)
param(
    [Parameter(Mandatory)] [string]$Platform,
    [Parameter(Mandatory)] [string]$Cases,
    [int]$Replica = 1,
    [int]$Count = 20
)
$ErrorActionPreference = 'Stop'
$root = (Resolve-Path (Join-Path $PSScriptRoot '..' '..')).Path
Set-Location $root
New-Item -ItemType Directory -Force results | Out-Null

dotnet build bench/Sdcb.SimdGcdrNet.Benchmark -c Release -o artifacts/bench
if ($LASTEXITCODE -ne 0) { throw "benchmark build failed: $LASTEXITCODE" }
$dll = 'artifacts/bench/Sdcb.SimdGcdrNet.Benchmark.dll'

# A cold VM biases the first process: discard a short run first.
Write-Output '::group::VM warmup (discarded)'
dotnet $dll --ci bench --backend cpu --count 3 --case-id vm-warmup --out (Join-Path ([IO.Path]::GetTempPath()) 'vm-warmup.json')
Write-Output '::endgroup::'
if ($LASTEXITCODE -ne 0) { throw "VM warmup failed: $LASTEXITCODE" }

$failed = @()
foreach ($case in $Cases.Split(';', [StringSplitOptions]::RemoveEmptyEntries)) {
    $id, $backend, $envs = $case.Split(',', 3)
    $caseId = "$Platform-$id"
    $saved = @{}
    foreach ($kv in "$envs".Split(' ', [StringSplitOptions]::RemoveEmptyEntries)) {
        $k, $v = $kv.Split('=', 2)
        $saved[$k] = [Environment]::GetEnvironmentVariable($k)
        [Environment]::SetEnvironmentVariable($k, $v)
    }
    Write-Output "::group::$caseId (replica $Replica) $envs"
    try {
        dotnet $dll --ci bench --backend $backend --count $Count --case-id $caseId --replica $Replica --out "results/$caseId-r$Replica.json"
        if ($LASTEXITCODE -ne 0) { $failed += $caseId }
    }
    finally {
        foreach ($k in $saved.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) }
        Write-Output '::endgroup::'
    }
}
if ($failed) { throw "benchmark failed: $($failed -join ', ')" }
