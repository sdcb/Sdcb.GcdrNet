# Downloads the RealDAE corpus (50 images + ground truths, ~36 MB) from
# https://github.com/sdcb/DocGainFixture into tests/testdata/realdae (git-ignored).
# Needed by CorpusTests (skipped without it) and the benchmark's --corpus / --smoke modes.
#   ./tools/fetch-fixture.ps1           # no-op when already present
#   ./tools/fetch-fixture.ps1 -Force    # re-download
param(
    [switch]$Force,
    [string]$Repo = 'https://github.com/sdcb/DocGainFixture.git',
    [string]$Ref = 'main'
)
$ErrorActionPreference = 'Stop'
$dst = [IO.Path]::GetFullPath([IO.Path]::Combine($PSScriptRoot, '..', 'tests', 'testdata', 'realdae'))
if ((Test-Path (Join-Path $dst 'manifest.json')) -and -not $Force) {
    Write-Output "fixture already present: $((Resolve-Path $dst).Path)"
    return
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("DocGainFixture-" + [Guid]::NewGuid().ToString('N'))
try {
    git clone --quiet --depth 1 --branch $Ref $Repo $tmp
    if ($LASTEXITCODE -ne 0) { throw "git clone $Repo failed: $LASTEXITCODE" }
    if (Test-Path $dst) { Remove-Item $dst -Recurse -Force }
    Move-Item (Join-Path $tmp 'realdae') $dst
    $n = (Get-ChildItem $dst -Filter *.webp).Count
    Write-Output "fixture: $n images -> $((Resolve-Path $dst).Path)"
}
finally {
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
}
