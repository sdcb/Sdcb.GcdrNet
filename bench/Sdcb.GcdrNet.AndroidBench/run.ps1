# One command for the phone: [build] -> [adb install] -> [push testdata] -> launch ->
# wait for <run>.done -> pull files/out/* into artifacts/android/. If the process dies
# without writing .done, logcat is saved next to the log.
#
#   .\run.ps1 -Build -Install -Push -Bench "--bench 1152 1600 5"
#   .\run.ps1 -Bench "--corpus vulkan"
#   .\run.ps1 -Build -Install -GcdrEnv "DOTNET_EnableHWIntrinsic=0" -Bench "--bench 512 512 1"
param(
    [string]$Bench = "--bench",
    [string]$Env = "",
    [string]$GcdrEnv = "",
    [string]$Run = "",
    [switch]$Build,
    [switch]$Install,
    [switch]$Push,
    [int]$TimeoutSec = 7200,
    [string]$Adb = $(if ($env:ADB) { $env:ADB } elseif ($env:ANDROID_HOME) { "$env:ANDROID_HOME\platform-tools\adb.exe" } else { "adb" }),
    [string]$AndroidSdk = $env:ANDROID_HOME,
    [string]$Jdk = $env:JAVA_HOME
)
$ErrorActionPreference = "Stop"
$here = $PSScriptRoot
$root = (Resolve-Path "$here\..\..").Path
$pkg = "com.sdcb.gcdrnet.bench"
$remote = "/sdcard/Android/data/$pkg/files"
$local = "$root\artifacts\android"
New-Item -ItemType Directory -Force $local | Out-Null
if (-not $Run) { $Run = "r" + (Get-Date -Format "MMdd-HHmmss") }

if ($Build) {
    Push-Location $here
    try {
        $props = @("-p:GcdrEnv=$($GcdrEnv.Replace(';', '%3B'))")
        if ($AndroidSdk) { $props += "-p:AndroidSdkDirectory=$AndroidSdk" }
        if ($Jdk) { $props += "-p:JavaSdkDirectory=$Jdk" }
        dotnet build -c Release @props -v:q -nologo -clp:ErrorsOnly | Out-Host
        if ($LASTEXITCODE -ne 0) { throw "build failed" }
    } finally { Pop-Location }
}
if ($Install) {
    $apk = "$here\bin\Release\net10.0-android\android-arm64\$pkg-Signed.apk"
    & $Adb install -r -g $apk | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "adb install failed" }
}

function Start-Bench([string]$id, [string]$argv, [string]$envs) {
    # a previous run may still be exiting: an intent delivered to it would be
    # swallowed by the dying activity instead of starting a fresh process
    & $Adb shell "am force-stop $pkg"
    for ($i = 0; $i -lt 20 -and ((& $Adb shell "pidof $pkg") -join ""); $i++) { Start-Sleep -Milliseconds 500 }
    & $Adb shell "rm -f $remote/out/$id.done $remote/out/$id.log"
    $cmd = "am start -S -n $pkg/$pkg.MainActivity --es run '$id' --es args '$argv'"
    if ($envs) { $cmd += " --es env '$envs'" }
    & $Adb shell $cmd | Out-Null
}

if ($Push) {
    # the app must create testdata/ out/ itself: directories made by the shell
    # user are not readable by the app
    & $Adb shell "rm -rf $remote/testdata $remote/out"
    Start-Bench "init" "--init" ""
    for ($i = 0; $i -lt 30 -and -not ((& $Adb shell "cat $remote/out/init.done 2>/dev/null") -join ""); $i++) { Start-Sleep 1 }
    & $Adb push "$root\tests\testdata\." "$remote/testdata/" | Out-Host
}

& $Adb logcat -c
Start-Bench $Run $Bench $Env
Write-Output "[$Run] $Bench $(if ($Env) { "(env $Env)" })"

$t0 = Get-Date
$seen = 0
$started = $false
$result = $null
while (((Get-Date) - $t0).TotalSeconds -lt $TimeoutSec) {
    Start-Sleep -Seconds 2
    $done = (& $Adb shell "cat $remote/out/$Run.done 2>/dev/null") -join ""
    $log = & $Adb shell "cat $remote/out/$Run.log 2>/dev/null"
    if ($log) {
        $lines = @($log)
        for ($i = $seen; $i -lt $lines.Count; $i++) { Write-Output $lines[$i] }
        $seen = $lines.Count
    }
    if ($done -match "^\d+$") { $result = [int]$done; break }
    $procId = (& $Adb shell "pidof $pkg") -join ""
    if ($procId) { $started = $true }
    elseif ($started -or ((Get-Date) - $t0).TotalSeconds -gt 30) {
        # the app writes .done and then kills itself: look once more
        Start-Sleep -Seconds 1
        $done = (& $Adb shell "cat $remote/out/$Run.done 2>/dev/null") -join ""
        $lines = @(& $Adb shell "cat $remote/out/$Run.log 2>/dev/null")
        for ($i = $seen; $i -lt $lines.Count; $i++) { Write-Output $lines[$i] }
        if ($done -match "^\d+$") { $result = [int]$done }
        break
    }
}

& $Adb pull "$remote/out/." "$local" 2>&1 | Out-Null
if ($null -eq $result) {
    $lc = "$local\$Run-logcat.txt"
    & $Adb logcat -d -v time "GcdrBench:*" "DOTNET:*" "AndroidRuntime:*" "DEBUG:*" "libc:*" "ActivityManager:*" "lowmemorykiller:*" "*:F" "*:S" | Out-File -Encoding utf8 $lc
    Write-Output "[$Run] process ended without .done - logcat saved to $lc"
    exit 98
}
if ($result -ne 0) {
    & $Adb logcat -d -v time "GcdrBench:*" "DOTNET:*" "AndroidRuntime:*" "*:F" "*:S" | Out-File -Encoding utf8 "$local\$Run-logcat.txt"
}
Write-Output "[$Run] exit $result -> $local\$Run.log"
exit $result
