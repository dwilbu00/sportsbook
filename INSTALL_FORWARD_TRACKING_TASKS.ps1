<#
.NAME
    INSTALL_FORWARD_TRACKING_TASKS.ps1
.AUTHOR
    ODI Sportsbook Value Finder
.PURPOSE
    Install or remove Windows scheduled tasks for exact-line closing snapshots
    and daily prediction outcome resolution.
.USAGE
    powershell -ExecutionPolicy Bypass -File .\INSTALL_FORWARD_TRACKING_TASKS.ps1
    powershell -ExecutionPolicy Bypass -File .\INSTALL_FORWARD_TRACKING_TASKS.ps1 -Remove
#>

param(
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$ClosingTaskName = "Sportsbook Forward Tracker - Closing Odds"
$ResolveTaskName = "Sportsbook Forward Tracker - Resolve Outcomes"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $ClosingTaskName -Confirm:$false -ErrorAction SilentlyContinue
    Unregister-ScheduledTask -TaskName $ResolveTaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Removed forward-tracking scheduled tasks."
    exit 0
}

$PythonPath = (Get-Command python -ErrorAction Stop).Source
$ScriptDirectory = Split-Path -Parent $MyInvocation.MyCommand.Path
$TrackerPath = Join-Path $ScriptDirectory "forward_tracker.py"
if (-not (Test-Path $TrackerPath)) {
    throw "forward_tracker.py was not found at $TrackerPath"
}

$Principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited
$ClosingSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
$ResolveSettings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 2)

$ClosingAction = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument ('"{0}" --capture-closing --window-minutes 10' -f $TrackerPath) `
    -WorkingDirectory $ScriptDirectory
$ClosingTrigger = New-ScheduledTaskTrigger `
    -Once `
    -At (Get-Date).AddMinutes(1) `
    -RepetitionInterval (New-TimeSpan -Minutes 5) `
    -RepetitionDuration (New-TimeSpan -Days 3650)
Register-ScheduledTask `
    -TaskName $ClosingTaskName `
    -Action $ClosingAction `
    -Trigger $ClosingTrigger `
    -Principal $Principal `
    -Settings $ClosingSettings `
    -Description "Capture one exact player-prop price snapshot within 10 minutes of game time." `
    -Force | Out-Null

$ResolveAction = New-ScheduledTaskAction `
    -Execute $PythonPath `
    -Argument ('"{0}" --resolve' -f $TrackerPath) `
    -WorkingDirectory $ScriptDirectory
$ResolveTrigger = New-ScheduledTaskTrigger -Daily -At "6:00 AM"
Register-ScheduledTask `
    -TaskName $ResolveTaskName `
    -Action $ResolveAction `
    -Trigger $ResolveTrigger `
    -Principal $Principal `
    -Settings $ResolveSettings `
    -Description "Resolve completed sportsbook predictions and run gated recalibration." `
    -Force | Out-Null

Write-Host "Installed:"
Write-Host "  $ClosingTaskName (every 5 minutes; API calls only near logged events)"
Write-Host "  $ResolveTaskName (daily at 6:00 AM)"
Write-Host "Python: $PythonPath"
Write-Host "Storage is configured through PREDICTION_LOG_BLOB_URL or .streamlit\secrets.toml."
