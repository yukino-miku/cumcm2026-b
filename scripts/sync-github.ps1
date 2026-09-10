#Requires -Version 7.0
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Message
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Invoke-Git {
    param([string[]]$GitArguments)
    & git @GitArguments
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArguments -join ' ') failed (exit $LASTEXITCODE)."
    }
}

$projectRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
Push-Location -LiteralPath $projectRoot
try {
    $actualRoot = Invoke-Git -GitArguments @('rev-parse', '--show-toplevel')
    if ([IO.Path]::GetFullPath($actualRoot) -ne $projectRoot) {
        throw 'Repository root does not match the project directory.'
    }
    $branch = Invoke-Git -GitArguments @('branch', '--show-current')
    if ($branch -ne 'main') {
        throw "Expected main branch; found '$branch'. Review before syncing."
    }
    $remote = Invoke-Git -GitArguments @('remote', 'get-url', '--push', '--all', 'origin')
    $allowedRemotes = @(
        'https://github.com/yukino-miku/cumcm2026-a.git',
        'https://github.com/yukino-miku/cumcm2026-a',
        'git@github.com:yukino-miku/cumcm2026-a.git'
    )
    if (@($remote).Count -ne 1 -or $remote -notin $allowedRemotes) {
        throw 'Unexpected push destination. Review origin before syncing.'
    }

    Invoke-Git -GitArguments @('add', '--all')
    & git diff --cached --quiet
    $diffStatus = $LASTEXITCODE
    if ($diffStatus -eq 1) {
        Invoke-Git -GitArguments @('commit', '-m', $Message)
    } elseif ($diffStatus -ne 0) {
        throw "Could not inspect staged changes (exit $diffStatus)."
    }

    Invoke-Git -GitArguments @('push', '--set-upstream', 'origin', 'main')
    $localHead = Invoke-Git -GitArguments @('rev-parse', 'HEAD')
    $remoteHead = & gh api --hostname github.com repos/yukino-miku/cumcm2026-a/branches/main --jq .commit.sha
    if ($LASTEXITCODE -ne 0) {
        throw 'The commit was pushed, but GitHub API verification failed. Retry syncing.'
    }
    if ($localHead -ne $remoteHead) {
        throw "Verification failed: local $localHead differs from remote $remoteHead."
    }
    $remaining = Invoke-Git -GitArguments @('status', '--porcelain')
    if ($remaining) {
        throw 'The commit was pushed, but new local changes remain. Review and sync again.'
    }
    Write-Output "SYNC_OK $localHead https://github.com/yukino-miku/cumcm2026-a"
} finally {
    Pop-Location
}
