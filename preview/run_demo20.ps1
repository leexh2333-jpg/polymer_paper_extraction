[CmdletBinding()]
param(
    [ValidateSet("Preflight", "Verify", "Cached", "Fresh")]
    [string]$Mode = "Preflight",

    [string]$ConfigPath,
    [string]$InputDir,
    [string]$OutputDir,
    [string]$RefList,
    [string]$PythonExe = "python",

    [ValidateRange(1, 64)]
    [int]$Workers = 1,

    [ValidateRange(1, 64)]
    [int]$LlmWorkers = 1,

    [switch]$AllowModelCalls
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PreviewRoot = $PSScriptRoot
$TestcodeRoot = Split-Path -Parent $PreviewRoot
$ExtractionRoot = Join-Path $TestcodeRoot "extraction"
$BaselineOutput = Join-Path $ExtractionRoot "output_test"

if (-not $ConfigPath) {
    $ConfigPath = Join-Path $ExtractionRoot "config\pipeline.yaml"
}
if (-not $InputDir) {
    $InputDir = Join-Path $BaselineOutput "_pipeline\processed\documents"
}
if (-not $RefList) {
    $RefList = Join-Path $PreviewRoot "demo_latest_20_refs.txt"
}

function Resolve-ExistingFile([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "$Label does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-ExistingDirectory([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label does not exist: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function Resolve-OutputDirectory([string]$Path) {
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

function Test-SamePath([string]$Left, [string]$Right) {
    $leftFull = [IO.Path]::GetFullPath($Left).TrimEnd('\', '/')
    $rightFull = [IO.Path]::GetFullPath($Right).TrimEnd('\', '/')
    return [string]::Equals(
        $leftFull,
        $rightFull,
        [StringComparison]::OrdinalIgnoreCase
    )
}

function Assert-Preflight {
    param(
        [string]$ResolvedConfig,
        [string]$ResolvedInput,
        [string]$ResolvedRefList
    )

    $pythonCommand = Get-Command $PythonExe -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python executable not found: $PythonExe"
    }

    $refs = @(
        Get-Content -LiteralPath $ResolvedRefList -Encoding UTF8 |
            ForEach-Object { $_.Trim() } |
            Where-Object { $_ -and -not $_.StartsWith("#") }
    )
    if ($refs.Count -ne 20) {
        throw "The ref-list must contain exactly 20 refs; found $($refs.Count): $ResolvedRefList"
    }
    $uniqueRefs = @($refs | Sort-Object -Unique)
    if ($uniqueRefs.Count -ne 20) {
        throw "The ref-list contains duplicate refs: $ResolvedRefList"
    }

    $missing = @()
    foreach ($ref in $refs) {
        $documentPath = Join-Path $ResolvedInput "${ref}_document.json"
        if (-not (Test-Path -LiteralPath $documentPath -PathType Leaf)) {
            $missing += $documentPath
        }
    }
    if ($missing.Count -gt 0) {
        throw "Missing $($missing.Count) normalized inputs:`n$($missing -join "`n")"
    }

    & $PythonExe -c "import sys; from pathlib import Path; from llm_client import load_pipeline_config; load_pipeline_config(Path(sys.argv[1])); print('pipeline.yaml: OK')" $ResolvedConfig
    if ($LASTEXITCODE -ne 0) {
        throw "pipeline.yaml validation failed: $ResolvedConfig"
    }

    Write-Host "Preflight passed: 20/20 normalized inputs exist."
    Write-Host "Config : $ResolvedConfig"
    Write-Host "Input  : $ResolvedInput"
    Write-Host "RefList: $ResolvedRefList"
}

$ConfigPath = Resolve-ExistingFile $ConfigPath "Config file"
$InputDir = Resolve-ExistingDirectory $InputDir "Normalized input directory"
$RefList = Resolve-ExistingFile $RefList "20-paper ref-list"

Push-Location $ExtractionRoot
try {
    Assert-Preflight `
        -ResolvedConfig $ConfigPath `
        -ResolvedInput $InputDir `
        -ResolvedRefList $RefList

    if ($Mode -eq "Preflight") {
        exit 0
    }

    if ($Mode -eq "Verify") {
        if (-not $OutputDir) {
            $OutputDir = $BaselineOutput
        }
        $OutputDir = Resolve-ExistingDirectory $OutputDir "Verification output directory"
        $verifyReport = Join-Path $OutputDir "_batch\demo20_verify_report.json"
        & $PythonExe (Join-Path $PreviewRoot "verify_demo20.py") `
            --ref-list $RefList `
            --output-dir $OutputDir `
            --expected-count 20 `
            --report-out $verifyReport
        exit $LASTEXITCODE
    }

    if (-not $PSBoundParameters.ContainsKey("OutputDir")) {
        throw "$Mode requires an explicit -OutputDir to prevent accidental baseline writes."
    }
    if (-not $AllowModelCalls) {
        throw "$Mode may call model APIs. Add -AllowModelCalls after confirming cost and credentials."
    }

    $OutputDir = Resolve-OutputDirectory $OutputDir
    if (Test-SamePath $OutputDir $InputDir) {
        throw "OutputDir cannot equal InputDir: $OutputDir"
    }
    if ($Mode -eq "Fresh") {
        if (Test-SamePath $OutputDir $BaselineOutput) {
            throw "Fresh mode cannot overwrite baseline output_test: $BaselineOutput"
        }
        if (Test-Path -LiteralPath $OutputDir) {
            $firstItem = Get-ChildItem -LiteralPath $OutputDir -Force | Select-Object -First 1
            if ($firstItem) {
                throw "Fresh OutputDir must not exist or must be empty: $OutputDir"
            }
        }
    }

    $batchDir = Join-Path $OutputDir "_batch"
    $stateDb = Join-Path $batchDir "demo20_state.sqlite3"
    $summaryOut = Join-Path $batchDir "demo20_run_summary.json"
    $verifyReport = Join-Path $batchDir "demo20_verify_report.json"

    $batchArgs = @(
        (Join-Path $ExtractionRoot "batch_runner.py"),
        "--config", $ConfigPath,
        "--input-dir", $InputDir,
        "--output-dir", $OutputDir,
        "--state-db", $stateDb,
        "--summary-out", $summaryOut,
        "--ref-list", $RefList,
        "--workers", "$Workers",
        "--llm-workers", "$LlmWorkers",
        "--retry-failed",
        "--retry-interrupted",
        "--recheck-completed",
        "--preview"
    )
    if ($Mode -eq "Fresh") {
        $batchArgs += "--force"
    }

    Write-Host "Starting $Mode run. Model/API calls are authorized."
    Write-Host "Output : $OutputDir"
    Write-Host "StateDB: $stateDb"
    & $PythonExe @batchArgs
    $batchExit = $LASTEXITCODE

    & $PythonExe (Join-Path $PreviewRoot "verify_demo20.py") `
        --ref-list $RefList `
        --output-dir $OutputDir `
        --expected-count 20 `
        --report-out $verifyReport
    $verifyExit = $LASTEXITCODE

    if ($batchExit -ne 0 -or $verifyExit -ne 0) {
        Write-Host "20-paper acceptance failed. batch=$batchExit, verify=$verifyExit"
        exit 1
    }

    Write-Host "20/20 complete pipeline acceptance passed."
    Write-Host "Run summary  : $summaryOut"
    Write-Host "Verify report: $verifyReport"
    exit 0
}
finally {
    Pop-Location
}
