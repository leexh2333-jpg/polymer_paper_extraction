param(
    [switch]$Strict,
    [switch]$DryRun,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

$EnvFile = Join-Path $Root ".env"
if (Test-Path -LiteralPath $EnvFile) {
    Get-Content -LiteralPath $EnvFile -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { return }
        $name, $value = $line.Split("=", 2)
        $name = $name.Trim()
        $value = $value.Trim()
        if ($name -and -not [Environment]::GetEnvironmentVariable($name, "Process")) {
            [Environment]::SetEnvironmentVariable($name, $value, "Process")
        }
    }
}

if (-not $DryRun -and -not $env:DMX_API_KEY -and -not $env:LLM_API_KEY) {
    throw "当前配置 provider=dmx。请设置 DMX_API_KEY，或复制 .env.example 为 .env 后填写。"
}

$OutputDir = Join-Path $Root $(if ($Strict) { "output_strict" } else { "output_preview" })
$ArgsList = @(
    (Join-Path $Root "extraction\batch_runner.py"),
    "--config", (Join-Path $Root "extraction\config\pipeline.yaml"),
    "--input-dir", (Join-Path $Root "sample_data\processed_documents"),
    "--output-dir", $OutputDir,
    "--ref-list", (Join-Path $Root "preview\demo_latest_20_refs.txt"),
    "--state-db", (Join-Path $OutputDir "_batch\batch_state.sqlite3"),
    "--summary-out", (Join-Path $OutputDir "_batch\run_summary.json"),
    "--workers", "1",
    "--llm-workers", "1"
)
if (-not $Strict) { $ArgsList += "--preview" }
if ($DryRun) { $ArgsList += "--dry-run" }
if ($Force) { $ArgsList += "--force" }

& python @ArgsList
exit $LASTEXITCODE
