param(
    [switch]$Strict,
    [switch]$DryRun,
    [switch]$Ocr,
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

if (-not $DryRun) {
    if (-not $env:MINERU_API_KEY) { throw "PDF 全流程需要 MINERU_API_KEY。" }
    if (-not $env:DMX_API_KEY -and -not $env:LLM_API_KEY) { throw "当前抽取配置需要 DMX_API_KEY。" }
}

$Work = Join-Path $Root "work_pdf_pipeline"
$OutputDir = Join-Path $Root $(if ($Strict) { "output_pdf_strict" } else { "output_pdf_preview" })
$ArgsList = @(
    (Join-Path $Root "pipeline_runner.py"),
    "--input-dir", (Join-Path $Root "source_pdfs"),
    "--mineru-output", (Join-Path $Work "mineru"),
    "--organized-root", (Join-Path $Work "organized"),
    "--processed-output", (Join-Path $Work "processed"),
    "--output-dir", $OutputDir,
    "--config", (Join-Path $Root "extraction\config\pipeline.yaml"),
    "--env-file", $EnvFile,
    "--ref-list", (Join-Path $Root "preview\demo_latest_20_refs.txt"),
    "--workers", "1",
    "--llm-workers", "1"
)
if (-not $Strict) { $ArgsList += "--preview" }
if ($DryRun) { $ArgsList += "--dry-run" }
if ($Ocr) { $ArgsList += "--ocr" }
if ($Force) { $ArgsList += "--force" }

& python @ArgsList
exit $LASTEXITCODE
