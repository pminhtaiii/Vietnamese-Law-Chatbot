# ============================================================
# run_eval_sweep.ps1 — Automated Hyperparameter Sweep
# ============================================================
#
# Runs the evaluation pipeline across multiple configurations
# and logs each run to MLflow for visual comparison.
#
# Usage:
#   cd evaluation
#   .\run_eval_sweep.ps1
#
# Prerequisites:
#   pip install mlflow google-generativeai cohere pandas tqdm httpx
#   Set GEMINI_API_KEY and COHERE_API_KEY environment variables
# ============================================================

$ErrorActionPreference = "Stop"

# ── Configuration ──────────────────────────────────────────────
$GEMINI_KEY    = $env:GEMINI_API_KEY
$COHERE_KEY    = $env:COHERE_API_KEY
$N_SAMPLES     = 50                      # Keep small for sweeps; increase for final benchmark
$SEED          = 42
$BACKEND_URL   = "http://localhost:8000"
$USE_BACKEND   = $true                   # Set to $false to use oracle GT context

# ── Sweep Grid ─────────────────────────────────────────────────
# Each combination will be evaluated and logged as a separate MLflow run.
$TOP_K_VALUES       = @(3, 5, 10)
$GENERATOR_MODELS   = @("gemini-3.1-flash-lite-preview")
$JUDGE_MODEL        = "command-r-plus"
$MLFLOW_EXPERIMENT  = "legal-rag-evaluation"

# ── Validation ─────────────────────────────────────────────────
if (-not $GEMINI_KEY) {
    Write-Host "ERROR: GEMINI_API_KEY environment variable is not set." -ForegroundColor Red
    Write-Host "  Set it with:  `$env:GEMINI_API_KEY = 'your-key-here'" 
    exit 1
}
if (-not $COHERE_KEY) {
    Write-Host "ERROR: COHERE_API_KEY environment variable is not set." -ForegroundColor Red
    Write-Host "  Set it with:  `$env:COHERE_API_KEY = 'your-key-here'"
    exit 1
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$EvalScript = Join-Path $ScriptDir "legal_rag_evaluator.py"

if (-not (Test-Path $EvalScript)) {
    Write-Host "ERROR: Cannot find $EvalScript" -ForegroundColor Red
    exit 1
}

# ── Run Sweep ──────────────────────────────────────────────────
$TotalRuns = $TOP_K_VALUES.Count * $GENERATOR_MODELS.Count
$RunIndex = 0
$StartTime = Get-Date

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Legal RAG Evaluation — Parameter Sweep              ║" -ForegroundColor Cyan
Write-Host "╠══════════════════════════════════════════════════════════╣" -ForegroundColor Cyan
Write-Host "║  Total runs planned : $TotalRuns" -ForegroundColor Cyan
Write-Host "║  Samples per run    : $N_SAMPLES" -ForegroundColor Cyan
Write-Host "║  top_k values       : $($TOP_K_VALUES -join ', ')" -ForegroundColor Cyan
Write-Host "║  Generator models   : $($GENERATOR_MODELS -join ', ')" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

foreach ($gen_model in $GENERATOR_MODELS) {
    foreach ($top_k in $TOP_K_VALUES) {
        $RunIndex++
        $RunName   = "sweep_topk${top_k}_${gen_model}"
        $OutputDir = Join-Path $ScriptDir "eval_results" "sweep_topk${top_k}_${gen_model}"

        Write-Host "────────────────────────────────────────────────────" -ForegroundColor DarkGray
        Write-Host "[$RunIndex/$TotalRuns] Running: $RunName" -ForegroundColor Yellow
        Write-Host "  top_k=$top_k  generator=$gen_model" -ForegroundColor DarkGray

        $args_list = @(
            $EvalScript,
            "--gemini_key",       $GEMINI_KEY,
            "--cohere_key",       $COHERE_KEY,
            "--n_samples",        $N_SAMPLES,
            "--top_k",            $top_k,
            "--generator_model",  $gen_model,
            "--judge_model",      $JUDGE_MODEL,
            "--output_dir",       $OutputDir,
            "--seed",             $SEED,
            "--mlflow_run_name",  $RunName,
            "--mlflow_experiment", $MLFLOW_EXPERIMENT
        )

        if (-not $USE_BACKEND) {
            $args_list += "--no_backend"
        } else {
            $args_list += @("--backend_url", $BACKEND_URL)
        }

        try {
            & python @args_list
            Write-Host "  ✓ $RunName completed" -ForegroundColor Green
        }
        catch {
            Write-Host "  ✗ $RunName FAILED: $_" -ForegroundColor Red
        }

        Write-Host ""
    }
}

# ── Summary ────────────────────────────────────────────────────
$Elapsed = (Get-Date) - $StartTime
Write-Host "╔══════════════════════════════════════════════════════════╗" -ForegroundColor Green
Write-Host "║     Sweep Complete                                      ║" -ForegroundColor Green
Write-Host "║  Total runs  : $TotalRuns" -ForegroundColor Green
Write-Host "║  Elapsed     : $($Elapsed.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
Write-Host "╚══════════════════════════════════════════════════════════╝" -ForegroundColor Green
Write-Host ""
Write-Host "View results in MLflow:" -ForegroundColor Cyan
Write-Host "  cd $ScriptDir" -ForegroundColor White
Write-Host "  mlflow ui" -ForegroundColor White
Write-Host "  Then open http://127.0.0.1:5000 in your browser." -ForegroundColor White
