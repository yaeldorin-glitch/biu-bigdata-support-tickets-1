# Windows helper - run this from inside the project folder.
#
#   .\run.ps1            check the setup, then run the pipeline
#   .\run.ps1 -Stack     also bring up Docker (Kafka/Spark/MinIO/Elasticsearch)
#   .\run.ps1 -Api       start the web API only
#   .\run.ps1 -Check     check the setup and stop
#
# If PowerShell refuses with "running scripts is disabled on this system", run:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# and then run this script again. That relaxes the policy for this window only.

param(
    [switch]$Stack,
    [switch]$Api,
    [switch]$Check,
    [switch]$Full
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot   # always operate on the project folder, wherever it was launched from

function Say($msg, $colour = "White") { Write-Host $msg -ForegroundColor $colour }
function Ok($msg)   { Say "  [OK]   $msg" "Green" }
function Warn($msg) { Say "  [WARN] $msg" "Yellow" }
function Bad($msg)  { Say "  [FAIL] $msg" "Red" }

Say ""
Say "=================================================================" "Cyan"
Say " Ticket Intelligence - setup check" "Cyan"
Say "=================================================================" "Cyan"
Say "  folder: $PSScriptRoot"
Say ""

$problems = 0

# --- python ---------------------------------------------------------------
try {
    $pyVersion = (python --version 2>&1) -join ""
    Ok "Python found: $pyVersion"
} catch {
    Bad "Python not found. Install it from python.org and tick 'Add to PATH'."
    $problems++
}

# --- package installed ----------------------------------------------------
$installed = $false
try {
    python -c "import tickets" 2>$null
    if ($LASTEXITCODE -eq 0) { $installed = $true }
} catch { }

if ($installed) {
    Ok "The 'tickets' package is installed"
} else {
    Warn "The 'tickets' package is not installed yet - installing now..."
    python -m pip install -e . --quiet
    if ($LASTEXITCODE -eq 0) { Ok "Installed" } else { Bad "pip install failed"; $problems++ }
}

# --- dataset --------------------------------------------------------------
# The full CSV is not in the repository (CC BY-NC), so this is the most common
# reason someone gets numbers that do not match the report.
$raw = Join-Path $PSScriptRoot "data\raw\tickets.csv"
if (Test-Path $raw) {
    $sizeMb = [math]::Round((Get-Item $raw).Length / 1MB, 1)
    Ok "Full dataset found ($sizeMb MB)"
} else {
    Warn "data\raw\tickets.csv is MISSING."
    Warn "The pipeline will fall back to the 300-row sample and the numbers"
    Warn "will NOT match the report. Put the Kaggle CSV there and rename it"
    Warn "to tickets.csv - see data\README.md."
}

# --- docker ---------------------------------------------------------------
$dockerUp = $false
try {
    docker info 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) { $dockerUp = $true; Ok "Docker is running" }
    else { Warn "Docker is installed but not running - start Docker Desktop" }
} catch {
    Warn "Docker not found (only needed for the full stack, not for the pipeline)"
}

if ($dockerUp) {
    # This script always uses docker-compose.slim.yml, sized to fit a WSL2
    # allocation of ~5GB (peaks at ~3.8GB) -- NOT the ~8GB the default
    # docker-compose.yml assumes. Below ~4GB, even the slim stack's
    # Elasticsearch can be OOM-killed on startup.
    try {
        $memBytes = (docker info --format "{{.MemTotal}}" 2>$null)
        $memGb = [math]::Round([double]$memBytes / 1GB, 1)
        if ($memGb -lt 4) {
            Warn "Docker memory is ${memGb}GB, which is tight even for the slim stack."
            Warn "Close other memory-heavy apps (Chrome especially) before -Stack."
        } else {
            Ok "Docker memory: ${memGb}GB"
        }
    } catch { }
}

Say ""
if ($problems -gt 0) { Bad "$problems blocking problem(s) - fix them before continuing"; exit 1 }
if ($Check) { Ok "Check complete."; exit 0 }

# --- the API only ---------------------------------------------------------
if ($Api) {
    Say "Starting the API at http://localhost:8000/docs ..." "Cyan"
    Say "(Ctrl+C to stop)"
    $env:OFFLINE_API = "1"
    python -m uvicorn tickets.serving.api:app --host 0.0.0.0 --port 8000
    exit 0
}

# --- the docker stack -----------------------------------------------------
if ($Stack) {
    if (-not $dockerUp) { Bad "Docker is not running - start Docker Desktop first"; exit 1 }

    Say "=================================================================" "Cyan"
    Say " Starting the slim stack (first run pulls ~3GB and takes 30-40 min)" "Cyan"
    Say "=================================================================" "Cyan"
    docker compose -f docker-compose.slim.yml up -d --build

    Say ""
    Say "Waiting for Elasticsearch..." "Cyan"
    $ready = $false
    foreach ($i in 1..60) {
        try {
            Invoke-RestMethod "http://localhost:9200/_cluster/health" -TimeoutSec 3 | Out-Null
            $ready = $true; break
        } catch { Start-Sleep -Seconds 5; Write-Host "." -NoNewline }
    }
    Say ""
    if (-not $ready) {
        Bad "Elasticsearch did not come up. Check: docker compose -f docker-compose.slim.yml logs elasticsearch"
        Bad "Most likely cause: not enough free memory for even the slim stack."
        exit 1
    }
    Ok "Elasticsearch is up"

    Say "Creating the Elasticsearch indices..." "Cyan"
    # Must happen BEFORE any write: the dense_vector mapping cannot be added
    # afterwards without a full reindex. (stream_job.py also creates it on its
    # own if this is skipped, so this is a convenience step, not a strict gate.)
    python -c "from tickets.serving.es_client import create_indices; print(create_indices())"

    $offlineModel = Join-Path $PSScriptRoot "output\offline_embedding_model.joblib"
    if (-not (Test-Path $offlineModel)) {
        Warn "No fitted offline embedding model yet - the Spark job needs one"
        Warn "(it cannot fit itself on a single streaming batch). Fitting now..."
        $env:EMBEDDING_BACKEND = "offline"
        tickets-pipeline --limit 3000
    }

    Say ""
    Ok "Stack is ready (slim: no Kibana):"
    Say "    MinIO console http://localhost:9001   (minioadmin / minioadmin)"
    Say "    Spark UI      http://localhost:4040"
    Say ""
    Say "To stream new tickets live (optional -- the index already has data in it" "Cyan"
    Say "from previous runs, so this is only needed to show live streaming), in TWO" "Cyan"
    Say "separate PowerShell windows (both in this folder):" "Cyan"
    Say "    1)  tickets-producer --rate 200"
    Say "    2)  docker compose -f docker-compose.slim.yml exec spark spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 docker/spark/run_module.py tickets.spark.stream_job --trigger-seconds 5"
    Say ""
    Say "(spark-submit's target is docker/spark/run_module.py, not stream_job.py" "DarkGray"
    Say " directly -- spark-submit runs its target as a bare script, which breaks" "DarkGray"
    Say " stream_job.py's relative imports. See docs/DEMO.md.)" "DarkGray"
    Say ""
    Say "See docs\DEMO.md for the full runbook." "Cyan"
    Say ""
    Say "=================================================================" "Cyan"
    Say " Starting the demo link now: http://localhost:8000/docs" "Cyan"
    Say "=================================================================" "Cyan"
    Say "First request after this takes about a minute -- it is loading the AI" "DarkGray"
    Say "model, not stuck. Leave this window open for the whole demo; Ctrl+C stops it." "DarkGray"
    Say ""
    python -m uvicorn tickets.serving.api:app --host 0.0.0.0 --port 8000
    exit 0
}

# --- default: run the pipeline -------------------------------------------
Say "=================================================================" "Cyan"
Say " Running the pipeline" "Cyan"
Say "=================================================================" "Cyan"
Say ""

if ($Full) { tickets-pipeline --full } else { tickets-pipeline --limit 5000 }

Say ""
Ok "Done. Results written to output\report.json"
Say ""
Say "What next:" "Cyan"
Say "    .\run.ps1 -Full     run over the whole dataset"
Say "    .\run.ps1 -Api      start the web API (http://localhost:8000/docs)"
Say "    .\run.ps1 -Stack    bring up Kafka + Spark + MinIO + Elasticsearch"
Say ""
