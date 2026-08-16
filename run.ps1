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
    # Elasticsearch is OOM-killed below ~8GB, which is the single most common
    # cause of the stack failing to come up.
    try {
        $memBytes = (docker info --format "{{.MemTotal}}" 2>$null)
        $memGb = [math]::Round([double]$memBytes / 1GB, 1)
        if ($memGb -lt 7) {
            Warn "Docker memory is ${memGb}GB. Raise it to 8GB or more:"
            Warn "Docker Desktop -> Settings -> Resources -> Memory,"
            Warn "otherwise Elasticsearch will be killed on startup."
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
    Say " Starting the stack (first run pulls ~3GB and takes 30-40 min)" "Cyan"
    Say "=================================================================" "Cyan"
    docker compose up -d

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
        Bad "Elasticsearch did not come up. Check: docker compose logs elasticsearch"
        Bad "Most likely cause: Docker memory below 8GB."
        exit 1
    }
    Ok "Elasticsearch is up"

    Say "Creating the Elasticsearch indices..." "Cyan"
    # Must happen BEFORE any write: the dense_vector mapping cannot be added
    # afterwards without a full reindex.
    python -c "from tickets.serving.es_client import create_indices; print(create_indices())"

    Say ""
    Ok "Stack is ready:"
    Say "    Kibana        http://localhost:5601"
    Say "    MinIO console http://localhost:9001   (minioadmin / minioadmin)"
    Say "    Spark UI      http://localhost:8080"
    Say ""
    Say "Next, in TWO separate PowerShell windows (both in this folder):" "Cyan"
    Say "    1)  tickets-producer --rate 200"
    Say "    2)  docker compose exec spark spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 /opt/project/src/tickets/spark/stream_job.py"
    Say ""
    Say "See docs\DEMO.md for the full runbook."
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
