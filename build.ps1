# TPT Local Chat - build pipeline (Windows).
#
# Free edition only - this app is intentionally not a revenue line (apps.md
# §T2-19): its job is brand visibility and funneling into TPT Chat PDF. There
# is no Pro feature flag, no Gumroad listing, and no separate desktop build:
#   cargo build --release --target wasm32-unknown-unknown
#   wasm-bindgen --target web  ->  web repo public/apps/local-chat/ when
#   TPT_WEB_REPO is set (glue named local-chat.js, wasm keeps its _bg.wasm
#   name - the glue resolves it via import.meta.url), otherwise dist\web.
#
# Env:
#   TPT_WEB_REPO  path to the tptsolutions.co.nz web repo; the app is copied
#                 to <repo>\public\apps\local-chat\ for the hub runner.
#                 Defaults to the known local checkout below if unset.

[CmdletBinding()]
param(
    # Skip the cargo build and re-run wasm-bindgen against an existing
    # release build.
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$Crate = "tpt-app-local-chat"
$Target = "wasm32-unknown-unknown"
$OutName = "local-chat"
$Slug = "local-chat"
$DefaultWebRepo = "D:\Programming\2 WIP\TPT Electrician\tpt-electrician-nz-2"

Write-Host "== TPT Local Chat build ==" -ForegroundColor Cyan

# Where the browser-loadable bundle goes.
$WebRepo = if ($env:TPT_WEB_REPO) { $env:TPT_WEB_REPO } elseif (Test-Path $DefaultWebRepo) { $DefaultWebRepo } else { $null }
$Dest = if ($WebRepo) {
    Join-Path $WebRepo "public\apps\local-chat"
} else {
    Join-Path $PSScriptRoot "dist\web"
}

Write-Host " edition: free (the only tier - local model, in-browser)"
Write-Host " dest:    $Dest"

if (-not $SkipBuild) {
    Write-Host "-- cargo build --release --target $Target" -ForegroundColor Cyan
    cargo build --release --target $Target -p $Crate
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}

$Wasm = Join-Path $PSScriptRoot "target\$Target\release\tpt_app_local_chat.wasm"
if (-not (Test-Path $Wasm)) { throw "missing $Wasm - run the cargo build step first" }

Write-Host "-- wasm-bindgen --target web --out-name $OutName" -ForegroundColor Cyan
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
wasm-bindgen --target web --out-name $OutName --out-dir $Dest $Wasm
if ($LASTEXITCODE -ne 0) { throw "wasm-bindgen failed" }

$Glue = Join-Path $Dest "$OutName.js"
$BgWasm = Join-Path $Dest "$($OutName)_bg.wasm"
foreach ($artifact in @($Glue, $BgWasm)) {
    if (-not (Test-Path $artifact)) { throw "wasm-bindgen did not produce $artifact" }
}

# Payload size report (local-model weights are tracked separately from this
# code-size line - they are the dominant size cost for this app).
$raw = (Get-Item $BgWasm).Length
Add-Type -AssemblyName System.IO.Compression
$rawBytes = [System.IO.File]::ReadAllBytes($BgWasm)
$memoryStream = [System.IO.MemoryStream]::new()
$gzipStream = [System.IO.Compression.GZipStream]::new($memoryStream, [System.IO.Compression.CompressionLevel]::Optimal)
$gzipStream.Write($rawBytes, 0, $rawBytes.Length)
$gzipStream.Dispose()
$gzipped = $memoryStream.ToArray().Length
$memoryStream.Dispose()
$gzKb = [math]::Round($gzipped / 1KB, 1)
Write-Host "-- $(Split-Path $BgWasm -Leaf): $raw bytes raw, $gzipped bytes gzipped ($gzKb KB gz)" -ForegroundColor Cyan

Write-Host "-- done" -ForegroundColor Green
if (-not $WebRepo) {
    Write-Host "   (set TPT_WEB_REPO to copy the bundle into the hub web repo)"
} else {
    Write-Host ""
    Write-Host "Paste into src/lib/tpt-apps-registry.ts (TPT_APPS_REGISTRY), then flip status to 'live':" -ForegroundColor Cyan
    Write-Host @"
  {
    slug: '$Slug',
    name: 'TPT Local Chat',
    description: 'Chat with a small language model running entirely in your browser - no cloud, no API key.',
    category: 'developer',
    status: 'live',
    kind: 'demo',
    wasm: { entry: '/apps/$Slug/$OutName.js' },
    // free app - deliberately no gumroadUrl/price (apps.md §T2-19)
  },
"@
}
