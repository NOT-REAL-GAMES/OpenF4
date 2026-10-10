#requires -Version 7.4
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$root = Join-Path $repo ('build/compatibility/driver-test-' + [Guid]::NewGuid().ToString('N'))
$fakeBuild = Join-Path $root 'input'
$proof = Join-Path $fakeBuild 'asset-native-proof-01'
$fixture = Join-Path $proof 'NativeFixtures.assets/meshes/fixture.nif'
$private = Join-Path $proof 'settings-backup/not-an-asset.nif'
[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($fixture))
[void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($private))
[IO.File]::WriteAllText($fixture, 'synthetic proof bytes, not a real NIF')
[IO.File]::WriteAllText($private, 'must not be archived')
$evidence = Join-Path $root 'evidence'
$driver = Join-Path $repo 'tools/compatibility.ps1'
$shell = (Get-Process -Id $PID).Path
function Require($Condition, $Message) { if (-not $Condition) { throw $Message } }
try {
    $log = & $shell -NoProfile -File $driver -Mode baseline -BuildDirectory $fakeBuild -OutputDirectory $evidence 2>&1
    Require ($LASTEXITCODE -eq 0) "Baseline failed: $log"
    $copied = Join-Path $evidence 'historical/asset-native-proof-01/NativeFixtures.assets/meshes/fixture.nif'
    Require ((Get-FileHash -LiteralPath $fixture).Hash -eq (Get-FileHash -LiteralPath $copied).Hash) 'Archive changed proof bytes'
    Require (-not (Test-Path -LiteralPath (Join-Path $evidence 'historical/asset-native-proof-01/settings-backup'))) 'Private settings were archived'
    $manifestPath = Join-Path $evidence 'manifest.json'
    $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    Require ($manifest.profileStatus -eq 'Not proven' -and -not $manifest.sourceMatchesBuild) 'Historical evidence was promoted to measured proof'
    Require ($manifest.results[0].status -eq 'Reported pass') 'Historical report lost its status'
    foreach ($artifact in $manifest.artifacts) {
        Require ((Get-FileHash -LiteralPath (Join-Path $evidence $artifact.path)).Hash.ToLowerInvariant() -eq $artifact.sha256) 'Manifest SHA-256 mismatch'
    }
    $before = (Get-FileHash -LiteralPath $manifestPath).Hash
    $null = & $shell -NoProfile -File $driver -Mode baseline -BuildDirectory $fakeBuild -OutputDirectory $evidence 2>&1
    Require ($LASTEXITCODE -ne 0) 'Existing evidence directory was accepted'
    Require ((Get-FileHash -LiteralPath $manifestPath).Hash -eq $before) 'Rejected run changed existing evidence'
    $outside = Join-Path $repo ('forbidden-evidence-' + [Guid]::NewGuid().ToString('N'))
    $null = & $shell -NoProfile -File $driver -Mode baseline -BuildDirectory $fakeBuild -OutputDirectory $outside 2>&1
    Require ($LASTEXITCODE -ne 0 -and -not (Test-Path -LiteralPath $outside)) 'Out-of-scope output directory was created'
    Write-Host 'Compatibility evidence driver tests passed'
} finally {
    # Keep failed evidence too; this is a bounded, uniquely named build artifact.
    Write-Host "Driver test evidence: $root"
}
