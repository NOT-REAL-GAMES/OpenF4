#requires -Version 7.4
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$root = Join-Path $repo ('build/compatibility/game-test-' + [Guid]::NewGuid().ToString('N'))
$driver = Join-Path $repo 'tools/compatibility-game.ps1'
$shell = (Get-Process -Id $PID).Path
function Require($Condition, $Message) { if (-not $Condition) { throw $Message } }
function Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Write-Fixture([string]$Path, [string]$Text) {
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path))
    [IO.File]::WriteAllText($Path, $Text)
}
function Json($Value, [string]$Path) { Write-Fixture $Path ($Value | ConvertTo-Json -Depth 12) }
function Fixture([string]$Name) {
    $case = Join-Path $root $Name
    $run = Join-Path $case 'run'; $game = Join-Path $case 'game'
    $documents = Join-Path $case 'documents'; $local = Join-Path $case 'local'
    Write-Fixture (Join-Path $game 'Fallout4.exe') 'Synthetic executable; never launch this file'
    [void][IO.Directory]::CreateDirectory((Join-Path $game 'Data'))
    [void][IO.Directory]::CreateDirectory($local)
    foreach ($name in @('Fallout4.ini', 'Fallout4Prefs.ini')) { Write-Fixture (Join-Path $documents $name) "[General]`r`nsLocalSavePath=Saves\`r`n[SaveGame]`r`nbSaveOnPause=1`r`n" }
    Write-Fixture (Join-Path $local 'Plugins.txt') "# User settings`r`n*PersonalPlugin.esp`r`n"
    $artifacts = @()
    foreach ($generation in @('E0', 'E10')) {
        $capture = Join-Path $run "runtime-$generation"
        Json @{ status='Pass'; references=@(0,1,2,3) } (Join-Path $capture 'runtime.json')
        $rows = @()
        foreach ($index in 0..3) {
            foreach ($type in @(@('meshes','nif'), @('materials','bgsm'), @('textures','dds'))) {
                $relative = "$($type[0])/openf4/$generation/$index/fixture.$($type[1])"
                $source = Join-Path $run "fixtures/$generation.assets/$relative"
                Write-Fixture $source "Synthetic native bytes $generation $index $($type[1])"
                $captured = "consumed/$index-$($type[1]).bin"
                Write-Fixture (Join-Path $capture $captured) ([IO.File]::ReadAllText($source))
                $rows += (@{ resolvedPath=$source; capturedPath=$captured; source='project' } | ConvertTo-Json -Compress)
            }
        }
        Write-Fixture (Join-Path $capture 'assets.jsonl') ($rows -join "`n")
        $plugin = Join-Path $run "fixtures/Fallout4-$generation.esp"
        Write-Fixture $plugin "Synthetic plugin $generation"
        $artifacts += @{path="fixtures/Fallout4-$generation.esp"; sha256=(Hash $plugin)}
    }
    Json @{automatedStatus='Pass';sourceMatchesBuild=$true;artifacts=$artifacts} (Join-Path $run 'manifest.json')
    $reused = Join-Path $game 'Data/meshes/openf4/E0/0/fixture.nif'
    Write-Fixture $reused ([IO.File]::ReadAllText((Join-Path $run 'fixtures/E0.assets/meshes/openf4/E0/0/fixture.nif')))
    $original = @{}
    foreach ($path in @((Join-Path $documents 'Fallout4.ini'), (Join-Path $documents 'Fallout4Prefs.ini'), (Join-Path $local 'Plugins.txt'))) { $original[$path] = Hash $path }
    @{run=$run;game=$game;documents=$documents;local=$local;original=$original;reused=$reused;reusedHash=(Hash $reused)}
}
function Invoke-Driver($Fixture, [string]$Action, [bool]$ShouldPass = $true) {
    $proofArgument = if ($Fixture.ContainsKey('worldPackage')) { @('-WorldPackage', $Fixture.worldPackage) } else { @() }
    $log = & $shell -NoProfile -File $driver -Action $Action -RunDirectory $Fixture.run -GameDirectory $Fixture.game `
        -SettingsDirectory $Fixture.documents -LocalSettingsDirectory $Fixture.local @proofArgument 2>&1
    Require (($LASTEXITCODE -eq 0) -eq $ShouldPass) "Unexpected $Action outcome: $log"
}
function World-Fixture([string]$Name) {
    $fixture = Fixture $Name
    $fixture.worldPackage = Join-Path ([IO.Path]::GetDirectoryName($fixture.run)) 'world-package'
    $outputs = @('Data/OF4_Proof.esp', 'Data/meshes/openf4/proof/cube.nif', 'Data/materials/openf4/proof/cube.bgsm', 'Data/textures/openf4/proof/cube.dds', 'Data/scripts/OpenF4/Wake.pex')
    $report = "OpenF4 native world geometry proof`nNOT A COMPLETE PLAYABLE-WORLD EXPORT`n`nOutput SHA-256:`n"
    foreach ($relative in $outputs) {
        $path = Join-Path $fixture.worldPackage $relative
        Write-Fixture $path "Synthetic world proof bytes: $relative"
        $report += (Hash $path) + '  ' + $relative + "`n"
    }
    Write-Fixture (Join-Path $fixture.worldPackage 'export-report.txt') $report
    # The separate manual proof mode must not require or manufacture an automated Pass.
    Write-Fixture (Join-Path $fixture.run 'manifest.json') '{ no automated acceptance exists for this world proof'
    $fixture
}
function Unchanged-Preflight($Fixture) {
    Require (-not (Test-Path -LiteralPath (Join-Path $Fixture.run 'fallout4-test'))) 'Rejected proof created staging state'
    foreach ($path in $Fixture.original.Keys) { Require ((Hash $path) -eq $Fixture.original[$path]) 'Rejected proof changed settings' }
    Require (-not (Test-Path -LiteralPath (Join-Path $Fixture.game 'Data/OF4_Proof.esp'))) 'Rejected proof installed its plugin'
}
function Restored($Fixture) {
    foreach ($path in $Fixture.original.Keys) { Require ((Hash $path) -eq $Fixture.original[$path]) "Original setting was not restored: $path" }
    Require (-not (Test-Path -LiteralPath (Join-Path $Fixture.documents 'Fallout4Custom.ini'))) 'Originally missing custom INI was not removed'
    Require (-not (Test-Path -LiteralPath (Join-Path $Fixture.local 'loadorder.txt'))) 'Originally missing loadorder was not removed'
    Require ((Hash $Fixture.reused) -eq $Fixture.reusedHash) 'Reused asset changed or disappeared'
    $state = Get-Content -LiteralPath (Join-Path $Fixture.run 'fallout4-test/state.json') -Raw | ConvertFrom-Json
    Require ([bool]$state.restoredUtc) 'Restore did not complete its journal'
    foreach ($file in $state.files | Where-Object installedByTest) {
        Require (-not (Test-Path -LiteralPath $file.destination)) 'Test asset remained installed'
        Require (-not (Test-Path -LiteralPath $file.stagingPath)) 'Staging temporary remained installed'
    }
}
try {
    $normal = Fixture 'normal'
    Invoke-Driver $normal stage
    $state = Get-Content -LiteralPath (Join-Path $normal.run 'fallout4-test/state.json') -Raw | ConvertFrom-Json
    Require (($state.files | Where-Object { -not $_.installedByTest }).Count -eq 1) 'Existing identical asset was not reused'
    foreach ($file in $state.files) { Require ((Hash $file.destination) -eq $file.sha256) 'Installed bytes differ from journal' }
    foreach ($setting in $state.settings | Where-Object existed) { Require ((Hash $setting.backup) -eq $normal.original[$setting.path]) 'Original settings backup differs' }
    Require ((Get-Content -LiteralPath (Join-Path $normal.documents 'Fallout4.ini') -Raw) -match 'bSaveOnPause=0') 'Autosave was not disabled in synthetic settings'
    Invoke-Driver $normal restore
    Restored $normal
    $stateHash = Hash (Join-Path $normal.run 'fallout4-test/state.json')
    Invoke-Driver $normal restore $false
    Require ((Hash (Join-Path $normal.run 'fallout4-test/state.json')) -eq $stateHash) 'Repeated restore changed completed journal'

    $conflict = Fixture 'conflict'
    $conflictingPath = Join-Path $conflict.game 'Data/materials/openf4/E0/0/fixture.bgsm'
    Write-Fixture $conflictingPath 'Existing unrelated user material'
    $conflictHash = Hash $conflictingPath
    Invoke-Driver $conflict stage $false
    Require ((Hash $conflictingPath) -eq $conflictHash) 'Staging overwrote conflicting user data'
    Require (-not (Test-Path -LiteralPath (Join-Path $conflict.run 'fallout4-test'))) 'Conflict was detected after staging started'
    foreach ($path in $conflict.original.Keys) { Require ((Hash $path) -eq $conflict.original[$path]) 'Failed preflight changed a setting' }

    $interrupted = Fixture 'interrupted'
    Invoke-Driver $interrupted stage
    $stateRoot = Join-Path $interrupted.run 'fallout4-test'
    $state = Get-Content -LiteralPath (Join-Path $stateRoot 'state.json') -Raw | ConvertFrom-Json
    $file = @($state.files | Where-Object installedByTest)[0]
    Move-Item -LiteralPath $file.destination -Destination (Join-Path $stateRoot 'test-removed-complete-asset.bin')
    Write-Fixture $file.stagingPath 'Incomplete asset bytes from interrupted copy'
    Write-Fixture $state.settings[0].restoreTemp 'Incomplete settings restoration'
    Write-Fixture (Join-Path $stateRoot 'state.json.interrupted.tmp') '{ truncated journal update'
    Invoke-Driver $interrupted restore
    Restored $interrupted
    Require (@(Get-ChildItem -LiteralPath (Join-Path $stateRoot 'recovered-temporary') -File).Count -eq 2) 'Interrupted writes were not recovered'
    Require ((Get-Content -LiteralPath (Join-Path $stateRoot 'state.json.interrupted.tmp') -Raw) -eq '{ truncated journal update') 'Interrupted journal evidence was overwritten'

    $junction = Fixture 'junction'
    $linkedDocuments = Join-Path ([IO.Path]::GetDirectoryName($junction.documents)) 'linked-documents'
    $null = New-Item -ItemType Junction -Path $linkedDocuments -Value $junction.documents
    $junction.documents = $linkedDocuments
    Invoke-Driver $junction stage $false
    Require (-not (Test-Path -LiteralPath (Join-Path $junction.run 'fallout4-test'))) 'Settings junction was not rejected before staging'
    foreach ($path in $junction.original.Keys) { Require ((Hash $path) -eq $junction.original[$path]) 'Rejected junction changed a setting' }

    $world = World-Fixture 'world-normal'
    $worldReused = Join-Path $world.game 'Data/meshes/openf4/proof/cube.nif'
    Write-Fixture $worldReused ([IO.File]::ReadAllText((Join-Path $world.worldPackage 'Data/meshes/openf4/proof/cube.nif')))
    $worldReusedHash = Hash $worldReused
    Invoke-Driver $world stage
    $state = Get-Content -LiteralPath (Join-Path $world.run 'fallout4-test/state.json') -Raw | ConvertFrom-Json
    Require ($state.mode -eq 'WorldGeometryProof' -and $state.proofOnly -and $state.gameChecks -eq 'Not run') 'Manual world proof was not labeled honestly'
    Require ($state.exportReportSha256 -eq (Hash (Join-Path $world.worldPackage 'export-report.txt'))) 'Proof report hash missing from journal'
    Require ($state.files.Count -eq 5 -and @($state.files | Where-Object { -not $_.installedByTest }).Count -eq 1) 'World proof file ownership is wrong'
    Require ((Hash (Join-Path $world.game 'Data/scripts/OpenF4/Wake.pex')) -eq (Hash (Join-Path $world.worldPackage 'Data/scripts/OpenF4/Wake.pex'))) 'Staged script differs from exported PEX'
    foreach ($file in $state.files) { Require ((Hash $file.destination) -eq $file.sha256) 'World proof bytes changed during installation' }
    Require ((Get-Content -LiteralPath (Join-Path $world.local 'Plugins.txt') -Raw) -ceq "# Isolated OpenF4 compatibility test`r`n*OF4_Proof.esp`r`n") 'World proof selected the wrong plugins'
    Require ((Get-Content -LiteralPath (Join-Path $world.local 'loadorder.txt') -Raw) -ceq "Fallout4.esm`r`nOF4_Proof.esp`r`n") 'World proof load order is wrong'
    Invoke-Driver $world restore
    Restored $world
    Require ((Hash $worldReused) -eq $worldReusedHash) 'World proof restore removed an identical preexisting dependency'

    foreach ($failure in @('tamper', 'extra', 'missing', 'banner', 'duplicate', 'unsafe', 'wrong-extension', 'two-plugins', 'conflict', 'junction')) {
        $invalid = World-Fixture ('world-' + $failure)
        $reportPath = Join-Path $invalid.worldPackage 'export-report.txt'
        $report = [IO.File]::ReadAllText($reportPath)
        switch ($failure) {
            'tamper' { Write-Fixture (Join-Path $invalid.worldPackage 'Data/meshes/openf4/proof/cube.nif') 'Changed since export' }
            'extra' { Write-Fixture (Join-Path $invalid.worldPackage 'Data/textures/openf4/proof/unlisted.dds') 'Unattested extra' }
            'missing' { Move-Item -LiteralPath (Join-Path $invalid.worldPackage 'Data/OF4_Proof.esp') -Destination (Join-Path $invalid.worldPackage 'removed-plugin.esp') }
            'banner' { Write-Fixture $reportPath $report.Replace('NOT A COMPLETE PLAYABLE-WORLD EXPORT', 'Fully certified playable world') }
            'duplicate' { Write-Fixture $reportPath ($report + ('0' * 64) + "  Data/OF4_PROOF.esp`n") }
            'unsafe' { Write-Fixture $reportPath ($report + ('0' * 64) + "  Data/meshes/../escape.nif`n") }
            'wrong-extension' {
                $path = Join-Path $invalid.worldPackage 'Data/meshes/openf4/proof/model.glb'
                Write-Fixture $path 'Not a native model'
                Write-Fixture $reportPath ($report + (Hash $path) + "  Data/meshes/openf4/proof/model.glb`n")
            }
            'two-plugins' {
                $path = Join-Path $invalid.worldPackage 'Data/OF4_Another.esp'
                Write-Fixture $path 'Second plugin'
                Write-Fixture $reportPath ($report + (Hash $path) + "  Data/OF4_Another.esp`n")
            }
            'conflict' { Write-Fixture (Join-Path $invalid.game 'Data/materials/openf4/proof/cube.bgsm') 'Unrelated existing material' }
            'junction' {
                $null = New-Item -ItemType Junction -Path (Join-Path $invalid.worldPackage 'Data/linked') -Value $invalid.documents
            }
        }
        Invoke-Driver $invalid stage $false
        Unchanged-Preflight $invalid
        if ($failure -eq 'conflict') {
            Require ((Get-Content -LiteralPath (Join-Path $invalid.game 'Data/materials/openf4/proof/cube.bgsm') -Raw) -eq 'Unrelated existing material') 'World proof overwrote conflicting material'
        }
    }
    Write-Host 'Compatibility game staging/restore tests passed; only synthetic settings were used.'
} finally {
    Write-Host "Synthetic game evidence retained: $root"
}
