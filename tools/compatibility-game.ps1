#requires -Version 7.4
[CmdletBinding()]
param(
    [ValidateSet('stage', 'restore')][string]$Action = 'stage',
    [Parameter(Mandatory)][string]$RunDirectory,
    [string]$WorldPackage,
    [string]$GameDirectory = 'C:/Program Files (x86)/Steam/steamapps/common/Fallout 4',
    [string]$SettingsDirectory = (Join-Path ([Environment]::GetFolderPath('MyDocuments')) 'My Games/Fallout4'),
    [string]$LocalSettingsDirectory = (Join-Path $env:LOCALAPPDATA 'Fallout4')
)
$ErrorActionPreference = 'Stop'
if (Get-Process Fallout4 -ErrorAction SilentlyContinue) { throw 'Close Fallout 4 without saving before staging/restoring.' }
$run = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($RunDirectory))
$game = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($GameDirectory))
$documents = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($SettingsDirectory))
$local = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($LocalSettingsDirectory))
$data = Join-Path $game 'Data'
$stateRoot = Join-Path $run 'fallout4-test'
$stateFile = Join-Path $stateRoot 'state.json'
function Hash([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
function Assert-PlainPath([string]$Path) {
    $current = [IO.Path]::GetFullPath($Path)
    $allowCloud = $current -eq $documents -or $current -eq $local -or (Inside $current $documents) -or (Inside $current $local)
    while ($current) {
        if (Test-Path -LiteralPath $current) {
            if ((Get-Item -LiteralPath $current -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                $cloudPlaceholder = $false
                if ($allowCloud) {
                    $tagOutput = & fsutil.exe reparsepoint query $current 2>&1
                    $tagMatches = [regex]::Matches(($tagOutput -join "`n"), '(?m)^[^:\r\n]+:\s*(0x[0-9a-fA-F]{8})\s*$')
                    if ($LASTEXITCODE -eq 0 -and $tagMatches.Count -eq 1) {
                        $tag = [Convert]::ToUInt32($tagMatches[0].Groups[1].Value.Substring(2), 16)
                        # Cloud placeholders retain the same path; symlink/junction/unknown tags do not.
                        $cloudPlaceholder = ($tag -band 0xffff0fffL) -eq 0x9000001aL
                    }
                }
                if (-not $cloudPlaceholder) { throw "Refusing reparse-point path: $current" }
            }
        }
        $current = [IO.Path]::GetDirectoryName($current)
    }
}
function Publish-Temporary([string]$Temporary, [string]$Destination, [switch]$Replace) {
    Assert-PlainPath $Temporary; Assert-PlainPath $Destination
    if ($Replace -and (Test-Path -LiteralPath $Destination)) { [IO.File]::Replace($Temporary, $Destination, [NullString]::Value) }
    else { [IO.File]::Move($Temporary, $Destination) }
}
function Write-Atomic([string]$Text, [string]$Path, [string]$Temporary) {
    Assert-PlainPath $Path; Assert-PlainPath $Temporary
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Path))
    $stream = [IO.File]::Open($Temporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $bytes = [Text.UTF8Encoding]::new($false).GetBytes($Text)
        $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true)
    } finally { $stream.Dispose() }
    Publish-Temporary $Temporary $Path -Replace
}
function Json($Value, [string]$Path) {
    Write-Atomic ($Value | ConvertTo-Json -Depth 12) $Path ($Path + '.' + [Guid]::NewGuid().ToString('N') + '.tmp')
}
function Inside([string]$Path, [string]$Root) {
    ([IO.Path]::GetFullPath($Path)).StartsWith([IO.Path]::GetFullPath($Root) + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)
}
function Copy-New([string]$Source, [string]$Destination, [string]$Temporary, [switch]$Replace) {
    if (-not $Temporary) { $Temporary = $Destination + '.' + [Guid]::NewGuid().ToString('N') + '.tmp' }
    Assert-PlainPath $Source; Assert-PlainPath $Destination; Assert-PlainPath $Temporary
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Destination))
    $inputFile = [IO.File]::OpenRead($Source)
    try {
        $outputFile = [IO.File]::Open($Temporary, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
        try { $inputFile.CopyTo($outputFile); $outputFile.Flush($true) } finally { $outputFile.Dispose() }
    } finally { $inputFile.Dispose() }
    if ((Hash $Source) -ne (Hash $Temporary)) { throw "Incomplete staging copy: $Source" }
    Publish-Temporary $Temporary $Destination -Replace:$Replace
}
function Set-Ini([string]$Text, [string]$Section, [string]$Key, [string]$Value) {
    $sectionPattern = '(?ms)(^\[' + [regex]::Escape($Section) + '\][^\r\n]*\r?\n)(.*?)(?=^\[|\z)'
    if (-not [regex]::IsMatch($Text, $sectionPattern)) { return $Text.TrimEnd() + "`r`n[$Section]`r`n$Key=$Value`r`n" }
    [regex]::Replace($Text, $sectionPattern, [Text.RegularExpressions.MatchEvaluator]{ param($m)
        $keyPattern = '(?m)^\s*' + [regex]::Escape($Key) + '\s*=[^\r\n]*'
        $body = $m.Groups[2].Value
        if ([regex]::IsMatch($body, $keyPattern)) { $body = [regex]::Replace($body, $keyPattern, "$Key=$Value") }
        else { $body = $body.TrimEnd() + "`r`n$Key=$Value`r`n" }
        $m.Groups[1].Value + $body
    })
}
function Read-WorldPackage([string]$Path) {
    $package = [IO.Path]::TrimEndingDirectorySeparator([IO.Path]::GetFullPath($Path))
    $packageData = Join-Path $package 'Data'
    $report = Join-Path $package 'export-report.txt'
    foreach ($path in @($package, $packageData, $report)) { Assert-PlainPath $path }
    if (-not (Test-Path -LiteralPath $packageData -PathType Container)) { throw 'World proof has no Data directory.' }
    $lines = [IO.File]::ReadAllLines($report)
    if ($lines.Count -lt 4 -or $lines[0] -cne 'OpenF4 native world geometry proof' -or
        $lines[1] -cne 'NOT A COMPLETE PLAYABLE-WORLD EXPORT') { throw 'World package is not a geometry proof export.' }
    $headings = @(0..($lines.Count - 1) | Where-Object { $lines[$_] -ceq 'Output SHA-256:' })
    if ($headings.Count -ne 1 -or $headings[0] -eq $lines.Count - 1) { throw 'World proof has no unique output hash list.' }
    $expected = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($line in $lines[($headings[0] + 1)..($lines.Count - 1)]) {
        if ($line -cnotmatch '^([0-9a-f]{64})  (Data/.+)$') { throw 'Malformed world proof output hash entry.' }
        $digest = $Matches[1]; $relative = $Matches[2]
        foreach ($part in $relative.Split('/')) {
            if (-not $part -or $part -in @('.', '..') -or $part -match '[<>:"\\|?*\x00-\x1f]' -or $part -match '[ .]$' -or
                $part -match '^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)') { throw 'Unsafe world proof output path.' }
        }
        if ($relative -notmatch '^Data/OF4_[A-Za-z0-9_]+\.esp$' -and
            $relative -notmatch '^Data/(meshes/.+\.nif|materials/.+\.bgsm|textures/.+\.dds|scripts/.+\.pex)$') {
            throw "Only a native OF4 proof plugin and NIF/BGSM/DDS/PEX dependencies may be staged: $relative"
        }
        if (-not $expected.TryAdd($relative, $digest)) { throw 'Duplicate world proof output path.' }
    }
    $plugins = @($expected.Keys | Where-Object { $_ -match '^Data/OF4_[A-Za-z0-9_]+\.esp$' })
    if ($plugins.Count -ne 1) { throw 'World proof must contain exactly one root OF4_*.esp plugin.' }
    $result = [Collections.Generic.List[object]]::new()
    $directories = [Collections.Generic.Stack[string]]::new(); $directories.Push($packageData)
    while ($directories.Count) {
        foreach ($entry in Get-ChildItem -LiteralPath $directories.Pop() -Force) {
            # Inspect each directory before descending so even an unlisted junction is rejected.
            Assert-PlainPath $entry.FullName
            if ($entry.PSIsContainer) { $directories.Push($entry.FullName); continue }
            $relative = [IO.Path]::GetRelativePath($package, $entry.FullName).Replace('\', '/')
            if (-not $expected.ContainsKey($relative)) { throw "Unlisted world proof file: $relative" }
            $digest = Hash $entry.FullName
            if ($digest -ne $expected[$relative]) { throw "World proof changed after export: $relative" }
            $result.Add([ordered]@{ source=$entry.FullName; destination=(Join-Path $data $relative.Substring(5)); sha256=$digest; installedByTest=$true })
        }
    }
    if ($result.Count -ne $expected.Count) { throw 'World proof is missing a hashed output file.' }
    @{ path=$package; report=$report; reportSha256=(Hash $report); plugin=$plugins[0].Substring(5); files=$result }
}

foreach ($path in @($run, $game, $data, $stateRoot, $stateFile, $documents, $local)) { Assert-PlainPath $path }

if ($Action -eq 'restore') {
    $state = Get-Content -LiteralPath $stateFile -Raw | ConvertFrom-Json
    if ($state.restoredUtc) { throw 'This test setup has already been restored.' }
    if ([IO.Path]::GetFullPath($state.gameDirectory) -ne $game) { throw 'Game path differs from staged state.' }
    if ($state.settingsDirectory -ne $documents -or $state.localSettingsDirectory -ne $local) { throw 'Settings paths differ from staged state.' }
    # Complete preflight before moving any installed data or restoring settings.
    foreach ($file in $state.files) {
        if (-not (Inside $file.destination $data)) { throw 'Invalid staged asset path' }
        if ($file.stagingPath -ne ($file.destination + '.openf4-stage-' + $state.stagingId + '.tmp')) { throw 'Invalid staging temporary path' }
        Assert-PlainPath $file.destination; Assert-PlainPath $file.stagingPath
        if (Test-Path -LiteralPath $file.destination) {
            if ((Hash $file.destination) -ne $file.sha256) { throw "Installed test asset changed; refusing to move it: $($file.destination)" }
        }
    }
    foreach ($setting in $state.settings) {
        if ([IO.Path]::GetDirectoryName($setting.path) -notin @($documents, $local) -or
            [IO.Path]::GetFileName($setting.path) -notin @('Fallout4.ini','Fallout4Prefs.ini','Fallout4Custom.ini','Plugins.txt','loadorder.txt')) {
            throw 'Invalid settings restore target'
        }
        if ($setting.editTemp -ne ($setting.path + '.openf4-edit-' + $state.stagingId + '.tmp') -or
            $setting.restoreTemp -ne ($setting.path + '.openf4-restore-' + $state.stagingId + '.tmp')) { throw 'Invalid settings temporary path' }
        foreach ($path in @($setting.path, $setting.backup, $setting.editTemp, $setting.restoreTemp)) { Assert-PlainPath $path }
        if ($setting.existed -and ((-not (Inside $setting.backup $stateRoot)) -or (Hash $setting.backup) -ne $setting.sha256)) {
            throw 'Settings backup failed its integrity check'
        }
    }
    # Interrupted writes never replace a live file; retain their partial bytes inside the journal.
    $temporaries = @($state.files | ForEach-Object { $_.stagingPath }) + @($state.settings | ForEach-Object { $_.editTemp; $_.restoreTemp })
    foreach ($temporary in $temporaries) {
        if (Test-Path -LiteralPath $temporary) {
            $recovered = Join-Path $stateRoot ('recovered-temporary/' + [Guid]::NewGuid().ToString('N') + '-' + [IO.Path]::GetFileName($temporary))
            Assert-PlainPath $recovered
            [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($recovered))
            Move-Item -LiteralPath $temporary -Destination $recovered
        }
    }
    [void][IO.Directory]::CreateDirectory((Join-Path $stateRoot 'post-test-settings'))
    foreach ($setting in $state.settings) {
        if (Test-Path -LiteralPath $setting.path) {
            $archive = Join-Path $stateRoot ('post-test-settings/' + [IO.Path]::GetFileName($setting.path))
            if (-not (Test-Path -LiteralPath $archive)) { Copy-New $setting.path $archive }
        }
        if ($setting.existed) {
            Copy-New $setting.backup $setting.path $setting.restoreTemp -Replace
            if ((Hash $setting.path) -ne $setting.sha256) { throw "Settings restoration hash mismatch: $($setting.path)" }
        } elseif (Test-Path -LiteralPath $setting.path) {
            Move-Item -LiteralPath $setting.path -Destination (Join-Path $stateRoot ('created-' + [IO.Path]::GetFileName($setting.path)))
        }
    }
    foreach ($file in $state.files) {
        if ($file.installedByTest -and (Test-Path -LiteralPath $file.destination)) {
            $target = Join-Path $stateRoot ('recovered-install/' + [IO.Path]::GetRelativePath($data, $file.destination))
            Assert-PlainPath $target
            [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($target))
            Move-Item -LiteralPath $file.destination -Destination $target
        }
    }
    $state.restoredUtc = [DateTime]::UtcNow.ToString('o'); Json $state $stateFile
    Write-Host 'Settings restored with matching hashes; installed test files recovered. Existing saves were not touched.'
    exit 0
}

if (Test-Path -LiteralPath $stateRoot) { throw 'Test state directory exists; restore it rather than overwriting it.' }
$executable = Join-Path $game 'Fallout4.exe'
if (-not (Test-Path -LiteralPath $executable -PathType Leaf) -or -not (Test-Path -LiteralPath $data -PathType Container)) { throw 'Fallout 4 installation was not found.' }
$files = [Collections.Generic.List[object]]::new()
$stagingId = [Guid]::NewGuid().ToString('N')
if ($WorldPackage) {
    $proof = Read-WorldPackage $WorldPackage
    $files = $proof.files
    $plugins = @($proof.plugin)
} else {
$manifest = Get-Content -LiteralPath (Join-Path $run 'manifest.json') -Raw | ConvertFrom-Json
if ($manifest.automatedStatus -ne 'Pass' -or -not $manifest.sourceMatchesBuild) { throw 'Stage only a passing, source-identified automated run.' }
$plugins = @('OpenF4CompatibilityE0.esp', 'OpenF4CompatibilityE10.esp')
foreach ($generation in @('E0', 'E10')) {
    $capture = Join-Path $run "runtime-$generation"
    $runtime = Get-Content -LiteralPath (Join-Path $capture 'runtime.json') -Raw | ConvertFrom-Json
    if ($runtime.status -ne 'Pass') { throw "Missing passing $generation runtime gate" }
    $assetRoot = Join-Path $run "fixtures/$generation.assets"
    $seen = @{}
    foreach ($row in (Get-Content -LiteralPath (Join-Path $capture 'assets.jsonl') | ForEach-Object { $_ | ConvertFrom-Json })) {
        if (-not $row.capturedPath -or $seen.ContainsKey($row.resolvedPath)) { continue }
        $seen[$row.resolvedPath] = $true
        $source = [IO.Path]::GetFullPath($row.resolvedPath)
        $captured = [IO.Path]::GetFullPath((Join-Path $capture $row.capturedPath))
        if (-not (Inside $source $assetRoot) -or -not (Inside $captured $capture)) { throw 'Asset capture escaped fixture scope' }
        Assert-PlainPath $source; Assert-PlainPath $captured
        $relative = [IO.Path]::GetRelativePath($assetRoot, $source)
        if ($relative -notmatch '^(meshes|materials|textures)[\\/]openf4[\\/]' -or [IO.Path]::GetExtension($source) -notin @('.nif','.bgsm','.dds')) { throw 'Only validated generated native dependencies may be staged.' }
        $hash = Hash $source
        if ($hash -ne (Hash $captured)) { throw "Fixture changed after runtime verification: $source" }
        $files.Add([ordered]@{ source=$source; destination=(Join-Path $data $relative); sha256=$hash; installedByTest=$true })
    }
    $plugin = Join-Path $run "fixtures/Fallout4-$generation.esp"
    $artifact = @($manifest.artifacts | Where-Object { $_.path -eq "fixtures/Fallout4-$generation.esp" })
    if ($artifact.Count -ne 1 -or (Hash $plugin) -ne $artifact[0].sha256) { throw 'Proof plugin differs from verified evidence' }
    $files.Add([ordered]@{ source=$plugin; destination=(Join-Path $data "OpenF4Compatibility$generation.esp"); sha256=(Hash $plugin); installedByTest=$true })
}
}
foreach ($file in $files) {
    if (-not (Inside $file.destination $data)) { throw 'Asset target escaped Data directory' }
    Assert-PlainPath $file.source; Assert-PlainPath $file.destination
    $file.stagingPath = $file.destination + '.openf4-stage-' + $stagingId + '.tmp'
    if (Test-Path -LiteralPath $file.destination) {
        if ((Hash $file.destination) -ne $file.sha256) { throw "Refusing existing conflicting file: $($file.destination)" }
        $file.installedByTest = $false
    }
}
$settings = [Collections.Generic.List[object]]::new()
[void][IO.Directory]::CreateDirectory((Join-Path $stateRoot 'settings-backup'))
foreach ($path in @((Join-Path $documents 'Fallout4.ini'), (Join-Path $documents 'Fallout4Prefs.ini'), (Join-Path $documents 'Fallout4Custom.ini'), (Join-Path $local 'Plugins.txt'), (Join-Path $local 'loadorder.txt'))) {
    $backup = Join-Path $stateRoot ('settings-backup/' + [IO.Path]::GetFileName($path))
    $exists = Test-Path -LiteralPath $path -PathType Leaf
    if ($exists) { Copy-New $path $backup }
    $settings.Add([ordered]@{ path=$path; existed=$exists; backup=$backup; sha256=$(if ($exists) { Hash $backup } else { $null })
        editTemp=($path + '.openf4-edit-' + $stagingId + '.tmp'); restoreTemp=($path + '.openf4-restore-' + $stagingId + '.tmp') })
}
$state = [ordered]@{ schema=1; stagedUtc=[DateTime]::UtcNow.ToString('o'); restoredUtc=$null
    stagingId=$stagingId; settingsDirectory=$documents; localSettingsDirectory=$local
    gameDirectory=$game; executableSha256=(Hash $executable); executableVersion=(Get-Item -LiteralPath $executable).VersionInfo.FileVersion
    files=$files; settings=$settings; gameChecks='Not run'; reviewer=$null }
if ($WorldPackage) {
    $state.mode = 'WorldGeometryProof'
    $state.proofOnly = $true
    $state.worldPackage = $proof.path
    $state.exportReportSha256 = $proof.reportSha256
    $state.limitations = 'Manual geometry acceptance only; NPC navigation, native LOD and complete playable-world compatibility are not certified.'
}
Json $state $stateFile
try {
    foreach ($file in $files) {
        if ($file.installedByTest) { Copy-New $file.source $file.destination $file.stagingPath }
        if ((Hash $file.destination) -ne $file.sha256) { throw 'Installed native package hash mismatch' }
    }
    foreach ($setting in $settings | Where-Object { $_.path.EndsWith('.ini') }) {
        $text = if ($setting.existed) { [IO.File]::ReadAllText($setting.path) } else { '' }
        $text = Set-Ini $text 'General' 'sLocalSavePath' ('OpenF4Compatibility-' + [IO.Path]::GetFileName($run) + '\')
        foreach ($key in @('bSaveOnPause','bSaveOnTravel','bSaveOnWait','bSaveOnRest','bSaveOnInteriorExteriorSwitch')) { $text = Set-Ini $text 'SaveGame' $key '0' }
        if ($setting.path.EndsWith('Fallout4Custom.ini')) {
            $text = Set-Ini $text 'Archive' 'bInvalidateOlderFiles' '1'
            $text = Set-Ini $text 'Archive' 'sResourceDataDirsFinal' ''
        }
        Write-Atomic $text $setting.path $setting.editTemp
    }
    foreach ($setting in $settings | Where-Object { -not $_.path.EndsWith('.ini') }) {
        $text = if ([IO.Path]::GetFileName($setting.path) -eq 'Plugins.txt') {
            "# Isolated OpenF4 compatibility test`r`n" + (($plugins | ForEach-Object { '*' + $_ }) -join "`r`n") + "`r`n"
        } else { "Fallout4.esm`r`n" + ($plugins -join "`r`n") + "`r`n" }
        Write-Atomic $text $setting.path $setting.editTemp
    }
    $state.activeSettings = @($settings | ForEach-Object { @{ path=$_.path; sha256=(Hash $_.path) } })
    Json $state $stateFile
} catch {
    Write-Warning "Staging interrupted. Backups and restore journal are preserved at $stateRoot. Run this script with -Action restore before launching Fallout 4."
    throw
}
if ($WorldPackage) {
    Write-Host "Staged $($proof.plugin) for isolated MANUAL geometry-proof acceptance. No automated game pass is claimed."
    Write-Host "Navigation, native LOD and full playable-world compatibility are not certified. Use the arrival instructions in $($proof.report)."
} else {
    Write-Host "Staged E0 and E10 packages. Main menu: coc qasmoke; help OF4CompatE0 4; help OF4CompatE10 4."
    Write-Host 'Spawn returned IDs with player.placeatme. _0 asymmetric movable, _1 doorway, _2 compound, _3 nonuniform mass compound.'
}
Write-Host "Quit WITHOUT saving, then restore: ./tools/compatibility-game.ps1 -Action restore -RunDirectory `"$run`""
