#requires -Version 7.4
[CmdletBinding()]
param(
    [ValidateSet('baseline', 'verify')][string]$Mode = 'verify',
    [Parameter(Mandatory)][string]$OutputDirectory,
    [string]$BuildDirectory = 'build',
    [switch]$SkipRuntime
)

$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$output = [IO.Path]::GetFullPath($OutputDirectory)
$build = [IO.Path]::GetFullPath($BuildDirectory)
if (Test-Path -LiteralPath $output) { throw "Evidence directory already exists: $output" }
# Evidence must never become part of the source snapshot being captured.
if (-not $output.StartsWith((Join-Path $repo 'build') + [IO.Path]::DirectorySeparatorChar,
        [StringComparison]::OrdinalIgnoreCase)) { throw 'Use a new evidence directory under the repository build/ directory.' }
[void][IO.Directory]::CreateDirectory($output)
$steps = [Collections.Generic.List[object]]::new()
$results = [Collections.Generic.List[object]]::new()
$manifest = [ordered]@{
    schema = 1; mode = $Mode; startedUtc = [DateTime]::UtcNow.ToString('o')
    sourceCommit = ''; sourceMatchesBuild = $false; configuration = 'Release'
    fixtureRevision = 1; toleranceRevision = 1; reviewer = $null
    profileStatus = 'Not proven'; steps = $steps; results = $results
    environment = [ordered]@{ os = [Environment]::OSVersion.ToString(); processArchitecture = [Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture.ToString() }
}

function Write-Json($Value, [string]$Path) {
    [IO.File]::WriteAllText($Path, ($Value | ConvertTo-Json -Depth 30), [Text.UTF8Encoding]::new($false))
}
function File-Evidence([string]$Path, [string]$Root) {
    $file = Get-Item -LiteralPath $Path
    [ordered]@{ path = [IO.Path]::GetRelativePath($Root, $file.FullName).Replace('\', '/'); bytes = $file.Length
        sha256 = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant() }
}
function Copy-Evidence([string]$Source, [string]$Destination) {
    [void][IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Destination))
    Copy-Item -LiteralPath $Source -Destination $Destination -ErrorAction Stop
    if ((Get-FileHash -LiteralPath $Source).Hash -ne (Get-FileHash -LiteralPath $Destination).Hash) {
        throw "Evidence copy differs from its source: $Source"
    }
}
function Source-Snapshot([switch]$Copy) {
    $paths = & git -C $repo ls-files --cached --others --exclude-standard -- src tests apps tools CMakeLists.txt vcpkg.json README.md .gitignore
    if ($LASTEXITCODE -ne 0) { throw 'Cannot enumerate source snapshot' }
    @($paths | Sort-Object -Unique | ForEach-Object {
        $path = Join-Path $repo $_
        if (Test-Path -LiteralPath $path -PathType Leaf) {
            if ($Copy) { Copy-Evidence $path (Join-Path $output "source/$_") }
            File-Evidence $path $repo
        }
    })
}
function Capture-BuildInputs {
    foreach ($relative in @('CMakeCache.txt', 'vcpkg_installed/vcpkg/status',
            'vcpkg_installed/x64-windows/share/Jolt/JoltConfig.cmake',
            'vcpkg_installed/x64-windows/share/Jolt/JoltConfigVersion.cmake',
            'vcpkg_installed/x64-windows/share/Jolt/JoltTargets.cmake')) {
        $path = Join-Path $build $relative
        if (Test-Path -LiteralPath $path -PathType Leaf) { Copy-Evidence $path (Join-Path $output "build-inputs/$relative") }
    }
}
function Run-Step([string]$Name, [string]$Executable, [string[]]$Arguments, [int]$TimeoutSeconds = 180) {
    $stdout = Join-Path $output "$Name.stdout.log"
    $stderr = Join-Path $output "$Name.stderr.log"
    # All supplied arguments are paths or fixed tokens; embedded quotes are not needed.
    $quoted = @($Arguments | ForEach-Object {
        if ($_.Contains('"')) { throw 'Unsupported quote in process argument' }
        '"' + $_.TrimEnd('\') + '"'
    })
    $step = [ordered]@{ name = $Name; executable = $Executable; arguments = $Arguments
        startedUtc = [DateTime]::UtcNow.ToString('o'); exitCode = $null; status = 'Fail' }
    $steps.Add($step)
    $process = Start-Process -FilePath $Executable -ArgumentList $quoted -WorkingDirectory $repo -WindowStyle Hidden `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
    if (-not $process.WaitForExit($TimeoutSeconds * 1000)) {
        $process.Kill($true); $process.WaitForExit()
        throw "$Name timed out after $TimeoutSeconds seconds (artifacts preserved)"
    }
    $step.exitCode = $process.ExitCode
    $step.finishedUtc = [DateTime]::UtcNow.ToString('o')
    if ($process.ExitCode -ne 0) { throw "$Name failed ($($process.ExitCode)); see $stdout and $stderr" }
    $step.status = 'Pass'
}

$exitStatus = 0
try {
    $manifest.sourceCommit = (& git -C $repo rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot identify repository revision' }
    $diffPath = Join-Path $output 'tracked-changes.patch'
    & git -C $repo diff --binary HEAD "--output=$diffPath" -- src tests apps tools CMakeLists.txt vcpkg.json README.md .gitignore
    if ($LASTEXITCODE -ne 0) { throw 'Cannot capture tracked source changes' }
    $before = Source-Snapshot -Copy
    Write-Json $before (Join-Path $output 'source-files.json')
    Capture-BuildInputs
    if ($Mode -eq 'baseline') {
        foreach ($name in @('native-proof-03', 'native-proof-04', 'asset-native-proof-01', 'asset-integration-smoke-03.assets', 'asset-integration-smoke-04.assets')) {
            $path = Join-Path $build $name
            if (-not (Test-Path -LiteralPath $path -PathType Container)) { continue }
            foreach ($file in Get-ChildItem -LiteralPath $path -File -Recurse) {
                # Do not archive settings backups, recovered installs, or user-specific data.
                $relative = [IO.Path]::GetRelativePath($path, $file.FullName)
                if ($relative -match '^(settings-backup|post-test-settings|recovered-install)[\\/]') { continue }
                if ($file.Extension -notin @('.nif', '.bgsm', '.bgem', '.dds', '.esp', '.esm', '.log')) { continue }
                Copy-Evidence $file.FullName (Join-Path $output "historical/$name/$relative")
            }
        }
        foreach ($file in Get-ChildItem -LiteralPath $build -File) {
            if ($file.Name -match '^asset-integration-.*\.(log|esm)$') {
                Copy-Evidence $file.FullName (Join-Path $output "historical/$($file.Name)")
            }
        }
        $results.Add(@{ case = 'historical-fo4'; status = 'Reported pass'; detail = 'User-observed historical native/editor asset checks; not a measured compatibility certification. The doorway fixture is a static triangle mesh. Source snapshot was collected later and is not claimed to match historical binaries.' })
        $manifest.automatedStatus = 'Not run'
    } else {
        Run-Step 'configure' 'cmake' @('-S', $repo, '-B', $build) 600
        Run-Step 'build' 'cmake' @('--build', $build, '--config', 'Release', '--parallel', '8') 1200
        $after = Source-Snapshot
        if (($before | ConvertTo-Json -Depth 5 -Compress) -ne ($after | ConvertTo-Json -Depth 5 -Compress)) {
            throw 'Project source changed during the build; this run cannot identify the built source reliably.'
        }
        $manifest.sourceMatchesBuild = $true
        Capture-BuildInputs
        foreach ($file in Get-ChildItem -LiteralPath (Join-Path $build 'CMakeFiles') -Recurse -Filter CMakeCXXCompiler.cmake -File) {
            Copy-Evidence $file.FullName (Join-Path $output ('build-inputs/' + [IO.Path]::GetRelativePath($build, $file.FullName)))
        }
        if (Get-Command Get-CimInstance -ErrorAction SilentlyContinue) {
            $manifest.environment.graphics = @(Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion)
        }
        Run-Step 'ctest' 'ctest' @('--test-dir', $build, '-C', 'Release', '--output-on-failure') 600
        $binaryRoot = Join-Path $build 'bin/Release'
        $manifest.binaries = @(Get-ChildItem -LiteralPath $binaryRoot -File | Where-Object { $_.Extension -in @('.exe', '.dll') } | ForEach-Object { File-Evidence $_.FullName $binaryRoot })
        $fixtures = Join-Path $output 'fixtures'
        Run-Step 'fixtures' (Join-Path $binaryRoot 'openf4-asset-document-tests.exe') @('--compatibility-fixtures', $fixtures) 180
        $cases = @(Import-Csv -LiteralPath (Join-Path $fixtures 'cases.tsv') -Delimiter "`t")
        if ($cases.Count -lt 8) { throw 'Missing compatibility case inventory' }
        foreach ($case in $cases) {
            $results.Add($case)
            if ($case.status -notin @('Not run', 'Reported pass', 'Pass', 'Fail', 'Blocked', 'N/A')) { throw 'Unknown case status' }
            if ($case.status -eq 'Fail') { throw "Fixture case failed: $($case.case)" }
        }
        $oracle = Get-Content -LiteralPath (Join-Path $fixtures 'source-fixtures.json') -Raw | ConvertFrom-Json
        if ($oracle.fixtures.Count -ne 4 -or @($oracle.fixtures | Where-Object dynamic).Count -ne 3) { throw 'Missing static/movable fixture inventory' }
        $cycles = @(Import-Csv -LiteralPath (Join-Path $fixtures 'cycles.tsv') -Delimiter "`t")
        foreach ($name in @($oracle.fixtures.name) + @('foreign-native')) {
            $chain = @($cycles | Where-Object { $_.case -eq $name })
            if ($chain.Count -ne 11 -or @($chain | Where-Object { $_.status -ne 'Pass' }).Count -ne 0 -or
                (@($chain.generation | ForEach-Object { [int]$_ } | Sort-Object -Unique) -join ',') -ne '0,1,2,3,4,5,6,7,8,9,10') {
                throw "Incomplete successful E0..E10 chain: $name"
            }
        }
        foreach ($generation in @('E0', 'E10')) {
            if ($SkipRuntime) {
                $results.Add(@{ case = "runtime-$generation"; status = 'Not run'; detail = 'Explicitly skipped; runtime gate remains open.' })
                continue
            }
            $capture = Join-Path $output "runtime-$generation"
            $oldCapture = $env:OPENF4_COMPATIBILITY_CAPTURE_DIR
            $oldValidation = $env:OPENF4_VULKAN_VALIDATION
            try {
                $env:OPENF4_COMPATIBILITY_CAPTURE_DIR = $capture
                $env:OPENF4_VULKAN_VALIDATION = '1'
                Run-Step "runtime-$generation" (Join-Path $binaryRoot 'openf4.exe') @((Join-Path $fixtures "$generation.esm")) 180
            } finally {
                $env:OPENF4_COMPATIBILITY_CAPTURE_DIR = $oldCapture
                $env:OPENF4_VULKAN_VALIDATION = $oldValidation
            }
            $runtime = Get-Content -LiteralPath (Join-Path $capture 'runtime.json') -Raw | ConvertFrom-Json
            if ($runtime.status -ne 'Pass') { throw "$generation runtime evidence did not pass" }
            if ($runtime.references.Count -ne $oracle.fixtures.Count) { throw 'Runtime fixture inventory differs from oracle' }
            foreach ($fixture in $oracle.fixtures) {
                $installed = @($runtime.references | Where-Object { $_.formId -eq $fixture.referenceId -and $_.baseId -eq $fixture.modelId })
                $kind = if ($fixture.dynamic) { 'MSTT' } else { 'STAT' }
                if ($installed.Count -ne 1 -or $installed[0].kind -ne $kind) { throw 'Runtime fixture identity/type differs from oracle' }
            }
            $assetRoot = Join-Path $fixtures "$generation.assets"
            $consumed = @(Get-Content -LiteralPath (Join-Path $capture 'assets.jsonl') | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.capturedPath })
            if ($consumed.Count -eq 0) { throw "$generation captured no fixture bytes" }
            foreach ($read in $consumed) {
                $published = [IO.Path]::GetFullPath($read.resolvedPath)
                $captured = [IO.Path]::GetFullPath((Join-Path $capture $read.capturedPath))
                if (-not $published.StartsWith($assetRoot + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase) -or
                    -not $captured.StartsWith($capture + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
                    throw 'Asset evidence path escaped the fixture/capture scope'
                }
                if (-not (Test-Path -LiteralPath $published -PathType Leaf) -or
                    (Get-FileHash -LiteralPath $captured).Hash -ne (Get-FileHash -LiteralPath $published).Hash) {
                    throw "Consumed bytes differ from published $generation package: $published"
                }
            }
            foreach ($reference in $runtime.references) {
                $model = $reference.modelPath.Replace('\', '/').ToLowerInvariant()
                if (-not $model.StartsWith('meshes/')) { $model = 'meshes/' + $model }
                $matches = @($consumed | Where-Object { $_.requestedPath.Replace('\', '/').ToLowerInvariant() -eq $model })
                if ($matches.Count -eq 0) { throw "No exact-byte read evidence for installed model $model" }
            }
            foreach ($extension in @('.nif', '.bgsm', '.dds')) {
                if (-not ($consumed | Where-Object { [IO.Path]::GetExtension($_.resolvedPath) -eq $extension })) {
                    throw "No consumed $extension evidence for $generation fixture dependencies"
                }
            }
            if (-not (Select-String -LiteralPath (Join-Path $output "runtime-$generation.stdout.log"), (Join-Path $output "runtime-$generation.stderr.log") -SimpleMatch 'Vulkan validation enabled: VK_LAYER_KHRONOS_validation')) {
                throw "$generation did not confirm enabled Vulkan validation"
            }
            $validation = Select-String -LiteralPath (Join-Path $output "runtime-$generation.stderr.log"), (Join-Path $output "runtime-$generation.stdout.log") -Pattern 'Validation Error|VUID-'
            if ($validation) { throw "$generation emitted Vulkan validation errors" }
            $results.Add(@{ case = "runtime-$generation"; status = 'Pass'; detail = 'Fresh process, installed bodies/instances, bounded simulation, identical consumed package bytes.' })
        }
        $results.Add(@{ case = 'fallout4-E0-E10'; status = 'Blocked'; detail = 'Requires a separately initiated isolated game check; no game installation, settings, or saves changed by this driver.' })
        $manifest.automatedStatus = 'Pass'
    }
} catch {
    $exitStatus = 1
    $manifest.automatedStatus = 'Fail'
    $manifest.error = $_.Exception.Message
    Write-Warning $_.Exception.Message
} finally {
    $manifest.finishedUtc = [DateTime]::UtcNow.ToString('o')
    $manifest.artifacts = @(Get-ChildItem -LiteralPath $output -Recurse -File | ForEach-Object { File-Evidence $_.FullName $output })
    Write-Json $manifest (Join-Path $output 'manifest.json')
    $lines = @("OpenF4 compatibility evidence: $Mode", "Automated checks: $($manifest.automatedStatus)", 'Profile: Not proven (all applicable gates and reviewer approval required)', '')
    $lines += @($results | ForEach-Object { "$($_.case): $($_.status) - $($_.detail)" })
    if ($manifest.Contains('error')) { $lines += "Failure: $($manifest.error)" }
    [IO.File]::WriteAllLines((Join-Path $output 'summary.txt'), $lines)
    Write-Host "Evidence preserved: $output"
}
exit $exitStatus
