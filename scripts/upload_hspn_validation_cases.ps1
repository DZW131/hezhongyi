param(
    [string]$Manifest = "reports\hspn_scoring_validation_20261014\unique_case_manifest.csv",
    [string]$Server = "duyanhong@10.15.20.178",
    [string]$HostKeyAlias = "10.15.20.77",
    [string]$RemoteRoot = "/home/duyanhong/Dataspace/HZY/raw_hspn_validation_20261014/task2_94",
    [int]$PartitionCount = 1,
    [int]$PartitionIndex = 0
)

$ErrorActionPreference = "Stop"
$allRows = @(Import-Csv -LiteralPath $Manifest | Where-Object { $_.source_dataset -eq "intrinsic_cell_task2_94" })
if ($PartitionCount -lt 1 -or $PartitionIndex -lt 0 -or $PartitionIndex -ge $PartitionCount) {
    throw "PartitionIndex must be in [0, PartitionCount)."
}
$rows = @(
    for ($index = 0; $index -lt $allRows.Count; $index += 1) {
        if (($index % $PartitionCount) -eq $PartitionIndex) { $allRows[$index] }
    }
)
if ($rows.Count -eq 0) {
    throw "No intrinsic_cell_task2_94 rows found in $Manifest"
}

$sshOptions = @("-o", "BatchMode=yes", "-o", "HostKeyAlias=$HostKeyAlias")
$uploaded = 0
$skipped = 0

foreach ($row in $rows) {
    $czi = [System.IO.FileInfo]$row.czi_path
    $db = [System.IO.FileInfo](Join-Path $czi.DirectoryName "slice.db")
    if (-not $czi.Exists) { throw "Missing CZI: $($czi.FullName)" }
    if (-not $db.Exists) { throw "Missing annotation DB: $($db.FullName)" }

    $remoteDir = "$RemoteRoot/$($row.case_id)"
    & ssh @sshOptions $Server "mkdir -p '$remoteDir'"
    if ($LASTEXITCODE -ne 0) { throw "Could not create $remoteDir" }

    foreach ($file in @($czi, $db)) {
        $remotePath = "$remoteDir/$($file.Name)"
        $remoteSize = & ssh @sshOptions $Server "test -f '$remotePath' && stat -c %s '$remotePath' || echo 0"
        if ($LASTEXITCODE -ne 0) { throw "Could not inspect $remotePath" }
        if ([int64]$remoteSize -eq $file.Length) {
            $skipped += 1
            Write-Host "[skip] $($row.case_id) $($file.Name)"
            continue
        }
        Write-Host "[copy] $($row.case_id) $($file.Name) ($([math]::Round($file.Length / 1MB, 1)) MB)"
        & scp @sshOptions -- $file.FullName "${Server}:$remoteDir/"
        if ($LASTEXITCODE -ne 0) { throw "Upload failed: $($file.FullName)" }
        $uploaded += 1
    }
}

Write-Host "Upload complete: partition=$PartitionIndex/$PartitionCount copied=$uploaded skipped=$skipped cases=$($rows.Count)"
