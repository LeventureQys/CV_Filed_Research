$ErrorActionPreference = 'Stop'
$base = 'D:\workshop\Processing\multi-device-cascade-host-cpp'
$cl = Join-Path $base 'Document\ChangingLog'
$main = Join-Path $cl '完整更新日志.md'
$utf8 = New-Object System.Text.UTF8Encoding($false)

function Get-Entries([string]$path) {
    $lines = Get-Content -LiteralPath $path -Encoding UTF8
    $idx = @(); for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^## ') { $idx += $i } }
    $out = New-Object System.Collections.ArrayList
    for ($k = 0; $k -lt $idx.Count; $k++) {
        $s = $idx[$k]
        $e = if ($k -lt $idx.Count - 1) { $idx[$k + 1] - 1 } else { $lines.Count - 1 }
        $text = ($lines[$s..$e] -join "`r`n")
        $d = '0000-00-00'
        if ($lines[$s] -match '^## (\d{4}-\d{2}-\d{2})') { $d = $Matches[1] }
        [void]$out.Add([pscustomobject]@{ date = $d; text = $text; bytes = $utf8.GetByteCount($text + "`r`n") })
    }
    return $out.ToArray()
}

$all = @(Get-Entries $main)
"main_entries=$($all.Count) bytes=$(($all | Measure-Object -Property bytes -Sum).Sum)"
$sorted = @($all | Sort-Object date, @{ Expression = { $_.text } })
$cut = 0; $acc = 0
foreach ($e in $sorted) { if ($acc -ge 100 * 1024) { break }; $acc += $e.bytes; $cut++ }
$migrate = @($sorted[0..($cut - 1)])
$keep = @($sorted[$cut..($sorted.Count - 1)])
"migrate=$($migrate.Count) bytes=$acc  keep=$($keep.Count) keep_bytes=$(($keep | Measure-Object -Property bytes -Sum).Sum)"
"range=$($migrate[0].date)..$($migrate[-1].date)"

$stamp = Get-Date -Format 'yyyy-MM-dd'
$header = "# 完整更新日志归档：$($migrate[0].date) 至 $($migrate[-1].date)`r`n`r`n> 本档于 $stamp 按「每档约 100KB」规则，从 完整更新日志.md 最旧端整批迁出（主文件超 100KB 上限触发归档）。`r`n> 条目内容未修改，按日期升序排列。`r`n`r`n"
$body = (($migrate | ForEach-Object { $_.text }) -join "`r`n`r`n") + "`r`n"
$arcPath = Join-Path $cl "Archived\完整更新日志_$($migrate[0].date)至$($migrate[-1].date).md"
[System.IO.File]::WriteAllText($arcPath, $header + $body, $utf8)
"archive=$(Split-Path $arcPath -Leaf) bytes=$((Get-Item $arcPath).Length)"

$lines = Get-Content -LiteralPath $main -Encoding UTF8
$firstEntryLine = ($lines | Select-String -Pattern '^## ' -Encoding UTF8 | Select-Object -First 1).LineNumber
$head = (($lines[0..($firstEntryLine - 2)] -join "`r`n") + "`r`n")
$keepBody = (($keep | ForEach-Object { $_.text }) -join "`r`n`r`n") + "`r`n"
[System.IO.File]::WriteAllText($main, $head + "`r`n" + $keepBody, $utf8)
"main_bytes=$((Get-Item $main).Length) main_entries=$((Select-String -LiteralPath $main -Pattern '^## ' -Encoding UTF8).Count)"
"arc_exists=$(Test-Path $arcPath)"
"DONE"
