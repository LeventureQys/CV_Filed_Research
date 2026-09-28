$ErrorActionPreference = 'Stop'
$base = 'D:\workshop\Processing\multi-device-cascade-host-cpp'
$main = Join-Path $base 'Document\ChangingLog\完整更新日志.md'
$arcDir = Join-Path $base 'Document\ChangingLog\Archived'
$utf8 = New-Object System.Text.UTF8Encoding($false)

$lines = Get-Content -LiteralPath $main -Encoding UTF8
$idx = @(); for ($i = 0; $i -lt $lines.Count; $i++) { if ($lines[$i] -match '^## ') { $idx += $i } }

$sizes = @()
for ($k = 0; $k -lt $idx.Count; $k++) {
    $s = $idx[$k]
    $e = if ($k -lt $idx.Count - 1) { $idx[$k + 1] - 1 } else { $lines.Count - 1 }
    $sizes += $utf8.GetByteCount((($lines[$s..$e] -join "`r`n") + "`r`n"))
}

$keep = 24   # 主文件保留最新 24 个条目
$startLine = $idx[$keep]
$entries = @()
for ($k = $keep; $k -lt $idx.Count; $k++) {
    $s = $idx[$k]
    $e = if ($k -lt $idx.Count - 1) { $idx[$k + 1] - 1 } else { $lines.Count - 1 }
    $d = '0000-00-00'
    if ($lines[$s] -match '^## (\d{4}-\d{2}-\d{2})') { $d = $Matches[1] }
    $entries += [pscustomobject]@{ date = $d; text = ($lines[$s..$e] -join "`r`n") }
}

$sorted = $entries   # 保持主文件原顺序（近期条目本身即非严格日期序，不重排）
$minD = ($entries | ForEach-Object { $_.date } | Sort-Object)[0]
$maxD = ($entries | ForEach-Object { $_.date } | Sort-Object)[-1]
$body = (($sorted | ForEach-Object { $_.text }) -join "`r`n`r`n") + "`r`n"
"migrated_entries=$($sorted.Count) min=$minD max=$maxD bytes=$($utf8.GetByteCount($body))"

$stamp = Get-Date -Format 'yyyy-MM-dd'
$arcHeader = "# 完整更新日志归档：$minD 至 $maxD`r`n`r`n> 本档于 $stamp 按「每档约 100KB」规则，从 完整更新日志.md 最旧端整批迁出（主文件超 100KB 上限触发归档）。`r`n> 条目内容未修改，按主文件原顺序保存。`r`n`r`n"
$arcName = "完整更新日志_${minD}至${maxD}.md"
[System.IO.File]::WriteAllText((Join-Path $arcDir $arcName), $arcHeader + $body, $utf8)
"archive=$arcName"

# 主文件保留首部 + 最新条目
$keepText = (($lines[0..($startLine - 1)] -join "`r`n") + "`r`n")
[System.IO.File]::WriteAllText($main, $keepText, $utf8)
"new_main_bytes=$($utf8.GetByteCount($keepText))"
