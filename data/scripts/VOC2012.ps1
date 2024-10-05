# Ellis Brown - PowerShell Version

$start = Get-Date

# 处理可选的下载目录
if ($args.Count -eq 0) {
    # 导航到 ~/data
    Write-Host "Navigating to ~/data/ ..."
    $downloadDir = "$HOME\data"
    if (-Not (Test-Path $downloadDir)) {
        New-Item -ItemType Directory -Path $downloadDir
    }
    Set-Location $downloadDir
} else {
    # 检查目录是否有效
    if (-Not (Test-Path -Path $args[0])) {
        Write-Host "$($args[0]) is not a valid directory"
        exit
    }
    Write-Host "Navigating to $($args[0]) ..."
    Set-Location $args[0]
}

# 下载数据
Write-Host "Downloading VOC2012 trainval ..."
Invoke-WebRequest -Uri "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar" -OutFile "VOCtrainval_11-May-2012.tar"
Write-Host "Done downloading."

# 解压数据
Write-Host "Extracting trainval ..."
& tar -xvf VOCtrainval_11-May-2012.tar

# 删除 tar 文件
Write-Host "Removing tar file ..."
Remove-Item VOCtrainval_11-May-2012.tar

# 计算运行时间
$end = Get-Date
$runtime = $end - $start
Write-Host "Completed in $($runtime.TotalSeconds) seconds"
