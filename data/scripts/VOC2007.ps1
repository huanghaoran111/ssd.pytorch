# 设置开始时间
$start = Get-Date

# 处理可选的下载目录
if ($args.Count -eq 0) {
    # 导航到用户目录下的data文件夹
    Write-Host "Navigating to ~/data/ ..."
    New-Item -ItemType Directory -Force -Path "$HOME\data"
    Set-Location "$HOME\data"
} else {
    # 检查目录是否有效
    if (-Not (Test-Path -Path $args[0])) {
        Write-Host "$args[0] is not a valid directory"
        exit
    }
    Write-Host "Navigating to" $args[0] "..."
    Set-Location $args[0]
}

# 下载数据
Write-Host "Downloading VOC2007 trainval ..."
Invoke-WebRequest -Uri "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar" -OutFile "VOCtrainval_06-Nov-2007.tar"

Write-Host "Downloading VOC2007 test data ..."
Invoke-WebRequest -Uri "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar" -OutFile "VOCtest_06-Nov-2007.tar"

Write-Host "Done downloading."

# 解压数据 
Write-Host "Extracting trainval ..."
& tar -xvf VOCtrainval_06-Nov-2007.tar

Write-Host "Extracting test ..."
& tar -xvf VOCtest_06-Nov-2007.tar

# 删除压缩包
Write-Host "Removing tar files ..."
Remove-Item VOCtrainval_06-Nov-2007.tar
Remove-Item VOCtest_06-Nov-2007.tar

# 计算并显示运行时间
$end = Get-Date
$runtime = $end - $start
Write-Host "Completed in $($runtime.TotalSeconds) seconds"
