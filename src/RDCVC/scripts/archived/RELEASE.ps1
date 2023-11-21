# 这是个 powershell 脚本，用于发布项目

# 用法：在项目根目录下执行 powershell .\scripts\RELEASE.ps1
# 作用：将 NeuralOpt 项目 发布至 NeuralOpt.release 目录下

# 需要发布的文件夹目录列表：
# NeuralOpt
# ├── CoreNN
# ├── CoreGA
# ├── ref
# ├── scripts
# ├── utils
# ├── .gitignore
# ├── README.md
# ├── NN_predict.py
# ├── NN_train.py
# ├── NN_test.py
# └── environment.yml

$dir_list = @(
"CoreNN",
"CoreGA",
"ref",
"scripts",
"utils"
);
$file_list = @(
".gitignore",
"README.md",
"GA_run.py",
"NN_predict.py",
"NN_train.py",
"NN_test.py",
"environment.yml"
);

# 原始代码目录
$work_dir = "D:\OneDrive\01 WORK\# DampersClusterControl\NeuralOpt\";
# 发布目录
$release_dir = "D:\OneDrive\01 WORK\# DampersClusterControl\NeuralOpt.release\";

# ======================== RUN ========================
Clear-Host; # 清空控制台
Set-Location $work_dir; # 设置工作目录
Write-Host "Current directory: $PWD" -ForegroundColor Green; # 输出当前目录

# 检查发布目录是否存在, 存在则提示
if (Test-Path -Path $release_dir -PathType Container)
{
    Write-Host "Release directory already exists, backing..." -ForegroundColor Red;
    # 复制备份为 NeuralOpt.release.bak 目录，该目录与 release_dir 同级，尾缀为当前时间
    $release_dir_bak = $release_dir.Substring(0, $release_dir.Length - 1) + ".bak" + (Get-Date -Format "yyyyMMddHHmmss");
    Copy-Item -Path $release_dir -Destination $release_dir_bak -Recurse -Force;
    Write-Host "Release directory backed up to $release_dir_bak" -ForegroundColor Yellow;
}

# 将 $dir_list 中的文件夹及其子目录复制到 $release_dir 中，若已存在的文件夹则覆盖
foreach ($dir in $dir_list)
{
    $src = $work_dir + $dir;
    $dst = $release_dir;
    Copy-Item -Path $src -Destination $dst -Recurse -Force;
}
Write-Host "Directories copied." -ForegroundColor Green;

# 复制文件
foreach ($file in $file_list)
{
    $src = $work_dir + $file;
    $dst = $release_dir + $file;
    Copy-Item -Path $src -Destination $dst -Force;
}
Write-Host "Files copied." -ForegroundColor Green;

# 输出提示信息
Write-Host "RELEASE.ps1 finished." -ForegroundColor Green;