$work_dir = "D:\OneDrive\01 WORK\# DampersClusterControl\NeuralOpt.release\";

Clear-Host; # 清空控制台
Set-Location $work_dir; # 设置工作目录
Write-Host "Current directory: $PWD" -ForegroundColor Green; # 输出当前目录

if (-not(Test-Path -Path "./GA_run.py" -PathType Leaf))
{
    # 文件不存在，输出提示信息
    Write-Host "File GA_run.py not found in parent directory" -ForegroundColor Red
}

conda activate ml-research
python GA_run.py `
    --nind 700 `
    --maxgen 150 `
    --nn_path "checkpoints/NN/mlp-L12_pretrain_BS128_LR0.003_EP2000_2023-05-30T23-24-05/"

Write-Host "GA_run.ps1 finished." -ForegroundColor Green
