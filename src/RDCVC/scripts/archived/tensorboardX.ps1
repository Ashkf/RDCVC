$work_dir = "D:\OneDrive\01 WORK\# DampersClusterControl\NeuralOpt\";

Clear-Host; # 清空控制台
Set-Location $work_dir; # 设置工作目录
Write-Host "Current directory: $PWD" -ForegroundColor Green; # 输出当前目录

conda activate ml-research
Write-Host "conda env activated" -ForegroundColor Green

# 激活 tensorboardX
tensorboard --logdir checkpoints\

