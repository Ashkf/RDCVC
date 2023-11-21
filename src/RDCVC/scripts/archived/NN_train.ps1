$work_dir = "D:\OneDrive\01 WORK\# DampersClusterControl\NeuralOpt\";

Clear-Host; # 清空控制台
Set-Location $work_dir; # 设置工作目录
Write-Host "Current directory: $PWD" -ForegroundColor Green; # 输出当前目录

if (-not(Test-Path -Path "./NN_train.py" -PathType Leaf))
{
    # 文件不存在，输出提示信息
    Write-Host "File NN_train.py not found in parent directory" -ForegroundColor Red
}

conda activate ml-research
python NN_train.py `
    --model_type cpn-mtl-mlp_6_36_3_36 `
    --data_type bim_mtl `
    --train_path ./data/SMKNRDC_bim2NN_20230708.db `
    --eval_path ./data/SMKNRDC_bim2NN_20230708.db `
    --lr 5E-05 `
    --batch_size 128 `
    --epochs 2000 `
    --save_every 100 `
    --resume_path "D:\OneDrive\01 WORK\# DampersClusterControl\NeuralOpt\checkpoints\NN\cpn-mtl-mlp_6_36_3_36_resume_BS128_LR0.0001_EP1500_2023-07-20T15-04-49\ckps\ckp_E1400-B0000.pth"
#--seed 99918 `
#--load_model_path XXX `
#--resume_path XXX `
Write-Host "NN_train.ps1 finished." -ForegroundColor Green
