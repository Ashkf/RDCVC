$work_dir = "D:\OneDrive\01 WORK\# DampersClusterControl\NeuralOpt\";

Clear-Host; # 清空控制台
Set-Location $work_dir; # 设置工作目录
Write-Host "Current directory: $PWD" -ForegroundColor Green; # 输出当前目录

if (-not(Test-Path -Path "./NN_test.py" -PathType Leaf))
{
    # 文件不存在，输出提示信息
    Write-Host "File NN_test.py not found in parent directory" -ForegroundColor Red
}

conda activate ml-research
python NN_test.py `
    --is_test `
    --model_type mlp-L5 `
    --data_type bim `
    --save_prefix 1-16-16-16-1_BIM2NN_test `
    --eval_path ./data/SMKNRDC_bim2NN.db `
    --load_model_path ./checkpoints/NN/mlp-L5_1-16-16-16-1_BIM2NN_2023-05-07T12-19-27/final_model.pth `
#    --from_checkpoint path `
Write-Host "NN_test.ps1 finished." -ForegroundColor Green
