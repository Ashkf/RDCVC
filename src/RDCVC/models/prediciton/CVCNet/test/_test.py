"""
    实际使用时需要将训练好的模型上在输入数据上运行，这里以测试集的数据为例，实际情况下只需要初始化模型之后将视频流中的图像帧作为模型的输入即可。 
    
    torch.no_grad() 
        停止autograd模块的工作，不计算和储存梯度，一般在用训练好的模型跑测试集时使用，因为测试集时不需要计算梯度更不会更新梯度。使用后可以加速计算时间，节约gpu的显存
"""


test_dataset = MyDataset(test_dataset_path, transforms)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False,
                                          num_workers=2)
model = Network().cuda()
# 对磁盘上的pickle文件进行解包 将gpu训练的模型加载到cpu上
model.load_stat_dict(torch.load(model_path, map_location=torch.device('cpu')))
mocel.eval()

with torch.no_grad():
    for batch in test_loader:
        test_images. test_gts = test_batch[0].cuda(), test_batch[1].cuda()
        test_preds = model(test_iamges)
        # 保存模型输出的图片
