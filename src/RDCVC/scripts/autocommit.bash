#!/bin/bash

set -e

tmp_folder_path="~/DockerImageReleaseTemp"

clear

# 检查临时文件夹是否存在
if [ -d "$tmp_folder_path" ]; then
  rm -rf "$tmp_folder_path"  # 存在则删除文件夹
fi

# 创建临时文件夹并进入
mkdir -p "$tmp_folder_path"
cd "$tmp_folder_path" || exit 1

echo "===================== 现存的 Images ====================="
docker images

read -rp "请输入 Git 代码库（默认为 git@github.com:Ashkf/NeuralOpt.git）:" git_repo_url
git_repo_url=${git_repo_url:-"git@github.com:Ashkf/NeuralOpt.git"}
read -rp "请输入 base image 的 REPOSITORY（默认为 ashkf/neural_opt）: " base_REPOSITORY
base_REPOSITORY=${base_REPOSITORY:-"ashkf/neural_opt"}
read -rp "请输入 base image 的 tag（例如 v0.6.0）: " base_tag
read -rp "请输入最新的镜像 tag: " new_tag
read -rp "作者（默认为 ashkf）：" author
author=${author:-"ashkf"}
read -rp "新镜像的message：" message

echo "===================== 拉取最新版本代码 ====================="
git clone --depth 1 --branch master "$git_repo_url"  # 拉取最新版本代码

# 运行镜像并取得容器ID
container_id=$(docker run -itd --name "image_commit" "$base_REPOSITORY:$base_tag")

# 拷贝代码到镜像内
docker cp /home/kai/DockerImageReleaseTemp/NeuralOpt "$container_id:/workspace/src"

# 提交镜像修改
docker commit -a="$author" -m "$message" "$container_id" "$base_REPOSITORY:$new_tag"

# 停止并删除容器
docker kill "$container_id" && docker rm "$container_id"

# 清理临时文件夹
rm -rf "$tmp_folder_path"

# 推送新镜像到 Docker Hub
docker push "$base_REPOSITORY:$new_tag"

clear
echo "镜像构建完成并推送到 Docker Hub，镜像名为: $base_REPOSITORY:$new_tag"