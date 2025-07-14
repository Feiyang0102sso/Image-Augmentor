# 图像数据增强模块 (Image Augmentor)

本项目旨在提供一个易于扩展和使用的工具，以在计算机视觉任务中提升模型的性能和泛化能力。

## 功能特性

- **配置**：通过 config.json 文件定义和修改数据增强的操作序列和参数。
- **多种增强操作**：内置多种常用的增强算法，涵盖几何变换与色彩变换。
- **两种模式**：支持对单张图像进行精细化对比，也支持对整个文件夹的图片进行批量处理。
- **结果可视化**：自动生成处理前后的对比图，方便直观地查看增强效果。

##  项目结构

以下是项目的核心文件结构：

```bash
augmentor/
├── demo_picture/
│   ├── batches/                  # 用于存放批量测试的图片
│   └── single_demo.JPEG          # 用于单图测试的示例图片
├── augmentor_demo.py             # 测试 / 演示脚本
├── augmentor_module.py           # 调度和加载各增强模块
├── BrightnessAdjust.py           # 亮度调整模块
├── ContrastAdjust.py             # 对比度调整模块
├── RandomAffine.py               # 随机仿射变换模块
├── RandomChoice.py               # 随机选择一个增强模块
├── RandomCrop.py                 # 随机裁剪模块
├── RandomHorizontalFlip.py       # 随机水平翻转模块
├── RandomHue.py                  # 随机色调调整模块
├── RandomRotation.py             # 随机旋转模块
├── RandomSaturation.py           # 随机饱和度调整模块
├── config.json                   # 默认配置文件
├── config_new.json               # 另一个示例配置，用来测试新功能
├── config_random_apply.json      # 另一个示例配置，用于测试概率应用
├── config_random_choice.json     # 另一个示例配置，用于测试随机选择
├── demo_log.txt                  # 演示脚本生成的日志文件
├── requirements.txt              # 项目依赖库
└── README.md                     # 项目说明文件
```

## 安装与依赖

1. Python 3.10

2. 安装所需的依赖库：

    ```
    pip install -r requirements.txt
    ```

## 使用步骤

按照以下步骤来运行数据增强演示。

### 1. 准备图片

- **单张图片测试**：将需要测试的单张图片放在 demo_picture/ 目录下，默认为 single_demo.JPEG。
- **批量图片测试**：将所有需要批量处理的图片放入 demo_picture/batches/ 文件夹内。

### 2. 配置增强流程 (config.json)

打开 config.json 文件，你可以自定义一个由多个操作组成的增强流水线（pipeline）。pipeline 是一个列表，列表中的每个对象代表一个增强操作。

#### 模板 一般流水线 config.json：

```json
{
  "pipeline": [
    {
      "name": "操作名称1",
      "params": {
        "参数A": "值1",
        "参数B": "值2"
      }
    },
    {
      "name": "操作名称2"
    }
  ]
}
```

#### 模板 概率应用 机制 config_random_apply.json：

该模式允许用户对每个增强操作，允许设置一个执行概率，只有当随机数满足条件时才应用该操作。

``` json
{
  "pipeline": [
    {
      "name": "操作名称1",
      "params": {
        "参数A": "值1",
        "参数B": "值2"
      },
      "prob": "概率 若为0.8则表示有80%概率执行此操作"
    },
    {
      "name": "操作名称2",
      "prob": 0.5  
    }
  ]
}

```

#### 模板 随机选择 机制 config_random_choice.json：

允许在配置文件中定义一组增强操作，模块在执行时从这组操作中随机选择一个或多个进行应用。

被选择的组将被放在名为RandomChoice的节点中

``` json
{
  "pipeline": [
    {
      "name": "RandomChoice",
      "params": {
        "choices": [
          {
            "name": "RandomHorizontalFlip",
            "params": { "p": 1 }
          },
          {
            "name": "RandomRotation",
            "params": { "angle_range": [50, 150] }
          },
          {
            "name": "BrightnessAdjust",
            "params": { "delta_range": [20, 60] }
          },
          {
            "name": "ContrastAdjust",
            "params": { "factor_range": [1.5, 2.5] }
          }
        ]
      }
    }
  ]
}

```



#### 支持的操作 (Supported Operations)

| name                   | 描述                               | 参数 (params)                                                |
| ---------------------- | ---------------------------------- | ------------------------------------------------------------ |
| random_horizontal_flip | 以指定概率对图像进行随机水平翻转。 | p: 翻转概率，浮点数，默认 0.5。                              |
| random_rotation        | 在指定角度范围内随机旋转图像。     | angle_range: 角度范围元组 (min, max)，默认 (-15, 15)。       |
| brightness_adjust      | 随机调整图像亮度。                 | delta_range: 亮度变化范围元组 (min, max)，默认 (-30, 30)。   |
| contrast_adjust        | 随机调整图像对比度。               | factor_range: 对比度因子范围元组 (min, max)，默认 (0.8, 1.2)。 |
| random_crop            | 从图像中随机裁剪指定大小区域。     | crop_size: 裁剪尺寸元组 (height, width)，例如 (200, 200)。   |
| random_affine          | 对图像应用随机仿射平移变换。       | max_translate: 最大平移比例，浮点数，默认 0.2（表示最大移动 20%）。 |
| random_saturation      | 随机调整图像饱和度。               | lower: 下界缩放因子，默认 0.5；upper: 上界缩放因子，默认 1.5。 |
| random_hue             | 随机调整图像色相。                 | delta: 色相偏移范围（度数），默认 18。                       |

*注意：如果某个操作不提供 params 字段，将使用其默认参数。*

### 3. 运行演示脚本

打开终端，进入 augmentor 目录，通过命令行参数指定运行模式。

#### 模式一：单张图片增强

此模式会处理 demo_picture/ 下的单张图片，并将原始图与流水线中每一步操作后的效果图并排显示和保存。

```bash
python multi_demo.py single
```

运行后，会生成并打开一张名为 single_augmented_comparison.png 的对比图。

#### 模式二：批量图片增强

此模式会处理 demo_picture/batches/ 文件夹下的所有图片，并为每张图片生成一行对比图，最终将所有结果合并在一张大图中。

```bash
python multi_demo.py batch
```

运行后，会在 demo_picture/batches/ 目录下生成一张名为 augmented_comparison_grid.png 的网格对比图。

#### 模式三：自定义config路径

支持自定义config路径，若无config参数，则将用默认的config.json进行配置

``` bash
python augmentor_demo.py single --config custom_config.json  # Custom config file
python augmentor_demo.py batch --config custom_config.json   # Custom config file
```



## 上传至github

### 初次上传

``` bash
git init #初始化

#配置
git config --global user.name "Feiyang0102sso"
git config --global user.email "wufeiyang2@gmail.com"

git add . #添加到缓存区
git commit -m "Initial commit" #添加到本地仓库
git remote add origin https://github.com/Feiyang0102sso/Image-Augmentor.git #添加远程仓库
git branch -M main #将分支重命名为 main
git push -u origin main #git push -u origin main

```



### 更新

``` bash
git status
#检测当前状态
git add --all
# 上传到本地的一个预备区
git commit -m "更新"
# 上传到本地仓库
git push
# 上传到网络上的github
```

## 更新记录

| 版本号 | 日期       | 更新内容摘要                                                 |
| :----- | ---------- | ------------------------------------------------------------ |
| v1.0.0 | 2025-07-5  | initial push                                                 |
| v1.1.0 | 2025-07-8  | 允许进行图片批处理<br />优化代码避免臃肿                     |
| v1.2.0 | 2025-07-14 | 新增log功能<br />新增4中方法，允许随机应用，随即选择<br />新增自定义config文件功能 |