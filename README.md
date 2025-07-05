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
├── demo_picture/ #测试文件夹
│   ├── batches/
│   │   └── (请在此处放入用于批量测试的图片) #多张图片测试
│   └── single_demo.JPEG  # 单张图片测试
├── augmentor_module.py   # 数据增强核心模块
├── augmentor_demo.py         # 演示脚本
└── config.json           # 增强流程配置文件
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

**模板 config.json：**

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

**示例 config.json：**

``` json
{
  // 根对象
  "pipeline": [ // 增强流程，一个包含多个操作的列表
    // --- 第一个操作：随机水平翻转 ---
    {
      "name": "random_horizontal_flip", // 操作名称
      "params": {                      // 操作参数
        "p": 0.5                     // 定义翻转的概率为50%
      }
    },
    // --- 第二个操作：随机旋转 ---
    {
      "name": "random_rotation",        // 操作名称
      "params": {
        "angle_range": [-20, 20]     // 定义旋转角度范围在-20到+20度之间
      }
    },
    // --- 第三个操作：对比度调整 (使用默认参数) ---
    {
      "name": "contrast_adjust"         // 操作名称
      // 此处省略了 "params"，将使用模块中定义的默认对比度调整范围
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

| 版本号 | 日期      | 更新内容摘要 |
| :----- | --------- | ------------ |
| v1.0.0 | 2025-07-5 | initial push |
|        |           |              |
|        |           |              |