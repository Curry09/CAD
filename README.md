# 文本 to CAD self-refine system

这个项目实现了对CADPrompt数据集中CAD模型的自动验证功能，可以比较生成的3D模型与Ground Truth的几何属性。通过迭代式自我纠正，提高cad模型产生的正确率和效果。
我们使用了：https://github.com/Kamel773/CAD_Code_Generation 中的数据集

## 功能特性

- 🔍 **自动验证**: 执行Python CADQuery代码并生成3D模型
- 📊 **属性比较**: 比较体积、表面积、顶点数、面数等几何属性
- 📈 **相似度评分**: 计算生成模型与Ground Truth的相似度分数
- 📁 **批量处理**: 支持验证整个数据集的所有样本
- 📋 **详细报告**: 生成包含详细比较结果的JSON报告

## 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖包：
- `numpy`: 数值计算
- `trimesh`: 3D网格处理
- `cadquery`: CAD建模库
- `pathlib`: 路径处理

## 使用方法

### 1. 验证单个样本

```python
from main import CADModelValidator

validator = CADModelValidator()
result = validator.validate_sample("../CAD_Code_Generation/CADPrompt/00000007")
print(f"验证分数: {result['comparison']['overall_score']:.3f}")
```

### 2. 验证整个数据集

```python
validator = CADModelValidator()
results = validator.validate_dataset("../CAD_Code_Generation/CADPrompt")
print(f"通过率: {results['summary']['pass_rate']:.2%}")
```

### 3. 运行完整验证

```bash
cd code
python main.py
```

### 4. 运行测试

```bash
python test_validation.py
```

## 验证指标

### 几何属性比较
- **体积** (Volume): 3D模型的体积
- **表面积** (Surface Area): 模型表面的总面积
- **顶点数** (Vertices): 网格中的顶点数量
- **面数** (Faces): 网格中的三角面数量

### 评分机制
- **相对误差**: `|生成值 - 真值| / |真值|`
- **匹配判定**: 相对误差 < 10% 视为匹配
- **总体分数**: 所有属性匹配分数的平均值
- **验证通过**: 总体分数 > 0.8 (80%)

## 输出格式

### 单个样本验证结果
```json
{
  "sample_dir": "样本目录路径",
  "generated_properties": {
    "volume": 0.367860,
    "surface_area": 4.501090,
    "num_vertices": 72,
    "num_faces": 140
  },
  "comparison": {
    "overall_score": 0.95,
    "matches": {
      "volume": true,
      "surface_area": true
    },
    "relative_errors": {
      "volume": 0.02,
      "surface_area": 0.01
    }
  },
  "validation_passed": true
}
```

### 数据集验证摘要
```json
{
  "summary": {
    "total_samples": 200,
    "valid_samples": 195,
    "passed_samples": 180,
    "pass_rate": 0.92,
    "average_score": 0.89,
    "score_std": 0.15
  }
}
```

## 文件结构

```
code/
├── main.py              # 主验证程序
├── test_validation.py   # 测试脚本
├── requirements.txt     # 依赖包列表
├── README.md           # 说明文档
└── validation_results.json  # 验证结果输出
```

## 验证流程

1. **加载Ground Truth**: 读取`Ground_Truth.json`和`Ground_Truth.stl`
2. **执行Python代码**: 运行`Python_Code.py`生成3D模型
3. **计算属性**: 分析生成模型的几何属性
4. **比较分析**: 与Ground Truth进行对比
5. **生成报告**: 输出详细的验证结果

## 故障排除

### 常见问题

1. **CADQuery安装失败**
   ```bash
   # 使用conda安装
   conda install -c conda-forge cadquery
   ```

2. **STL文件加载错误**
   - 检查文件路径是否正确
   - 确认STL文件格式有效

3. **Python代码执行失败**
   - 检查代码语法
   - 确认所有依赖已安装

### 调试模式

启用详细日志输出：
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 扩展功能

### 自定义验证指标
可以通过继承`CADModelValidator`类来添加新的验证指标：

```python
class CustomValidator(CADModelValidator):
    def custom_metric(self, mesh1, mesh2):
        # 实现自定义比较逻辑
        pass
```

### 可视化支持
集成3D可视化库来显示模型对比：

```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# 添加可视化代码
```

## 贡献指南

欢迎提交Issue和Pull Request来改进这个验证系统！

## 许可证

MIT License 
