import json
import numpy as np
import tempfile
import os
from pathlib import Path
from typing import Dict, Tuple, Any
import logging
import subprocess
import sys
import re
from openai import OpenAI
import uuid
import time
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

# 尝试导入psutil，如果没有则使用备用方案
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("Warning: psutil not available, performance monitoring will be limited")

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_to_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def convert_numpy_types(obj):
    """递归转换numpy类型为Python原生类型，用于JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj


class CADModelValidator:
    """CAD模型验证器，用于比较模型生成的代码与Ground Truth"""
    
    def __init__(self):
        self.tolerance = 1e-6  # 数值比较容差
        
    def load_ground_truth(self, sample_dir: str) -> Dict[str, Any]:
        """加载Ground Truth数据"""
        sample_path = Path(sample_dir)
        
        # 加载Ground Truth JSON
        gt_json_path = sample_path / "Ground_Truth.json"
        if gt_json_path.exists():
            with open(gt_json_path, 'r') as f:
                gt_data = json.load(f)
        else:
            gt_data = {}
            
        # 检查Ground Truth STL是否存在
        gt_stl_path = sample_path / "Ground_Truth.stl"
        stl_exists = gt_stl_path.exists()
        
        if stl_exists:
            logger.info(f"找到Ground Truth STL: {gt_stl_path}")
        else:
            logger.warning(f"未找到Ground Truth STL: {gt_stl_path}")
                
        return {
            "json_data": gt_data,
            "stl_exists": stl_exists,
            "stl_path": gt_stl_path
        }
    
    def execute_python_code(self, code_path: str) -> Dict[str, Any]:
        """执行Python代码生成3D模型"""
        try:
            # 读取Python代码
            with open(code_path, 'r') as f:
                code_content = f.read()
            
            logger.info(f"执行Python代码: {code_path}")
            logger.info(f"代码内容:\n{code_content}")
            
            # 创建临时目录执行代码
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_code_path = os.path.join(temp_dir, "temp_code.py")
                temp_stl_path = os.path.join(temp_dir, "generated_model.stl")
                
                # 修改代码中的导出路径
                modified_code = code_content.replace(
                    "'Ground_Truth.stl'", 
                    f"'{temp_stl_path}'"
                ).replace(
                    '"Ground_Truth.stl"', 
                    f'"{temp_stl_path}"'
                )
                
                # 写入修改后的代码
                with open(temp_code_path, 'w') as f:
                    f.write(modified_code)
                
                # 使用subprocess执行代码，避免直接exec的问题
                try:
                    result = subprocess.run([
                        sys.executable, temp_code_path
                    ], capture_output=True, text=True, timeout=60)
                    
                    if result.returncode != 0:
                        logger.error(f"代码执行失败: {result.stderr}")
                        return {"success": False, "error": result.stderr}
                    
                    # 检查生成的STL文件
                    if os.path.exists(temp_stl_path):
                        # 获取文件大小作为基本验证
                        file_size = os.path.getsize(temp_stl_path)
                        logger.info(f"成功生成STL文件，大小: {file_size} bytes")
                        
                        # 尝试读取STL文件的基本信息
                        stl_info = self.analyze_stl_file(temp_stl_path)
                        
                        return {
                            "success": True,
                            "file_size": file_size,
                            "stl_info": stl_info
                        }
                    else:
                        logger.error("Python代码执行后未找到生成的STL文件")
                        return {"success": False, "error": "未生成STL文件"}
                        
                except subprocess.TimeoutExpired:
                    logger.error("代码执行超时")
                    return {"success": False, "error": "执行超时"}
                    
        except Exception as e:
            logger.error(f"执行Python代码失败: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_stl_file(self, stl_path: str) -> Dict[str, Any]:
        """分析STL文件的详细信息，包括所有几何属性"""
        try:
            # 首先检查是否是ASCII格式的STL
            try:
                with open(stl_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    if first_line.startswith('solid'):
                        # ASCII格式STL
                        return self.analyze_ascii_stl(stl_path)
            except UnicodeDecodeError:
                pass
            
            # 如果不是ASCII格式，尝试作为二进制格式处理
            return self.analyze_binary_stl(stl_path)
                
        except Exception as e:
            logger.error(f"分析STL文件失败: {e}")
            return {"error": str(e)}
    
    def calculate_detailed_properties(self, vertices: np.ndarray, facets: list = None) -> Dict[str, Any]:
        """计算详细的几何属性"""
        if len(vertices) == 0:
            return {"error": "无顶点数据"}
        
        # 基本统计
        num_vertices = len(vertices)
        unique_vertices = np.unique(vertices.reshape(-1, 3), axis=0)
        num_unique_vertices = len(unique_vertices)
        
        # 计算边界框
        min_coords = np.min(unique_vertices, axis=0)
        max_coords = np.max(unique_vertices, axis=0)
        bbox_size = max_coords - min_coords
        bbox_center = (min_coords + max_coords) / 2
        
        # 尺寸信息
        width_mm = float(bbox_size[0])
        height_mm = float(bbox_size[1]) 
        depth_mm = float(bbox_size[2])
        
        # 长宽比计算
        aspect_ratio_xy = float(width_mm / height_mm) if height_mm != 0 else float('inf')
        aspect_ratio_xz = float(width_mm / depth_mm) if depth_mm != 0 else float('inf')
        aspect_ratio_yz = float(height_mm / depth_mm) if depth_mm != 0 else float('inf')
        
        # 体积和表面积估算
        estimated_volume = self.estimate_volume_from_bbox(bbox_size)
        estimated_surface_area = self.estimate_surface_area_from_bbox(bbox_size)
        
        # 面数和边数估算
        num_facets = len(facets) if facets else num_vertices // 3
        # 对于封闭网格，欧拉公式：V - E + F = 2，所以 E = V + F - 2
        estimated_edges = num_unique_vertices + num_facets - 2 if num_facets > 0 else 0
        
        # 判断是否为实体（简单启发式：检查是否有足够的面形成封闭体）
        is_solid = num_facets >= 4 and estimated_volume > 1e-10
        
        return {
            "Number_of_Faces": int(num_facets),
            "Number_of_Unique_Edges": int(max(0, estimated_edges)),
            "Number_of_Vertices": int(num_unique_vertices),
            "Width_mm": width_mm,
            "Height_mm": height_mm,
            "Depth_mm": depth_mm,
            "Bounding_Box_Center": f"Vector ({bbox_center[0]}, {bbox_center[1]}, {bbox_center[2]})",
            "Aspect_Ratio_XY": aspect_ratio_xy,
            "Aspect_Ratio_XZ": aspect_ratio_xz,
            "Aspect_Ratio_YZ": aspect_ratio_yz,
            "Is_Solid": is_solid,
            "Volume_cubic_mm": estimated_volume,
            "Surface_Area_square_mm": estimated_surface_area,
            "bounding_box": {
                "min": min_coords.tolist(),
                "max": max_coords.tolist(),
                "size": bbox_size.tolist(),
                "center": bbox_center.tolist()
            }
        }
    
    def analyze_ascii_stl(self, stl_path: str) -> Dict[str, Any]:
        """分析ASCII格式的STL文件"""
        with open(stl_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 统计基本信息
        lines = content.split('\n')
        facet_lines = [line for line in lines if line.strip().startswith('facet normal')]
        vertex_lines = [line for line in lines if line.strip().startswith('vertex')]
        
        # 提取顶点坐标
        vertices = []
        facets = []
        
        current_facet = []
        for line in vertex_lines:
            parts = line.strip().split()
            if len(parts) >= 4:
                try:
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    vertices.append([x, y, z])
                    current_facet.append([x, y, z])
                    
                    # 每3个顶点组成一个面
                    if len(current_facet) == 3:
                        facets.append(current_facet)
                        current_facet = []
                except ValueError:
                    continue
        
        if vertices:
            vertices = np.array(vertices)
            
            # 计算详细属性
            properties = self.calculate_detailed_properties(vertices, facets)
            properties.update({
                "format": "ASCII",
                "num_facet_lines": len(facet_lines),
                "num_vertex_lines": len(vertex_lines)
            })
            
            return properties
        else:
            return {
                "format": "ASCII",
                "num_facet_lines": len(facet_lines),
                "num_vertex_lines": len(vertex_lines),
                "error": "无法解析顶点坐标"
            }
    
    def analyze_binary_stl(self, stl_path: str) -> Dict[str, Any]:
        """分析二进制格式的STL文件"""
        import struct
        
        with open(stl_path, 'rb') as f:
            # 读取文件头（80字节）
            header = f.read(80)
            
            # 读取三角形数量（4字节）
            triangle_count_data = f.read(4)
            if len(triangle_count_data) < 4:
                return {"error": "文件格式错误：无法读取三角形数量"}
            
            triangle_count = struct.unpack('<I', triangle_count_data)[0]
            
            # 读取所有顶点
            vertices = []
            facets = []
            
            for i in range(triangle_count):
                # 每个三角形：法向量(12字节) + 3个顶点(36字节) + 属性(2字节) = 50字节
                triangle_data = f.read(50)
                if len(triangle_data) < 50:
                    break
                
                # 解析法向量（跳过）
                normal = struct.unpack('<3f', triangle_data[0:12])
                
                # 解析3个顶点
                facet_vertices = []
                for j in range(3):
                    vertex_start = 12 + j * 12
                    vertex_data = triangle_data[vertex_start:vertex_start + 12]
                    if len(vertex_data) >= 12:
                        x, y, z = struct.unpack('<3f', vertex_data)
                        vertices.append([x, y, z])
                        facet_vertices.append([x, y, z])
                
                if len(facet_vertices) == 3:
                    facets.append(facet_vertices)
        
        if vertices:
            vertices = np.array(vertices)
            
            # 计算详细属性
            properties = self.calculate_detailed_properties(vertices, facets)
            properties.update({
                "format": "Binary",
                "triangle_count_from_header": triangle_count
            })
            
            return properties
        else:
            return {
                "format": "Binary",
                "triangle_count_from_header": triangle_count,
                "error": "无法解析顶点坐标"
            }
    
    def estimate_volume_from_bbox(self, bbox_size):
        """从边界框估算体积（假设是圆柱体）"""
        if len(bbox_size) >= 3:
            # 假设是圆柱体：V = π * r² * h
            diameter = max(bbox_size[0], bbox_size[1])
            height = bbox_size[2]
            radius = diameter / 2
            volume = np.pi * radius * radius * height
            return float(volume)
        return 0.0
    
    def estimate_surface_area_from_bbox(self, bbox_size):
        """从边界框估算表面积（假设是圆柱体）"""
        if len(bbox_size) >= 3:
            # 假设是圆柱体：A = 2πr² + 2πrh
            diameter = max(bbox_size[0], bbox_size[1])
            height = bbox_size[2]
            radius = diameter / 2
            surface_area = 2 * np.pi * radius * radius + 2 * np.pi * radius * height
            return float(surface_area)
        return 0.0
    
    def compare_properties(self, generated_result: Dict, gt_data: Dict) -> Dict[str, Any]:
        """比较生成模型和Ground Truth的所有属性"""
        comparison = {
            "matches": {},
            "differences": {},
            "relative_errors": {},
            "overall_score": 0.0,
            "metric_scores": {},
            "detailed_comparison": {}
        }
        
        if not generated_result.get("success", False):
            comparison["error"] = generated_result.get("error", "生成失败")
            return comparison
        
        # 如果Ground Truth JSON存在，使用其中的数据
        if "Ground_Truth" in gt_data.get("json_data", {}):
            gt_props = gt_data["json_data"]["Ground_Truth"]
            gen_info = generated_result.get("stl_info", {})
            
            # 定义所有要比较的指标及其容差
            metrics_config = {
                # 几何计数指标
                "Number_of_Faces": {"tolerance": 0.3, "type": "integer"},
                "Number_of_Unique_Edges": {"tolerance": 0.3, "type": "integer"},
                "Number_of_Vertices": {"tolerance": 0.3, "type": "integer"},
                
                # 尺寸指标
                "Width_mm": {"tolerance": 0.1, "type": "float"},
                "Height_mm": {"tolerance": 0.1, "type": "float"},
                "Depth_mm": {"tolerance": 0.1, "type": "float"},
                
                # 长宽比指标
                "Aspect_Ratio_XY": {"tolerance": 0.2, "type": "float"},
                "Aspect_Ratio_XZ": {"tolerance": 0.3, "type": "float"},
                "Aspect_Ratio_YZ": {"tolerance": 0.3, "type": "float"},
                
                # 体积和表面积
                "Volume_cubic_mm": {"tolerance": 0.2, "type": "float"},
                "Surface_Area_square_mm": {"tolerance": 0.2, "type": "float"},
                
                # 布尔指标
                "Is_Solid": {"tolerance": 0.0, "type": "boolean"}
            }
            
            # 比较各个指标
            metric_scores = []
            
            for metric_name, config in metrics_config.items():
                if metric_name in gt_props and metric_name in gen_info:
                    gt_value = gt_props[metric_name]
                    gen_value = gen_info[metric_name]
                    
                    # 计算相对误差和分数
                    score, rel_error, match = self.compare_single_metric(
                        gt_value, gen_value, config["tolerance"], config["type"]
                    )
                    
                    # 记录详细比较结果
                    comparison["detailed_comparison"][metric_name] = {
                        "ground_truth": gt_value,
                        "generated": gen_value,
                        "relative_error": rel_error,
                        "score": score,
                        "match": match,
                        "tolerance": config["tolerance"],
                        "type": config["type"]
                    }
                    
                    comparison["relative_errors"][metric_name] = rel_error
                    comparison["matches"][metric_name] = match
                    comparison["metric_scores"][metric_name] = score
                    
                    # 添加到总分计算
                    metric_scores.append(score)
                    
                    # 记录差异
                    if config["type"] in ["integer", "float"]:
                        comparison["differences"][metric_name] = {
                            "generated": gen_value,
                            "ground_truth": gt_value,
                            "absolute_diff": abs(gen_value - gt_value) if isinstance(gen_value, (int, float)) and isinstance(gt_value, (int, float)) else "N/A",
                            "relative_error": rel_error
                        }
                    else:
                        comparison["differences"][metric_name] = {
                            "generated": gen_value,
                            "ground_truth": gt_value,
                            "match": match
                        }
            
            # 特殊处理边界框中心（字符串格式）
            if "Bounding_Box_Center" in gt_props and "Bounding_Box_Center" in gen_info:
                gt_center = self.parse_vector_string(gt_props["Bounding_Box_Center"])
                gen_center = self.parse_vector_string(gen_info["Bounding_Box_Center"])
                
                if gt_center is not None and gen_center is not None:
                    center_score, center_error = self.compare_vector(gt_center, gen_center, tolerance=0.1)
                    comparison["detailed_comparison"]["Bounding_Box_Center"] = {
                        "ground_truth": gt_props["Bounding_Box_Center"],
                        "generated": gen_info["Bounding_Box_Center"],
                        "relative_error": center_error,
                        "score": center_score,
                        "match": center_error < 0.1
                    }
                    metric_scores.append(center_score)
            
            # 计算总体分数
            if metric_scores:
                comparison["overall_score"] = np.mean(metric_scores)
                comparison["metrics_compared"] = len(metric_scores)
                comparison["metrics_available"] = len(metrics_config)
            
            # 基本成功分数（如果代码能执行并生成文件）
            if generated_result.get("success", False):
                comparison["execution_success"] = True
                if not metric_scores:  # 如果没有其他比较，给基础分数
                    comparison["overall_score"] = 0.5
        else:
            comparison["error"] = "Ground Truth数据不可用"
        
        return comparison
    
    def compare_single_metric(self, gt_value, gen_value, tolerance: float, metric_type: str) -> tuple:
        """比较单个指标"""
        try:
            if metric_type == "boolean":
                match = gt_value == gen_value
                score = 1.0 if match else 0.0
                rel_error = 0.0 if match else 1.0
                
            elif metric_type in ["integer", "float"]:
                if gt_value == 0:
                    # 避免除零错误
                    if gen_value == 0:
                        rel_error = 0.0
                        score = 1.0
                        match = True
                    else:
                        rel_error = float('inf')
                        score = 0.0
                        match = False
                else:
                    rel_error = abs(gen_value - gt_value) / abs(gt_value)
                    score = max(0.0, 1.0 - rel_error)
                    match = rel_error < tolerance
                    
                # 处理无穷大的情况
                if not np.isfinite(rel_error):
                    rel_error = 1.0
                    score = 0.0
                    match = False
                    
            else:
                # 字符串或其他类型
                match = str(gt_value) == str(gen_value)
                score = 1.0 if match else 0.0
                rel_error = 0.0 if match else 1.0
            
            return float(score), float(rel_error), bool(match)
            
        except Exception as e:
            logger.warning(f"比较指标时出错: {e}")
            return 0.0, 1.0, False
    
    def parse_vector_string(self, vector_str: str) -> list:
        """解析Vector字符串格式，如'Vector (0.375, 0.375, 0.0009765625)'"""
        try:
            # 提取括号内的数字
            import re
            pattern = r'Vector\s*\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)'
            match = re.match(pattern, vector_str.strip())
            if match:
                return [float(match.group(1)), float(match.group(2)), float(match.group(3))]
            return None
        except Exception:
            return None
    
    def compare_vector(self, gt_vector: list, gen_vector: list, tolerance: float = 0.1) -> tuple:
        """比较两个向量"""
        try:
            if len(gt_vector) != len(gen_vector):
                return 0.0, 1.0
            
            # 计算欧几里得距离
            gt_array = np.array(gt_vector)
            gen_array = np.array(gen_vector)
            
            # 计算相对误差（使用向量的模长）
            gt_norm = np.linalg.norm(gt_array)
            if gt_norm == 0:
                gen_norm = np.linalg.norm(gen_array)
                rel_error = 0.0 if gen_norm == 0 else 1.0
            else:
                distance = np.linalg.norm(gt_array - gen_array)
                rel_error = distance / gt_norm
            
            score = max(0.0, 1.0 - rel_error)
            return float(score), float(rel_error)
            
        except Exception:
            return 0.0, 1.0
    
    def validate_sample(self, sample_dir: str) -> Dict[str, Any]:
        """验证单个样本"""
        logger.info(f"开始验证样本: {sample_dir}")
        
        sample_path = Path(sample_dir)
        
        # 加载Ground Truth
        gt_data = self.load_ground_truth(sample_dir)
        
        # 查找Python代码文件
        python_code_path = sample_path / "Python_Code.py"
        if not python_code_path.exists():
            logger.error(f"未找到Python代码文件: {python_code_path}")
            return {"error": "Python代码文件不存在"}
        
        # 执行Python代码生成模型
        generated_result = self.execute_python_code(str(python_code_path))
        
        # 比较属性
        comparison = self.compare_properties(generated_result, gt_data)
        
        # 组装结果
        result = {
            "sample_dir": sample_dir,
            "generated_result": generated_result,
            "ground_truth_available": gt_data["json_data"] != {},
            "comparison": comparison,
            "validation_passed": comparison["overall_score"] > 0.6,  # 60%阈值
            "timestamp": str(np.datetime64('now'))
        }
        
        logger.info(f"验证完成，总体分数: {comparison['overall_score']:.3f}")
        return result
    
    def validate_dataset(self, dataset_dir: str) -> Dict[str, Any]:
        """验证整个数据集"""
        dataset_path = Path(dataset_dir)
        results = {}
        
        # 查找所有样本目录
        sample_dirs = [d for d in dataset_path.iterdir() if d.is_dir() and d.name.isdigit()]
        sample_dirs.sort()
        
        logger.info(f"找到 {len(sample_dirs)} 个样本")
        
        for sample_dir in sample_dirs:
            try:
                result = self.validate_sample(str(sample_dir))
                results[sample_dir.name] = result
            except Exception as e:
                logger.error(f"验证样本 {sample_dir.name} 失败: {e}")
                results[sample_dir.name] = {"error": str(e)}
        
        # 计算总体统计
        valid_results = [r for r in results.values() if "error" not in r]
        if valid_results:
            scores = [r["comparison"]["overall_score"] for r in valid_results]
            passed_count = sum(1 for r in valid_results if r["validation_passed"])
            
            summary = {
                "total_samples": len(sample_dirs),
                "valid_samples": len(valid_results),
                "passed_samples": passed_count,
                "pass_rate": passed_count / len(valid_results) if valid_results else 0,
                "average_score": np.mean(scores) if scores else 0,
                "score_std": np.std(scores) if scores else 0
            }
        else:
            summary = {
                "total_samples": len(sample_dirs),
                "valid_samples": 0,
                "passed_samples": 0,
                "pass_rate": 0,
                "average_score": 0,
                "score_std": 0
            }
        
        return {
            "summary": summary,
            "detailed_results": results
        }

class CADCodeGenerator:

    def __init__(self,max_round=5):
        self.llm=OpenAI(api_key="EMPTY", base_url="http://10.51.6.110/v1/")
        self.max_round=max_round
       
    
    def analyze_code_syntax(self, code: str) -> Dict[str, Any]:
        """静态分析代码语法和结构"""
        import ast
        try:
            tree = ast.parse(code)
            # 分析导入、函数调用、变量等
            imports = []
            function_calls = []
            variables = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend([alias.name for alias in node.names])
                elif isinstance(node, ast.ImportFrom):
                    imports.append(f"from {node.module}")
                elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                    function_calls.append(f"{ast.unparse(node.func)}")
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            variables.append(target.id)
            
            return {
                "valid_syntax": True, 
                "imports": imports,
                "function_calls": function_calls[:10],  # 限制数量
                "variables": variables[:10]
            }
        except SyntaxError as e:
            return {"valid_syntax": False, "error": str(e)}
        except Exception as e:
            return {"valid_syntax": False, "error": f"Analysis error: {str(e)}"}

    def check_cad_patterns(self, code: str) -> Dict[str, Any]:
        """检查CAD代码的常见模式和最佳实践"""
        issues = []
        suggestions = []
        good_patterns = []
        
        # 检查是否导入了cadquery
        if "import cadquery" not in code and "from cadquery" not in code:
            issues.append("Missing cadquery import")
            suggestions.append("Add 'import cadquery as cq'")
        else:
            good_patterns.append("Cadquery properly imported")
        
        # 检查是否有导出语句
        if "export" not in code:
            issues.append("Missing export statement")
            suggestions.append("Add 'cq.exporters.export(part, 'Ground_Truth.stl')'")
        else:
            good_patterns.append("Export statement found")
        
        # 检查常见的CAD操作
        cad_operations = ['Workplane', 'Sketch', 'extrude', 'revolve', 'loft', 'sweep']
        found_operations = [op for op in cad_operations if op in code]
        if found_operations:
            good_patterns.append(f"CAD operations found: {', '.join(found_operations)}")
        
        # 检查变量命名
        if 'part' in code or 'result' in code or 'model' in code:
            good_patterns.append("Good variable naming detected")
        
        return {
            "issues": issues, 
            "suggestions": suggestions,
            "good_patterns": good_patterns,
            "score": max(0, len(good_patterns) - len(issues))
        }

    def validate_stl_file(self, stl_path: str) -> Dict[str, Any]:
        """验证生成的STL文件的完整性"""
        try:
            if not os.path.exists(stl_path):
                return {"valid": False, "error": "STL file not found"}
            
            file_size = os.path.getsize(stl_path)
            if file_size < 84:  # 最小STL文件大小
                return {"valid": False, "error": "File too small to be valid STL"}
            
            import struct
            with open(stl_path, 'rb') as f:
                header = f.read(80)
                triangle_count_data = f.read(4)
                if len(triangle_count_data) < 4:
                    return {"valid": False, "error": "Invalid STL header"}
                
                triangle_count = struct.unpack('<I', triangle_count_data)[0]
                
            expected_size = 80 + 4 + triangle_count * 50  # STL格式计算
            size_diff = abs(file_size - expected_size)
            
            return {
                "valid": size_diff < 100,
                "triangle_count": triangle_count,
                "file_size": file_size,
                "expected_size": expected_size,
                "size_difference": size_diff
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}

    def calculate_geometry_properties(self, stl_path: str) -> Dict[str, Any]:
        """计算几何属性而不依赖LLM"""
        try:
            if not os.path.exists(stl_path):
                return {"error": "STL file not found"}
            
            # 简单的STL分析（不依赖外部库）
            import struct
            vertices = []
            
            with open(stl_path, 'rb') as f:
                header = f.read(80)
                triangle_count = struct.unpack('<I', f.read(4))[0]
                
                for i in range(triangle_count):
                    # 跳过法向量
                    f.read(12)
                    # 读取3个顶点
                    for j in range(3):
                        vertex = struct.unpack('<3f', f.read(12))
                        vertices.append(vertex)
                    # 跳过属性字节
                    f.read(2)
            
            if vertices:
                vertices = np.array(vertices)
                min_coords = np.min(vertices, axis=0)
                max_coords = np.max(vertices, axis=0)
                bbox_size = max_coords - min_coords
                center = (min_coords + max_coords) / 2
                
                # 估算体积（简单的边界框体积）
                estimated_volume = np.prod(bbox_size)
                
                return {
                    "bounding_box": {
                        "min": min_coords.tolist(),
                        "max": max_coords.tolist(),
                        "size": bbox_size.tolist(),
                        "center": center.tolist()
                    },
                    "estimated_volume": float(estimated_volume),
                    "triangle_count": triangle_count,
                    "vertex_count": len(vertices)
                }
            else:
                return {"error": "No vertices found"}
                
        except Exception as e:
            return {"error": str(e)}

    def analyze_code_complexity(self, code: str) -> Dict[str, Any]:
        """分析代码复杂度"""
        lines = code.split('\n')
        non_empty_lines = [l.strip() for l in lines if l.strip()]
        comment_lines = [l for l in non_empty_lines if l.startswith('#')]
        
        # 计算圈复杂度的简单版本
        complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
        complexity = 1  # 基础复杂度
        for line in non_empty_lines:
            for keyword in complexity_keywords:
                if keyword in line:
                    complexity += 1
        
        # 计算函数和类的数量
        function_count = sum(1 for line in non_empty_lines if line.startswith('def '))
        class_count = sum(1 for line in non_empty_lines if line.startswith('class '))
        
        return {
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "comment_lines": len(comment_lines),
            "cyclomatic_complexity": complexity,
            "function_count": function_count,
            "class_count": class_count,
            "complexity_level": "Low" if complexity <= 5 else "Medium" if complexity <= 10 else "High"
        }

    def monitor_execution_performance(self, code: str) -> Dict[str, Any]:
        """监控代码执行性能"""
        try:
            start_time = time.time()
            start_memory = 0
            
            if PSUTIL_AVAILABLE:
                import psutil
                process = psutil.Process(os.getpid())
                start_memory = process.memory_info().rss
            
            # 执行代码
            execution_result = self.execute_code_string(code)
            
            end_time = time.time()
            end_memory = start_memory
            
            if PSUTIL_AVAILABLE:
                end_memory = process.memory_info().rss
            
            execution_time = round(end_time - start_time, 3)
            memory_usage_mb = round((end_memory - start_memory) / 1024 / 1024, 2) if PSUTIL_AVAILABLE else 0
            
            return {
                "execution_time": execution_time,
                "memory_usage_mb": memory_usage_mb,
                "success": execution_result.get("success", False),
                "performance_level": "Fast" if execution_time < 1 else "Medium" if execution_time < 5 else "Slow",
                "file_size": execution_result.get("file_size", 0) if execution_result.get("success", False) else 0,
                "error": execution_result.get("error", "") if not execution_result.get("success", False) else ""
            }
        except Exception as e:
            return {"error": str(e), "success": False}

    def check_code_security(self, code: str) -> Dict[str, Any]:
        """检查代码安全性"""
        dangerous_patterns = [
            ('os.system', 'System command execution'),
            ('subprocess.call', 'Subprocess execution'),
            ('eval(', 'Dynamic code evaluation'),
            ('exec(', 'Dynamic code execution'),
            ('__import__', 'Dynamic import'),
            ('input(', 'User input'),
            ('raw_input(', 'User input'),
            ('open(', 'File operations'),
            ('file(', 'File operations'),
            ('compile(', 'Code compilation')
        ]
        
        security_issues = []
        safe_patterns = []
        
        for pattern, description in dangerous_patterns:
            if pattern in code:
                security_issues.append(f"{description}: {pattern}")
        
        # 检查安全的模式
        if 'import cadquery' in code:
            safe_patterns.append("Using safe CAD library")
        if 'cq.exporters.export' in code:
            safe_patterns.append("Using safe export function")
        
        return {
            "safe": len(security_issues) == 0,
            "security_issues": security_issues,
            "safe_patterns": safe_patterns,
            "risk_level": "Low" if len(security_issues) == 0 else "Medium" if len(security_issues) <= 2 else "High"
        }

    def combine_tool_results(self, syntax_analysis, pattern_check, security_check, 
                           stl_validation, geometry_props, complexity_analysis, performance_monitor) -> str:
        """综合所有工具的结果生成反馈"""
        feedback_parts = []
        
        # 语法分析
        if syntax_analysis.get("valid_syntax"):
            feedback_parts.append(f"✓ Syntax valid")
        else:
            feedback_parts.append(f"✗ Syntax error: {syntax_analysis.get('error', 'Unknown')}")
        
        # CAD模式检查
        pattern_score = pattern_check.get("score", 0)
        feedback_parts.append(f"✓ CAD patterns score: {pattern_score}")
        
        # 安全检查
        if security_check.get("safe"):
            feedback_parts.append("✓ Security: Safe")
        else:
            feedback_parts.append(f"⚠ Security issues: {len(security_check.get('security_issues', []))}")
        
        # STL验证
        if stl_validation.get("valid"):
            feedback_parts.append(f"✓ STL valid ({stl_validation.get('triangle_count', 0)} triangles)")
        else:
            feedback_parts.append(f"✗ STL invalid: {stl_validation.get('error', 'Unknown')}")
        
        # 几何属性
        if "error" not in geometry_props:
            volume = geometry_props.get("estimated_volume", 0)
            feedback_parts.append(f"✓ Volume: {volume:.3f}")
        
        # 复杂度
        complexity_level = complexity_analysis.get("complexity_level", "Unknown")
        feedback_parts.append(f"✓ Complexity: {complexity_level}")
        
        # 性能
        if "error" not in performance_monitor:
            exec_time = performance_monitor.get("execution_time", 0)
            feedback_parts.append(f"✓ Execution: {exec_time}s")
        
        return " | ".join(feedback_parts)

    def generate_code_base(self, prompt: str) -> str:

        system_prompt = """
You are a helpful assistant that generates CAD code based on the user's prompt.
Your input is a natural language prompt which describe the object , and you need to generate the corresponding CAD code.
Principle:
1. Your code must based on cadquery library.
2. Your code must in <code></code> code block so that it can be executed directly.
3. Your output should be saved in 'Ground_Truth.stl',like,cq.exporters.export(part, 'Ground_Truth.stl')

Example:
Input:
Write Python code using CADQuery to create a 3D model by extruding a circular sketch. The circle should have a radius of 0.75 units and the extrusion should be 0.20923 units high.
Output:
<think> your thinking process </think>
<code>
import cadquery as cq

radius:float = 1.5/2
height:float = 0.20923

sketch:cq.Sketch = cq.Sketch().circle(radius)
part:cq.Workplane = cq.Workplane("XY").placeSketch(sketch).extrude(height)
cq.exporters.export(part, 'Ground_Truth.stl')
</code>
"""
        
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = self.llm.chat.completions.create(
                        # model="ljh-qwen3-14b",
                        model="qwen2a5-7b-linjiahang",
                        #model="qwen2a5-32b-linjiahang-2",
                        messages=messages,
                        temperature=0.5,
                        max_tokens=10240,
                    )
        think_match=re.search(r"<think>(.*?)</think>", response.choices[0].message.content,re.DOTALL)
        code_match=re.search(r"<code>(.*?)</code>", response.choices[0].message.content,re.DOTALL)
        if code_match:
            code=code_match.group(1)
        else:
            code=None
            print("code is None++++++++++++++++++++")
            print(response.choices[0].message.content)

        return code
    
    def execute_code_string(self, code: str) -> Dict[str, Any]:
        """直接执行代码字符串并返回执行结果"""
        try:
            # 创建临时目录执行代码
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_stl_path = os.path.join(temp_dir, "Ground_Truth.stl")
                
                # 直接执行代码，不写入文件
                import subprocess
                import sys
                
                try:
                    # 使用python -c 直接执行代码字符串
                    result = subprocess.run([
                        sys.executable, "-c", code
                    ], capture_output=True, text=True, timeout=60, cwd=temp_dir)
                    
                    if result.returncode != 0:
                        return {"success": False, "error": result.stderr}
                    
                    # 检查生成的STL文件
                    if os.path.exists(temp_stl_path):
                        file_size = os.path.getsize(temp_stl_path)
                        return {
                            "success": True,
                            "file_size": file_size,
                            "stdout": result.stdout
                        }
                    else:
                        return {"success": False, "error": "未生成STL文件"}
                        
                except subprocess.TimeoutExpired:
                    return {"success": False, "error": "执行超时"}
                    
        except Exception as e:
            return {"success": False, "error": str(e)}
    

    def generate_code_ours(self, prompt: str) -> str:

        trace=[]
        SYSTEM_PROMPT = """
    You are a helpful assistant that generates CAD code based on the user's prompt.
    Your input is a natural language prompt which describe the object , and you need to generate the corresponding CAD code.
    Principle:
    1. Your code must based on cadquery library.
    2. Your code must in python code block so that it can be executed directly.
    3. Your output should be saved in 'Ground_Truth.stl',like,cq.exporters.export(part, 'Ground_Truth.stl')
    4. You will get the execution result of your code,and you should improve your code based on the execution result.
   
    Example:
    Input:
    Write Python code using CADQuery to create a 3D model by extruding a circular sketch. The circle should have a radius of 0.75 units and the extrusion should be 0.20923 units high.
    Output:
    <think> your thinking process </think>
    ```python
    import cadquery as cq

    radius:float = 1.5/2
    height:float = 0.20923

    sketch:cq.Sketch = cq.Sketch().circle(radius)
    part:cq.Workplane = cq.Workplane("XY").placeSketch(sketch).extrude(height)
    cq.exporters.export(part, 'Ground_Truth.stl')
    ```
    """
        #argue_cad
        messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        for i in range(self.max_round):
            response = self.llm.chat.completions.create(
                            model="qwen2a5-7b-linjiahang",
                            #model="qwen2a5-32b-linjiahang-2",
                            messages=messages,
                            temperature=0.5,
                            max_tokens=10240,
                        )
            messages.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })
            
            code_match=re.search(r"```python(.*?)```", response.choices[0].message.content,re.DOTALL)
            if code_match:
                code=code_match.group(1).strip()
                
                # 1. 静态分析工具
                syntax_analysis = self.analyze_code_syntax(code)
                pattern_check = self.check_cad_patterns(code)
                security_check = self.check_code_security(code)
                complexity_analysis = self.analyze_code_complexity(code)
                
                # 2. 性能监控和代码执行
                performance_monitor = self.monitor_execution_performance(code)
                execution_result = performance_monitor  # 性能监控已包含执行结果
                
                # 3. 如果执行成功，进行STL和几何分析
                stl_validation = {"valid": False, "error": "Execution failed"}
                geometry_props = {"error": "Execution failed"}
                
                if execution_result.get('success', False):
                    # 在临时目录中查找STL文件进行分析
                    with tempfile.TemporaryDirectory() as temp_dir:
                        temp_stl_path = os.path.join(temp_dir, "Ground_Truth.stl")
                        # 重新执行代码生成STL文件用于分析
                        temp_execution = self.execute_code_string(code)
                        if temp_execution.get('success', False) and os.path.exists(temp_stl_path):
                            stl_validation = self.validate_stl_file(temp_stl_path)
                            geometry_props = self.calculate_geometry_properties(temp_stl_path)
                
                # 4. 综合所有工具结果
                tool_feedback = self.combine_tool_results(
                    syntax_analysis, pattern_check, security_check, 
                    stl_validation, geometry_props, complexity_analysis, performance_monitor
                )
                
                # 5. 生成执行消息
                if execution_result.get('success', False):
                    execution_message = f"Code executed successfully! {tool_feedback}"
                    # 如果执行成功，直接退出循环
                    messages.append({
                        "role": "user", 
                        "content": execution_message
                    })
                    break
                else:
                    execution_message = f"Code execution failed. {tool_feedback}"
                    messages.append({
                        "role": "user", 
                        "content": execution_message
                    })
            else:
                execution_message = "Your code should be in python code block so that it can be executed directly."
                messages.append({
                    "role": "user", 
                    "content": execution_message
                })

            think_match=re.search(r"<think>(.*?)</think>", response.choices[0].message.content,re.DOTALL)
                
            # Record thinking process and code in trace
            trace.append({
                "round": i,
                "type": "execution_message",
                "content": execution_message
            })
            if think_match:
                trace.append({
                    "round": i,
                    "type": "thinking",
                    "content": think_match.group(1)
                })
            
            if code_match:
                trace.append({
                    "round": i,
                    "type": "code",
                    "content": code
                })
                # 记录所有工具的分析结果
                trace.append({
                    "round": i,
                    "type": "tool_analysis",
                    "content": {
                        "syntax_analysis": syntax_analysis,
                        "pattern_check": pattern_check,
                        "security_check": security_check,
                        "complexity_analysis": complexity_analysis,
                        "performance_monitor": performance_monitor,
                        "stl_validation": stl_validation,
                        "geometry_props": geometry_props
                    }
                })
            else:
                code=None
                trace.append({
                    "round": i,
                    "type": "error",
                    "content": "Failed to generate code",
                    "oringin_response":response.choices[0].message.content
                })
           
        uid=uuid.uuid4()
        trace.append({
            "message":messages,
        })
        save_to_json(trace,"/gpfs/users/linjiahang/nlp_project/cad_pj/final_pj/code/logs/debug/trace_"+str(uid)+".json")
    
        return code

class TESTWORKFLOW:
    def __init__(self,data_path:str="/gpfs/users/linjiahang/nlp_project/cad_pj/final_pj/CAD_Code_Generation/CADPrompt",log_file_path="/gpfs/users/linjiahang/nlp_project/cad_pj/final_pj/code/logs",max_workers=32):
        self.generator=CADCodeGenerator()
        self.data_path=data_path
        self.log_file_path=log_file_path
        self.max_workers=max_workers
        self.validator=CADModelValidator()
        
    def parallel_baseline(self):
        """并行处理所有样本"""
        from concurrent.futures import ThreadPoolExecutor
        dataset_path = Path(self.data_path)
        sample_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        sample_dirs.sort()
        sample_dirs = sample_dirs[:10]  # 限制处理前10个样本
        
        print(f"Processing {len(sample_dirs)} samples in parallel with {self.max_workers} workers")
        
        results = {}
        successful_generations = 0
        total_score = 0.0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_sample = {executor.submit(self.baseline, sample_dir): sample_dir for sample_dir in sample_dirs}
            
            # 收集结果
            for future in future_to_sample:
                try:
                    result = future.result()
                    if result:
                        sample_id = result["sample_id"]
                        results[sample_id] = result
                        
                        # 统计成功的样本数
                        if result.get("success", False):
                            successful_generations += 1
                        
                        # 所有样本的分数都计入总分（失败的为0分）
                        total_score += result.get("score", 0.0)
                        
                        print(f"Successfully processed sample: {sample_id}")
                except Exception as e:
                    sample_dir = future_to_sample[future]
                    print(f"Error processing sample {sample_dir.name}: {str(e)}")
                    # 处理异常的样本也计入总分（0分）
                    results[sample_dir.name] = {"sample_id": sample_dir.name, "error": str(e), "score": 0.0}
                    total_score += 0.0
        
        # 计算总体统计
        total_samples = len(sample_dirs)
        success_rate = successful_generations / total_samples if total_samples > 0 else 0
        average_score = total_score / total_samples if total_samples > 0 else 0  # 改为除以总样本数
        
        print(f"\n=== 并行基线方法结果摘要 ===")
        print(f"总样本数: {total_samples}")
        print(f"成功生成数: {successful_generations}")
        print(f"成功率: {success_rate:.2%}")
        print(f"平均分数: {average_score:.3f}")
        
        # 保存结果
        output_file = self.log_file_path + "/parallel_baseline_results.json"
        summary = {
            "total_samples": total_samples,
            "successful_generations": successful_generations,
            "success_rate": success_rate,
            "average_score": average_score,
            "detailed_results": results
        }
        
        # 转换numpy类型为Python原生类型
        summary_serializable = convert_numpy_types(summary)
        
        # 确保日志目录存在
        log_path = Path(self.log_file_path)
        log_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"详细结果已保存到: {output_file}")
        return summary

    def baseline(self, sample_dir):
        

        """处理单个样本"""
        sample_id = sample_dir.name
        print(f"\n处理样本 {sample_id}")
        
        try:
            # 读取自然语言描述
            prompt_file = sample_dir / "Natural_Language_Descriptions_Prompt_with_specific_measurements.txt"
            if not prompt_file.exists():
                prompt_file = sample_dir / "Natural_Language_Descriptions_Prompt.txt"
            
            if not prompt_file.exists():
                print(f"  跳过：未找到提示文件")
                return {"sample_id": sample_id, "error": "未找到提示文件"}
            
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            print(f"  提示: {prompt[:100]}...")
            
            # 使用LLM生成代码
            print("  生成代码中...")
            generated_code = self.generator.generate_code_base(prompt)
            
            if generated_code is None:
                print("  代码生成失败")
                return {"sample_id": sample_id, "error": "代码生成失败"}
            
            # 保存生成的代码到临时文件
            temp_code_path = sample_dir / "Generated_Code.py"
            with open(temp_code_path, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            
            print("  代码生成成功，开始验证...")
            
            # 验证生成的代码
            validation_result = self.validator.validate_sample(str(sample_dir))
            
            # 如果验证失败，尝试使用生成的代码
            if not validation_result.get("generated_result", {}).get("success", False):
                # 尝试执行生成的代码
                generated_result = self.validator.execute_python_code(str(temp_code_path))
                if generated_result.get("success", False):
                    # 重新加载Ground Truth并比较
                    gt_data = self.validator.load_ground_truth(str(sample_dir))
                    comparison = self.validator.compare_properties(generated_result, gt_data)
                    validation_result = {
                        "sample_dir": str(sample_dir),
                        "generated_result": generated_result,
                        "ground_truth_available": gt_data["json_data"] != {},
                        "comparison": comparison,
                        "validation_passed": comparison["overall_score"] > 0.6,
                        "timestamp": str(np.datetime64('now'))
                    }
            
            # 记录结果
            result = {
                "sample_id": sample_id,
                "prompt": prompt,
                "generated_code": generated_code,
                "validation_result": validation_result
            }
            
            # 统计成功率
            if validation_result.get("generated_result", {}).get("success", False):
                score = validation_result.get("comparison", {}).get("overall_score", 0.0)
                print(f"  验证成功，分数: {score:.3f}")
                result["success"] = True
                result["score"] = score
            else:
                print(f"  验证失败")
                result["success"] = False
                result["score"] = 0.0
            
            # 清理临时文件
            if temp_code_path.exists():
                temp_code_path.unlink()
            
            return result
            
        except Exception as e:
            print(f"  处理失败: {e}")
            return {"sample_id": sample_id, "error": str(e)}
        

    def parallel_ours(self):
        """并行处理所有样本"""
        from concurrent.futures import ThreadPoolExecutor
        dataset_path = Path(self.data_path)
        sample_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
        sample_dirs.sort()
        sample_dirs = sample_dirs[:30]  # 限制处理前10个样本
        
        print(f"Processing {len(sample_dirs)} samples in parallel with {self.max_workers} workers")
        
        results = {}
        successful_generations = 0
        total_score = 0.0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_sample = {executor.submit(self.ours, sample_dir): sample_dir for sample_dir in sample_dirs}
            
            # 收集结果
            for future in future_to_sample:
                try:
                    result = future.result()
                    if result:
                        sample_id = result["sample_id"]
                        results[sample_id] = result
                        
                        # 统计成功的样本数
                        if result.get("success", False):
                            successful_generations += 1
                        
                        # 所有样本的分数都计入总分（失败的为0分）
                        total_score += result.get("score", 0.0)
                        
                        print(f"Successfully processed sample: {sample_id}")
                except Exception as e:
                    sample_dir = future_to_sample[future]
                    print(f"Error processing sample {sample_dir.name}: {str(e)}")
                    # 处理异常的样本也计入总分（0分）
                    results[sample_dir.name] = {"sample_id": sample_dir.name, "error": str(e), "score": 0.0}
                    total_score += 0.0
        
        # 计算总体统计
        total_samples = len(sample_dirs)
        success_rate = successful_generations / total_samples if total_samples > 0 else 0
        average_score = total_score / total_samples if total_samples > 0 else 0  # 改为除以总样本数
        
        print(f"\n=== 并行基线方法结果摘要 ===")
        print(f"总样本数: {total_samples}")
        print(f"成功生成数: {successful_generations}")
        print(f"成功率: {success_rate:.2%}")
        print(f"平均分数: {average_score:.3f}")
        
        # 保存结果
        output_file = self.log_file_path + "/parallel_baseline_results.json"
        summary = {
            "total_samples": total_samples,
            "successful_generations": successful_generations,
            "success_rate": success_rate,
            "average_score": average_score,
            "detailed_results": results
        }
        
        # 转换numpy类型为Python原生类型
        summary_serializable = convert_numpy_types(summary)
        
        # 确保日志目录存在
        log_path = Path(self.log_file_path)
        log_path.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"详细结果已保存到: {output_file}")
        return summary

    def ours(self, sample_dir):
        

        """处理单个样本"""
        sample_id = sample_dir.name
        print(f"\n处理样本 {sample_id}")
        
        try:
            # 读取自然语言描述
            prompt_file = sample_dir / "Natural_Language_Descriptions_Prompt_with_specific_measurements.txt"
            if not prompt_file.exists():
                prompt_file = sample_dir / "Natural_Language_Descriptions_Prompt.txt"
            
            if not prompt_file.exists():
                print(f"  跳过：未找到提示文件")
                return {"sample_id": sample_id, "error": "未找到提示文件"}
            
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            
            print(f"  提示: {prompt[:100]}...")
            
            # 使用LLM生成代码
            print("  生成代码中...")
            generated_code = self.generator.generate_code_ours(prompt)
            
            if generated_code is None:
                print("  代码生成失败")
                return {"sample_id": sample_id, "error": "代码生成失败"}
            
            # 保存生成的代码到临时文件
            temp_code_path = sample_dir / "Generated_Code.py"
            with open(temp_code_path, 'w', encoding='utf-8') as f:
                f.write(generated_code)
            
            print("  代码生成成功，开始验证...")
            
            # 验证生成的代码
            validation_result = self.validator.validate_sample(str(sample_dir))
            
            # 如果验证失败，尝试使用生成的代码
            if not validation_result.get("generated_result", {}).get("success", False):
                # 尝试执行生成的代码
                generated_result = self.validator.execute_python_code(str(temp_code_path))
                if generated_result.get("success", False):
                    # 重新加载Ground Truth并比较
                    gt_data = self.validator.load_ground_truth(str(sample_dir))
                    comparison = self.validator.compare_properties(generated_result, gt_data)
                    validation_result = {
                        "sample_dir": str(sample_dir),
                        "generated_result": generated_result,
                        "ground_truth_available": gt_data["json_data"] != {},
                        "comparison": comparison,
                        "validation_passed": comparison["overall_score"] > 0.6,
                        "timestamp": str(np.datetime64('now'))
                    }
            
            # 记录结果
            result = {
                "sample_id": sample_id,
                "prompt": prompt,
                "generated_code": generated_code,
                "validation_result": validation_result
            }
            
            # 统计成功率
            if validation_result.get("generated_result", {}).get("success", False):
                score = validation_result.get("comparison", {}).get("overall_score", 0.0)
                print(f"  验证成功，分数: {score:.3f}")
                result["success"] = True
                result["score"] = score
            else:
                print(f"  验证失败")
                result["success"] = False
                result["score"] = 0.0
            
            # 清理临时文件
            if temp_code_path.exists():
                temp_code_path.unlink()
            
            return result
            
        except Exception as e:
            print(f"  处理失败: {e}")
            return {"sample_id": sample_id, "error": str(e)}

class PerformanceAnalyzer:
    """性能分析器 - 比较不同max_round设置的效果"""
    
    def __init__(self, data_path: str, log_file_path: str):
        self.data_path = data_path
        self.log_file_path = log_file_path
        self.results = {}
        
    def run_comparative_analysis(self, max_round_values=[1, 5], max_workers=32, num_samples=10):
        """运行比较分析"""
        print("=" * 80)
        print("CAD代码生成系统 - 性能比较分析")
        print("=" * 80)
        
        for max_round in max_round_values:
            print(f"\n🔄 测试 max_round = {max_round}")
            print("-" * 50)
            
            # 创建专门的工作流实例
            workflow = TESTWORKFLOW(
                data_path=self.data_path,
                log_file_path=self.log_file_path,
                max_workers=max_workers
            )
            
            # 设置max_round
            workflow.generator.max_round = max_round
            
            
            # 限制样本数量
            dataset_path = Path(self.data_path)
            sample_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            sample_dirs.sort()
            sample_dirs = sample_dirs[:num_samples]
            
            # 运行测试
            start_time = time.time()
            results = self._run_single_test(workflow, sample_dirs, max_round)
            end_time = time.time()
            
            # 添加时间信息
            results['total_time'] = end_time - start_time
            results['avg_time_per_sample'] = results['total_time'] / len(sample_dirs)
            
            # 保存结果
            self.results[f"max_round_{max_round}"] = results
            
            # 输出简要统计
            print(f"✅ 完成 max_round={max_round}")
            print(f"   成功率: {results['success_rate']:.2%}")
            print(f"   平均分数: {results['average_score']:.3f}")
            print(f"   总耗时: {results['total_time']:.1f}秒")
        
        return self.results
    
    def _run_single_test(self, workflow, sample_dirs, max_round):
        """运行单个测试"""
        from concurrent.futures import ThreadPoolExecutor
        
        results = {}
        successful_generations = 0
        total_score = 0.0
        scores = []
        execution_times = []
        round_counts = []
        tool_analysis_data = []
        
        with ThreadPoolExecutor(max_workers=workflow.max_workers) as executor:
            future_to_sample = {executor.submit(workflow.ours, sample_dir): sample_dir for sample_dir in sample_dirs}
            
            for future in future_to_sample:
                try:
                    result = future.result()
                    if result:
                        sample_id = result["sample_id"]
                        results[sample_id] = result
                        
                        # 基础统计
                        if result.get("success", False):
                            successful_generations += 1
                            score = result.get("score", 0.0)
                            scores.append(score)
                        else:
                            scores.append(0.0)
                        
                        total_score += result.get("score", 0.0)
                        
                        # 提取详细信息
                        self._extract_detailed_metrics(result, execution_times, round_counts, tool_analysis_data)
                        
                except Exception as e:
                    sample_dir = future_to_sample[future]
                    results[sample_dir.name] = {"sample_id": sample_dir.name, "error": str(e), "score": 0.0}
                    scores.append(0.0)
                    total_score += 0.0
        
        # 计算综合统计
        total_samples = len(sample_dirs)
        success_rate = successful_generations / total_samples if total_samples > 0 else 0
        average_score = total_score / total_samples if total_samples > 0 else 0
        
        return {
            "max_round": max_round,
            "total_samples": total_samples,
            "successful_generations": successful_generations,
            "success_rate": success_rate,
            "average_score": average_score,
            "scores": scores,
            "score_std": np.std(scores),
            "score_median": np.median(scores),
            "score_max": np.max(scores),
            "score_min": np.min(scores),
            "execution_times": execution_times,
            "round_counts": round_counts,
            "tool_analysis_data": tool_analysis_data,
            "detailed_results": results,
            "high_score_count": sum(1 for s in scores if s > 0.8),
            "medium_score_count": sum(1 for s in scores if 0.6 < s <= 0.8),
            "low_score_count": sum(1 for s in scores if 0 < s <= 0.6),
            "zero_score_count": sum(1 for s in scores if s == 0)
        }
    
    def _extract_detailed_metrics(self, result, execution_times, round_counts, tool_analysis_data):
        """提取详细指标"""
        # 这里可以从trace文件中提取更多信息
        # 暂时使用基础信息
        execution_times.append(1.0)  # 占位符
        round_counts.append(1)  # 占位符
        
        # 提取工具分析数据
        if "validation_result" in result:
            validation = result["validation_result"]
            if "comparison" in validation:
                comparison = validation["comparison"]
                tool_analysis_data.append({
                    "overall_score": comparison.get("overall_score", 0),
                    "execution_success": comparison.get("execution_success", False)
                })
    
    def generate_comparison_report(self):
        """生成比较报告"""
        if len(self.results) < 2:
            print("❌ 需要至少两个测试结果才能进行比较")
            return
        
        print("\n" + "=" * 80)
        print("📊 性能比较报告")
        print("=" * 80)
        
        # 创建比较表格
        comparison_data = []
        for config_name, result in self.results.items():
            comparison_data.append({
                "Configuration": config_name,
                "Success Rate": f"{result['success_rate']:.2%}",
                "Average Score": f"{result['average_score']:.3f}",
                "Score Std": f"{result['score_std']:.3f}",
                "Median Score": f"{result['score_median']:.3f}",
                "Max Score": f"{result['score_max']:.3f}",
                "Min Score": f"{result['score_min']:.3f}",
                "High Score(>0.8)": result['high_score_count'],
                "Medium Score(0.6-0.8)": result['medium_score_count'],
                "Low Score(0-0.6)": result['low_score_count'],
                "Failed Samples": result['zero_score_count'],
                "Total Time(s)": f"{result['total_time']:.1f}",
                "Avg Time/Sample": f"{result['avg_time_per_sample']:.1f}s"
            })
        
        df = pd.DataFrame(comparison_data)
        print("\n📋 Detailed Comparison Table:")
        print(df.to_string(index=False))
        
        # 计算改进幅度
        if "max_round_1" in self.results and "max_round_5" in self.results:
            r1 = self.results["max_round_1"]
            r5 = self.results["max_round_5"]
            
            success_improvement = (r5['success_rate'] - r1['success_rate']) * 100
            score_improvement = ((r5['average_score'] - r1['average_score']) / r1['average_score'] * 100) if r1['average_score'] > 0 else 0
            time_cost = r5['total_time'] - r1['total_time']
            
            print(f"\n🔍 Key Improvement Metrics:")
            print(f"   Success Rate Improvement: {success_improvement:+.1f} percentage points")
            print(f"   Average Score Improvement: {score_improvement:+.1f}%")
            print(f"   Additional Time Cost: {time_cost:+.1f} seconds")
            print(f"   High Score Samples Increase: {r5['high_score_count'] - r1['high_score_count']:+d}")
    
    def create_visualizations(self, save_path=None):
        """创建可视化图表"""
        if len(self.results) < 2:
            print("❌ 需要至少两个测试结果才能创建可视化")
            return
        
        # 设置字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('CAD Code Generation System Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. 成功率比较
        configs = list(self.results.keys())
        success_rates = [self.results[config]['success_rate'] for config in configs]
        
        axes[0, 0].bar(configs, success_rates, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 0].set_title('Success Rate Comparison')
        axes[0, 0].set_ylabel('Success Rate')
        axes[0, 0].set_ylim(0, 1)
        for i, v in enumerate(success_rates):
            axes[0, 0].text(i, v + 0.02, f'{v:.2%}', ha='center', va='bottom')
        
        # 2. 平均分数比较
        avg_scores = [self.results[config]['average_score'] for config in configs]
        
        axes[0, 1].bar(configs, avg_scores, color=['#FF6B6B', '#4ECDC4'])
        axes[0, 1].set_title('Average Score Comparison')
        axes[0, 1].set_ylabel('Average Score')
        axes[0, 1].set_ylim(0, 1)
        for i, v in enumerate(avg_scores):
            axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # 3. 分数分布箱线图
        score_data = []
        labels = []
        for config in configs:
            score_data.append(self.results[config]['scores'])
            labels.append(config)
        
        axes[0, 2].boxplot(score_data, labels=labels)
        axes[0, 2].set_title('Score Distribution Box Plot')
        axes[0, 2].set_ylabel('Score')
        
        # 4. 分数等级分布
        categories = ['High(>0.8)', 'Medium(0.6-0.8)', 'Low(0-0.6)', 'Failed(0)']
        
        x = np.arange(len(categories))
        width = 0.35
        
        for i, config in enumerate(configs):
            values = [
                self.results[config]['high_score_count'],
                self.results[config]['medium_score_count'],
                self.results[config]['low_score_count'],
                self.results[config]['zero_score_count']
            ]
            axes[1, 0].bar(x + i*width, values, width, label=config, alpha=0.8)
        
        axes[1, 0].set_title('Score Level Distribution')
        axes[1, 0].set_ylabel('Number of Samples')
        axes[1, 0].set_xticks(x + width/2)
        axes[1, 0].set_xticklabels(categories)
        axes[1, 0].legend()
        
        # 5. 时间成本比较
        total_times = [self.results[config]['total_time'] for config in configs]
        
        axes[1, 1].bar(configs, total_times, color=['#FF6B6B', '#4ECDC4'])
        axes[1, 1].set_title('Total Time Comparison')
        axes[1, 1].set_ylabel('Total Time (seconds)')
        for i, v in enumerate(total_times):
            axes[1, 1].text(i, v + max(total_times)*0.02, f'{v:.1f}s', ha='center', va='bottom')
        
        # 6. 综合性能雷达图
        metrics = ['Success Rate', 'Avg Score', 'High Score Rate', 'Stability', 'Efficiency']
        
        # 计算各项指标（归一化到0-1）
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        axes[1, 2].remove()
        ax_radar = fig.add_subplot(2, 3, 6, projection='polar')
        
        for config in configs:
            result = self.results[config]
            values = [
                result['success_rate'],
                result['average_score'],
                result['high_score_count'] / result['total_samples'],
                1 - (result['score_std'] / max(result['average_score'], 0.1)),  # 稳定性
                1 / (1 + result['avg_time_per_sample'] / 10)  # 效率（时间越短越好）
            ]
            values += values[:1]  # 闭合
            
            ax_radar.plot(angles, values, 'o-', linewidth=2, label=config)
            ax_radar.fill(angles, values, alpha=0.25)
        
        ax_radar.set_xticks(angles[:-1])
        ax_radar.set_xticklabels(metrics)
        ax_radar.set_ylim(0, 1)
        ax_radar.set_title('Comprehensive Performance Radar Chart')
        ax_radar.legend()
        
        plt.tight_layout()
        
        # 保存图表
        if save_path is None:
            save_path = os.path.join(self.log_file_path, "performance_comparison.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n📈 Visualization chart saved to: {save_path}")
        
        # 显示图表
        plt.show()
        
        return fig
    
    def save_detailed_report(self, filename=None):
        """保存详细报告"""
        if filename is None:
            filename = os.path.join(self.log_file_path, "performance_analysis_report.json")
        
        # 准备报告数据
        report = {
            "analysis_timestamp": str(np.datetime64('now')),
            "comparison_results": self.results,
            "summary": self._generate_summary()
        }
        
        # 转换numpy类型
        report_serializable = convert_numpy_types(report)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 详细报告已保存到: {filename}")
        return filename
    
    def _generate_summary(self):
        """生成摘要"""
        if "max_round_1" not in self.results or "max_round_5" not in self.results:
            return {"error": "缺少必要的测试结果"}
        
        r1 = self.results["max_round_1"]
        r5 = self.results["max_round_5"]
        
        return {
            "best_config": "max_round_5" if r5['average_score'] > r1['average_score'] else "max_round_1",
            "score_improvement": r5['average_score'] - r1['average_score'],
            "success_rate_improvement": r5['success_rate'] - r1['success_rate'],
            "time_cost_increase": r5['total_time'] - r1['total_time'],
            "recommendation": self._get_recommendation(r1, r5)
        }
    
    def _get_recommendation(self, r1, r5):
        """生成推荐"""
        score_improvement = r5['average_score'] - r1['average_score']
        time_cost = r5['total_time'] - r1['total_time']
        
        if score_improvement > 0.1 and time_cost < 60:
            return "Strongly recommend max_round=5: significant improvement with acceptable time cost"
        elif score_improvement > 0.05:
            return "Recommend max_round=5: noticeable improvement"
        elif score_improvement > 0:
            return "Slightly recommend max_round=5: minor improvement"
        else:
            return "Recommend max_round=1: saves time with comparable results"

def test_performance_comparison():
    """测试性能比较功能"""
    print("🚀 开始性能比较测试...")
    
    # 初始化分析器
    analyzer = PerformanceAnalyzer(
        data_path="/gpfs/users/linjiahang/nlp_project/cad_pj/final_pj/CAD_Code_Generation/CADPrompt",
        log_file_path="/gpfs/users/linjiahang/nlp_project/cad_pj/final_pj/code/logs"
    )
    
    # 运行比较分析
    results = analyzer.run_comparative_analysis(
        max_round_values=[1, 5],
        max_workers=32,
        num_samples=10
    )
    
    # 生成报告
    analyzer.generate_comparison_report()
    
    # 创建可视化
    analyzer.create_visualizations()
    
    # 保存详细报告
    analyzer.save_detailed_report()
    
    print("\n✅ 性能比较测试完成！")
    return analyzer

def main():
    """主函数 - 可以选择运行baseline测试或性能比较测试"""
    import sys
    
    print("=" * 60)
    print("CAD代码生成系统测试")
    print("=" * 60)
    print("请选择测试模式:")
    print("1. 运行baseline测试 (原有功能)")
    print("2. 运行性能比较测试 (max_round=1 vs max_round=5)")
    print("3. 直接运行性能比较 (无交互)")
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\n请输入选择 (1/2/3): ").strip()
    
    if choice == "2":
        # 运行性能比较测试
        analyzer = test_performance_comparison()
        return analyzer
    elif choice == "3":
        # 直接运行性能比较
        print("🚀 直接运行性能比较测试...")
        analyzer = PerformanceAnalyzer(
            data_path="/gpfs/users/linjiahang/nlp_project/cad_pj/final_pj/CAD_Code_Generation/CADPrompt",
            log_file_path="/gpfs/users/linjiahang/nlp_project/cad_pj/final_pj/code/logs"
        )
        results = analyzer.run_comparative_analysis(max_round_values=[1,5], num_samples=30)
        analyzer.generate_comparison_report()
        analyzer.create_visualizations()
        analyzer.save_detailed_report()
        return analyzer
    else:
        # 运行原有的baseline测试
        run_baseline_test()

def run_baseline_test():
    """运行原有的baseline测试"""
    # 清理debug日志文件
    debug_log_path = Path("/gpfs/users/linjiahang/nlp_project/cad_pj/final_pj/code/logs/debug")
    if debug_log_path.exists():
        print("清理debug日志文件...")
        for file in debug_log_path.glob("*"):
            if file.is_file():
                file.unlink()
                print(f"   删除文件: {file.name}")
        print("   debug日志文件清理完成")
    else:
        print("   debug日志目录不存在，跳过清理")
    print("=" * 60)
    print("CAD代码生成系统 - Baseline方法测试")
    print("=" * 60)
    
    # 初始化测试工作流
    print("\n1. 初始化测试工作流...")
    try:
        workflow = TESTWORKFLOW()
        print(f"   数据路径: {workflow.data_path}")
        print(f"   日志路径: {workflow.log_file_path}")
        print("   初始化成功!")
    except Exception as e:
        print(f"   初始化失败: {e}")
        return
    
    # 检查数据路径是否存在
    print("\n2. 检查数据路径...")
    dataset_path = Path(workflow.data_path)
    if not dataset_path.exists():
        print(f"   错误: 数据路径不存在 - {workflow.data_path}")
        return
    
    # 统计样本数量
    sample_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
    print(f"   找到 {len(sample_dirs)} 个样本目录")
    
    if len(sample_dirs) == 0:
        print("   错误: 没有找到任何样本目录")
        return
    
    # 检查日志目录
    print("\n3. 检查日志目录...")
    log_path = Path(workflow.log_file_path)
    if not log_path.exists():
        print(f"   创建日志目录: {workflow.log_file_path}")
        log_path.mkdir(parents=True, exist_ok=True)
    else:
        print(f"   日志目录已存在: {workflow.log_file_path}")
    
    # 运行baseline方法
    print("\n4. 运行Baseline方法...")
    print("-" * 40)
    
    try:
        start_time = np.datetime64('now')
        print(f"   开始时间: {start_time}")
        
        # 执行baseline方法
        baseline_results = workflow.parallel_ours()
        
        end_time = np.datetime64('now')
        print(f"   结束时间: {end_time}")
        print(f"    运行时间：{end_time - start_time}")
        
        # 显示详细结果
        print("\n5. 结果分析...")
        print("-" * 40)
        
        if baseline_results:
            print(f"   总样本数: {baseline_results['total_samples']}")
            print(f"   成功生成数: {baseline_results['successful_generations']}")
            print(f"   成功率: {baseline_results['success_rate']:.2%}")
            print(f"   平均分数: {baseline_results['average_score']:.3f}")
            
            # 分析详细结果
            detailed_results = baseline_results.get('detailed_results', {})
            
            # 统计不同类型的结果
            error_count = 0
            success_count = 0
            high_score_count = 0  # 分数 > 0.8
            medium_score_count = 0  # 0.6 < 分数 <= 0.8
            low_score_count = 0  # 分数 <= 0.6
            
            for sample_id, result in detailed_results.items():
                if "error" in result:
                    error_count += 1
                else:
                    validation_result = result.get("validation_result", {})
                    if validation_result.get("generated_result", {}).get("success", False):
                        success_count += 1
                        score = validation_result.get("comparison", {}).get("overall_score", 0.0)
                        
                        if score > 0.8:
                            high_score_count += 1
                        elif score > 0.6:
                            medium_score_count += 1
                        else:
                            low_score_count += 1
            
            print(f"\n   详细统计:")
            print(f"   - 错误样本: {error_count}")
            print(f"   - 成功样本: {success_count}")
            print(f"   - 高分样本 (>0.8): {high_score_count}")
            print(f"   - 中分样本 (0.6-0.8): {medium_score_count}")
            print(f"   - 低分样本 (≤0.6): {low_score_count}")
            
            # 显示具体样本结果
            print(f"\n   样本详情:")
            for sample_id, result in detailed_results.items():
                if "error" in result:
                    print(f"   - {sample_id}: 错误 - {result['error']}")
                else:
                    validation_result = result.get("validation_result", {})
                    if validation_result.get("generated_result", {}).get("success", False):
                        score = validation_result.get("comparison", {}).get("overall_score", 0.0)
                        passed = validation_result.get("validation_passed", False)
                        status = "通过" if passed else "未通过"
                        print(f"   - {sample_id}: {status} (分数: {score:.3f})")
                    else:
                        print(f"   - {sample_id}: 执行失败")
        
        else:
            print("   错误: baseline方法返回空结果")
            
    except Exception as e:
        print(f"   运行失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 检查输出文件
    print("\n6. 检查输出文件...")
    output_file = Path(workflow.log_file_path) / "parallel_baseline_results.json"
    if output_file.exists():
        file_size = output_file.stat().st_size
        print(f"   结果文件已生成: {output_file}")
        print(f"   文件大小: {file_size} bytes")
        
        # 验证JSON文件格式
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            print(f"   JSON格式验证: 成功")
            print(f"   保存的样本数: {len(saved_data.get('detailed_results', {}))}")
        except Exception as e:
            print(f"   JSON格式验证: 失败 - {e}")
    else:
        print(f"   警告: 结果文件未找到 - {output_file}")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)


if __name__ == "__main__":
    main() 