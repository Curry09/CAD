#!/usr/bin/env python3
"""
简化的验证测试脚本
用于测试CAD模型验证功能
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import CADModelValidator

def test_single_sample():
    """测试单个样本的验证"""
    print("=== 测试单个样本验证 ===")
    
    # 初始化验证器
    validator = CADModelValidator()
    
    # 测试样本路径
    sample_dir = "../CAD_Code_Generation/CADPrompt/00000007"
    
    if not Path(sample_dir).exists():
        print(f"错误: 样本目录不存在: {sample_dir}")
        return False
    
    try:
        # 执行验证
        result = validator.validate_sample(sample_dir)
        
        if "error" in result:
            print(f"验证失败: {result['error']}")
            return False
        
        # 打印结果摘要
        print(f"样本目录: {result['sample_dir']}")
        print(f"验证通过: {result['validation_passed']}")
        print(f"总体分数: {result['comparison']['overall_score']:.3f}")
        
        # 打印详细比较结果
        comparison = result['comparison']
        if comparison['differences']:
            print("\n属性比较:")
            for prop, diff in comparison['differences'].items():
                print(f"  {prop}:")
                print(f"    生成值: {diff['generated']:.6f}")
                print(f"    真值: {diff['ground_truth']:.6f}")
                print(f"    相对误差: {diff['relative_error']:.2%}")
                print(f"    匹配: {'✓' if comparison['matches'].get(prop, False) else '✗'}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ground_truth_loading():
    """测试Ground Truth数据加载"""
    print("\n=== 测试Ground Truth数据加载 ===")
    
    validator = CADModelValidator()
    sample_dir = "../CAD_Code_Generation/CADPrompt/00000007"
    
    if not Path(sample_dir).exists():
        print(f"错误: 样本目录不存在: {sample_dir}")
        return False
    
    try:
        gt_data = validator.load_ground_truth(sample_dir)
        
        print(f"JSON数据加载: {'✓' if gt_data['json_data'] else '✗'}")
        print(f"STL文件存在: {'✓' if gt_data['stl_exists'] else '✗'}")
        
        if gt_data['json_data'] and "Ground_Truth" in gt_data['json_data']:
            gt_props = gt_data['json_data']['Ground_Truth']
            print(f"Ground Truth属性:")
            for key, value in gt_props.items():
                print(f"  {key}: {value}")
        
        # 如果STL文件存在，尝试分析它
        if gt_data['stl_exists']:
            stl_info = validator.analyze_stl_file(str(gt_data['stl_path']))
            print(f"STL文件分析:")
            print(f"  格式: {stl_info.get('format', 'Unknown')}")
            print(f"  面数: {stl_info.get('num_facets', 'N/A')}")
            if 'bounding_box' in stl_info:
                bbox = stl_info['bounding_box']
                print(f"  边界框大小: {bbox.get('size', 'N/A')}")
            print(f"  估算体积: {stl_info.get('estimated_volume', 'N/A'):.6f}")
            print(f"  估算表面积: {stl_info.get('estimated_surface_area', 'N/A'):.6f}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_python_code_execution():
    """测试Python代码执行"""
    print("\n=== 测试Python代码执行 ===")
    
    validator = CADModelValidator()
    code_path = "../CAD_Code_Generation/CADPrompt/00000007/Python_Code.py"
    
    if not Path(code_path).exists():
        print(f"错误: Python代码文件不存在: {code_path}")
        return False
    
    try:
        # 读取并显示代码
        with open(code_path, 'r') as f:
            code_content = f.read()
        print("Python代码内容:")
        print(code_content)
        
        # 执行代码
        print("\n执行代码...")
        generated_result = validator.execute_python_code(code_path)
        
        if generated_result.get("success", False):
            print("✓ 代码执行成功")
            stl_info = generated_result.get("stl_info", {})
            print(f"生成模型属性:")
            print(f"  文件大小: {generated_result.get('file_size', 'N/A')} bytes")
            print(f"  格式: {stl_info.get('format', 'N/A')}")
            print(f"  面数: {stl_info.get('num_facets', 'N/A')}")
            print(f"  估算体积: {stl_info.get('estimated_volume', 'N/A'):.6f}")
            print(f"  估算表面积: {stl_info.get('estimated_surface_area', 'N/A'):.6f}")
            return True
        else:
            print("✗ 代码执行失败")
            print(f"错误: {generated_result.get('error', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("CAD模型验证功能测试")
    print("=" * 50)
    
    tests = [
        ("Ground Truth数据加载", test_ground_truth_loading),
        ("Python代码执行", test_python_code_execution),
        ("单个样本验证", test_single_sample),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n运行测试: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
            print(f"测试结果: {'通过' if success else '失败'}")
        except Exception as e:
            print(f"测试异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"  {test_name}: {status}")
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！验证功能正常工作。")
    else:
        print("⚠️  部分测试失败，请检查环境配置和依赖。")

if __name__ == "__main__":
    main() 