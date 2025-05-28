#!/usr/bin/env python3
"""
CAD模型验证系统使用示例
"""

from main import CADModelValidator
import json

def example_usage():
    """展示验证系统的基本使用方法"""
    
    print("CAD模型验证系统使用示例")
    print("=" * 50)
    
    # 1. 初始化验证器
    validator = CADModelValidator()
    
    # 2. 快速测试 - 验证前3个样本
    print("\n1. 快速测试 (前3个样本):")
    quick_results = validator.quick_test(3)
    
    # 3. 验证单个样本
    print("\n2. 验证单个样本:")
    sample_dir = "../CAD_Code_Generation/CADPrompt/00000007"
    result = validator.validate_sample(sample_dir)
    
    print(f"样本: {result['sample_dir']}")
    print(f"验证通过: {result['validation_passed']}")
    print(f"总体分数: {result['comparison']['overall_score']:.3f}")
    
    if result['comparison']['differences']:
        print("详细比较:")
        for prop, diff in result['comparison']['differences'].items():
            print(f"  {prop}: 生成值={diff['generated']:.3f}, "
                  f"真值={diff['ground_truth']:.3f}, "
                  f"误差={diff['relative_error']:.1%}")
    
    # 4. 分析Ground Truth文件
    print("\n3. 分析Ground Truth文件:")
    gt_data = validator.load_ground_truth(sample_dir)
    if gt_data['stl_exists']:
        stl_info = validator.analyze_stl_file(str(gt_data['stl_path']))
        print(f"STL格式: {stl_info.get('format', 'Unknown')}")
        print(f"面数: {stl_info.get('num_facets', 'N/A')}")
        print(f"估算体积: {stl_info.get('estimated_volume', 0):.6f}")
        print(f"估算表面积: {stl_info.get('estimated_surface_area', 0):.6f}")

def run_baseline_example():
    """运行基线验证示例（仅前10个样本）"""
    print("\n基线验证示例 (前10个样本)")
    print("=" * 50)
    
    validator = CADModelValidator()
    
    # 修改baseline方法来只处理前10个样本
    import os
    sample_dirs = sorted([
        d for d in os.listdir(validator.data_path) 
        if os.path.isdir(os.path.join(validator.data_path, d)) and d.isdigit()
    ])[:10]
    
    results = {
        "total_samples": 0,
        "successful_executions": 0,
        "failed_executions": 0,
        "execution_details": {}
    }
    
    for dir_name in sample_dirs:
        dir_path = os.path.join(validator.data_path, dir_name)
        results["total_samples"] += 1
        
        print(f"处理样本 {dir_name} ({results['total_samples']}/10)...")
        
        try:
            sample_result = validator.validate_sample(dir_path)
            
            if sample_result["generated_result"].get("success", False):
                results["successful_executions"] += 1
                status = "SUCCESS"
            else:
                results["failed_executions"] += 1
                status = "FAILED"
            
            results["execution_details"][dir_name] = {
                "status": status,
                "score": sample_result.get("comparison", {}).get("overall_score", 0.0)
            }
            
            print(f"  状态: {status}, 分数: {sample_result.get('comparison', {}).get('overall_score', 0.0):.3f}")
            
        except Exception as e:
            print(f"  错误: {e}")
            results["failed_executions"] += 1
    
    # 打印总结
    success_rate = results["successful_executions"] / results["total_samples"]
    print(f"\n基线验证总结:")
    print(f"总样本数: {results['total_samples']}")
    print(f"成功执行: {results['successful_executions']}")
    print(f"执行失败: {results['failed_executions']}")
    print(f"成功率: {success_rate:.1%}")

if __name__ == "__main__":
    # 运行基本使用示例
    example_usage()
    
    # 询问是否运行基线验证示例
    choice = input("\n是否运行基线验证示例 (前10个样本)? (y/n): ").strip().lower()
    if choice == 'y':
        run_baseline_example() 