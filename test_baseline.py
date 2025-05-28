#!/usr/bin/env python3
"""
测试基线方法的简单脚本
"""

import sys
import os
from pathlib import Path

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import TESTWORKFLOW

def test_baseline():
    """测试基线方法"""
    print("开始测试基线方法...")
    
    # 初始化工作流
    workflow = TESTWORKFLOW()
    
    # 运行基线方法
    results = workflow.baseline()
    
    print("\n测试完成！")
    return results

if __name__ == "__main__":
    test_baseline() 