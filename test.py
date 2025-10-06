#!/usr/bin/env python3
"""
测试torchvision RAFT光流法和CUDA支持
"""

import torch
import torchvision
import sys
import numpy as np
from PIL import Image
import warnings


def print_separator(title):
    """打印分隔线"""
    print(f"\n{'=' * 50}")
    print(f" {title}")
    print(f"{'=' * 50}")


def test_basic_info():
    """测试基本版本信息"""
    print_separator("基本信息")

    print(f"Python版本: {sys.version}")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"torchvision版本: {torchvision.__version__}")

    print(f"PyTorch编译时的CUDA版本: {torch.version.cuda}")


def test_cuda_availability():
    print_separator("CUDA Support Test")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA Availability: {cuda_available}")

    if cuda_available:
        print(f"CUDA Device Number: {torch.cuda.device_count()}")
        print(f"Current CUDA Device: {torch.cuda.current_device()}")
        print(f"GPU Name: {torch.cuda.get_device_name()}")
        print(f"GPU Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")
    else:
        print("CUDA unavailable，turn to use of CPU")


def test_raft_availability():
    """测试RAFT模型可用性"""
    print_separator("RAFT模型可用性测试")

    try:
        from torchvision.models.optical_flow import raft_large, raft_small
        print("✓ RAFT模型导入成功")

        # 测试模型构建器
        try:
            model = raft_large(weights=None)  # 不下载权重，只测试结构
            print("✓ raft_large模型构建成功")
            del model
        except Exception as e:
            print(f"✗ raft_large模型构建失败: {e}")

        try:
            model = raft_small(weights=None)
            print("✓ raft_small模型构建成功")
            del model
        except Exception as e:
            print(f"✗ raft_small模型构建失败: {e}")

    except ImportError as e:
        print(f"✗ RAFT模型导入失败: {e}")
        return False

    return True


def test_raft_weights():
    """测试RAFT预训练权重"""
    print_separator("RAFT预训练权重测试")

    try:
        from torchvision.models.optical_flow import Raft_Large_Weights
        weights = Raft_Large_Weights.DEFAULT
        print(f"✓ 默认权重: {weights}")

        # 检查transforms
        transforms = weights.transforms()
        print("✓ 权重transforms可用")

    except Exception as e:
        print(f"✗ 预训练权重测试失败: {e}")
        return False

    return True


def test_raft_inference():
    """测试RAFT推理功能"""
    print_separator("RAFT推理功能测试")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    try:
        from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
        from torchvision.utils import flow_to_image
        import torchvision.transforms.functional as F

        # 创建测试图像
        print("创建测试图像...")
        img1 = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)
        img2 = torch.randint(0, 256, (3, 256, 256), dtype=torch.uint8)

        # 预处理
        weights = Raft_Large_Weights.DEFAULT
        transforms = weights.transforms()

        # 调整尺寸 (必须能被8整除)
        img1 = F.resize(img1, size=[256, 256], antialias=False)
        img2 = F.resize(img2, size=[256, 256], antialias=False)

        # 应用transforms
        img1_batch, img2_batch = transforms(img1.unsqueeze(0), img2.unsqueeze(0))

        print("加载RAFT模型...")
        # 这里不下载权重，只测试模型结构
        model = raft_large(weights=None).to(device)
        model.eval()

        print("执行推理...")
        with torch.no_grad():
            img1_batch = img1_batch.to(device)
            img2_batch = img2_batch.to(device)

            # RAFT推理
            list_of_flows = model(img1_batch, img2_batch)

            # 获取最终流
            predicted_flow = list_of_flows[-1]
            print(f"✓ 光流计算成功，输出形状: {predicted_flow.shape}")

            # 测试流可视化
            flow_img = flow_to_image(predicted_flow)
            print(f"✓ 光流可视化成功，图像形状: {flow_img.shape}")

        print("✓ RAFT推理测试完全通过")
        return True

    except Exception as e:
        print(f"✗ RAFT推理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """测试内存使用情况"""
    print_separator("内存使用测试")

    if torch.cuda.is_available():
        print(f"GPU内存使用:")
        print(f"  已分配: {torch.cuda.memory_allocated() / 1024 ** 2:.1f} MB")
        print(f"  已缓存: {torch.cuda.memory_reserved() / 1024 ** 2:.1f} MB")

        # 清理内存
        torch.cuda.empty_cache()
        print("✓ GPU内存已清理")


def main():
    """主测试函数"""
    print("PyTorch RAFT和CUDA功能测试")

    # 忽略一些无害的警告
    warnings.filterwarnings("ignore", category=UserWarning)

    # 执行各项测试
    test_basic_info()
    test_cuda_availability()

    raft_available = test_raft_availability()
    if raft_available:
        weights_available = test_raft_weights()
        if weights_available:
            inference_success = test_raft_inference()
        else:
            inference_success = False
    else:
        inference_success = False

    test_memory_usage()

    # 总结
    print_separator("测试总结")

    cuda_ok = torch.cuda.is_available()

    print(f"CUDA支持: {'✓' if cuda_ok else '✗'}")
    print(f"RAFT可用: {'✓' if raft_available else '✗'}")
    print(f"RAFT推理: {'✓' if inference_success else '✗'}")

    if cuda_ok and raft_available and inference_success:
        print("\n 所有测试通过。")
    else:
        print("\n 部分测试失败，需要检查环境配置。")

        if not cuda_ok:
            print("   - 检查CUDA安装和GPU驱动")
        if not raft_available:
            print("   - 检查torchvision版本 (需要>=0.12)")
        if not inference_success:
            print("   - 检查模型加载和推理环境")


if __name__ == "__main__":
    main()