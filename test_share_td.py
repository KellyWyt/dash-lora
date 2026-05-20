import torch
import torch.nn as nn

print("="*60)
print("修正版：测试直接赋值 vs 复制数据的区别")
print("="*60)

# 测试参数
in_features = 10
out_features1 = 5  # 第一个Linear的输出维度
out_features2 = 3  # 第二个Linear的输出维度
batch_size = 2

# 方法1：直接赋值（共享参数）
print("\n方法1：直接赋值（共享参数）")
print("-"*40)

# 创建一个共享参数
shared_param = nn.Parameter(torch.randn(out_features1, in_features))
print(f"创建共享参数:")
print(f"  形状: {shared_param.shape}")
print(f"  内存地址: {shared_param.data_ptr():,}")
print(f"  参数ID: {id(shared_param)}")

# 创建两个Linear层，直接使用共享参数
linear1_shared = nn.Linear(in_features, out_features1, bias=False)
linear2_shared = nn.Linear(in_features, out_features2, bias=False)

# 直接赋值参数对象
linear1_shared.weight = shared_param
print(f"\nlinear1_shared.weight:")
print(f"  形状: {linear1_shared.weight.shape}")
print(f"  内存地址: {linear1_shared.weight.data_ptr():,}")
print(f"  参数ID: {id(linear1_shared.weight)}")
print(f"  是否同一对象: {linear1_shared.weight is shared_param}")

# 对于第二个Linear，需要从共享参数中取一部分
shared_param_part = nn.Parameter(shared_param[:out_features2, :])
linear2_shared.weight = shared_param_part
print(f"\nlinear2_shared.weight:")
print(f"  形状: {linear2_shared.weight.shape}")
print(f"  内存地址: {linear2_shared.weight.data_ptr():,}")
print(f"  参数ID: {id(linear2_shared.weight)}")
print(f"  是否同一对象: {linear2_shared.weight is shared_param}")
print(f"  是否是shared_param的一部分: {linear2_shared.weight is shared_param_part}")

# 方法2：复制数据（非共享）
print("\n\n方法2：复制数据（非共享）")
print("-"*40)

# 创建两个Linear层，通过复制数据
linear1_copy = nn.Linear(in_features, out_features1, bias=False)
linear2_copy = nn.Linear(in_features, out_features2, bias=False)

# 复制数据（创建独立的参数）
copy_param1 = nn.Parameter(torch.randn(out_features1, in_features))
copy_param2 = nn.Parameter(torch.randn(out_features2, in_features))

linear1_copy.weight.data = copy_param1.data.clone()  # 复制数据
linear2_copy.weight.data = copy_param2.data.clone()  # 复制数据

print(f"copy_param1:")
print(f"  形状: {copy_param1.shape}")
print(f"  内存地址: {copy_param1.data_ptr():,}")
print(f"  参数ID: {id(copy_param1)}")

print(f"\nlinear1_copy.weight:")
print(f"  形状: {linear1_copy.weight.shape}")
print(f"  内存地址: {linear1_copy.weight.data_ptr():,}")
print(f"  参数ID: {id(linear1_copy.weight)}")
print(f"  是否同一对象: {linear1_copy.weight is copy_param1}")

# 测试梯度更新
print("\n\n梯度更新测试")
print("-"*40)

# 创建优化器 - 正确添加所有参数
optimizer_shared = torch.optim.SGD([
    shared_param,  # 包含linear1_shared的权重
    shared_param_part,  # linear2_shared的权重
], lr=0.1)

optimizer_copy = torch.optim.SGD([
    copy_param1,  # 注意：不是linear1_copy.weight，而是原始参数
    copy_param2,
    linear1_copy.weight,  # 这些也需要单独添加
    linear2_copy.weight,
], lr=0.1)

# 输入数据
x = torch.randn(batch_size, in_features)

# 方法1：直接赋值的梯度测试
print("\n1. 直接赋值（共享）的梯度测试:")
# 清空梯度
optimizer_shared.zero_grad()

out1_shared = linear1_shared(x)
out2_shared = linear2_shared(x)
loss_shared = out1_shared.sum() + out2_shared.sum()
loss_shared.backward()

print(f"  shared_param梯度范数: {shared_param.grad.norm().item():.6f}")
print(f"  linear1_shared.weight梯度范数: {linear1_shared.weight.grad.norm().item():.6f}")
print(f"  shared_param_part梯度范数: {shared_param_part.grad.norm().item():.6f}")
print(f"  linear2_shared.weight梯度范数: {linear2_shared.weight.grad.norm().item():.6f}")

print(f"\n  梯度是否同一对象:")
print(f"    shared_param.grad is linear1_shared.weight.grad: {shared_param.grad is linear1_shared.weight.grad}")
print(f"    shared_param_part.grad is linear2_shared.weight.grad: {shared_param_part.grad is linear2_shared.weight.grad}")

# 记录更新前的值
shared_param_before = shared_param.data.clone()
linear1_shared_before = linear1_shared.weight.data.clone()

# 更新参数
optimizer_shared.step()

# 检查更新
print(f"\n  参数更新检查:")
shared_param_change = (shared_param.data - shared_param_before).norm().item()
linear1_shared_change = (linear1_shared.weight.data - linear1_shared_before).norm().item()
print(f"  shared_param变化: {shared_param_change:.6f}")
print(f"  linear1_shared.weight变化: {linear1_shared_change:.6f}")
print(f"  两个变化是否相等: {abs(shared_param_change - linear1_shared_change) < 1e-6}")

# 方法2：复制数据的梯度测试
print("\n2. 复制数据（非共享）的梯度测试:")
# 清空梯度
optimizer_copy.zero_grad()

out1_copy = linear1_copy(x)
out2_copy = linear2_copy(x)
loss_copy = out1_copy.sum() + out2_copy.sum()
loss_copy.backward()

print(f"  copy_param1梯度范数: {copy_param1.grad.norm().item():.6f}")
print(f"  linear1_copy.weight梯度范数: {linear1_copy.weight.grad.norm().item():.6f}")
print(f"  copy_param2梯度范数: {copy_param2.grad.norm().item():.6f}")
print(f"  linear2_copy.weight梯度范数: {linear2_copy.weight.grad.norm().item():.6f}")

print(f"\n  梯度是否同一对象:")
print(f"    copy_param1.grad is linear1_copy.weight.grad: {copy_param1.grad is linear1_copy.weight.grad}")
print(f"    copy_param2.grad is linear2_copy.weight.grad: {copy_param2.grad is linear2_copy.weight.grad}")

# 记录更新前的值
copy_param1_before = copy_param1.data.clone()
linear1_copy_before = linear1_copy.weight.data.clone()

# 更新参数
optimizer_copy.step()

# 检查更新
print(f"\n  参数更新检查:")
copy_param1_change = (copy_param1.data - copy_param1_before).norm().item()
linear1_copy_change = (linear1_copy.weight.data - linear1_copy_before).norm().item()
print(f"  copy_param1变化: {copy_param1_change:.6f}")
print(f"  linear1_copy.weight变化: {linear1_copy_change:.6f}")
print(f"  两个变化是否相等: {abs(copy_param1_change - linear1_copy_change) < 1e-6}")

# 最终验证
print("\n" + "="*60)
print("关键结论")
print("="*60)

print("\n1. 直接赋值:")
print(f"  - linear1_shared.weight is shared_param: {linear1_shared.weight is shared_param}")
print(f"  - 内存地址相同: {linear1_shared.weight.data_ptr() == shared_param.data_ptr()}")
print(f"  - 梯度对象相同: {shared_param.grad is linear1_shared.weight.grad}")
print(f"  - 更新同步: {abs(shared_param_change - linear1_shared_change) < 1e-6}")
print(f"  → 真正的参数共享！")

print("\n2. 复制数据:")
print(f"  - linear1_copy.weight is copy_param1: {linear1_copy.weight is copy_param1}")
print(f"  - 内存地址相同: {linear1_copy.weight.data_ptr() == copy_param1.data_ptr()}")
print(f"  - 梯度对象相同: {copy_param1.grad is linear1_copy.weight.grad}")
print(f"  - 更新同步: {abs(copy_param1_change - linear1_copy_change) < 1e-6}")
print(f"  → 参数独立，只是初始值相同！")

print("\n对于你的LoRA实现:")
print("如果要实现真正的参数共享，必须使用:")
print("  lora_A.weight = shared_param  # 直接赋值Parameter对象")
print("而不是:")
print("  lora_A.weight.data = shared_param.t().contiguous()  # 这只是复制数据")