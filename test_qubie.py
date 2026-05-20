import torch
import torch.nn as nn
import torch.nn.functional as F
import math

print("="*70)
print("对比两种LoRA实现方式")
print("="*70)

# 方法1：你的实现方式（使用nn.Linear作为LoRA组件）
print("\n" + "="*70)
print("1. 你的实现方式（使用nn.Linear作为LoRA组件）")
print("="*70)

class YourLoRALayer(nn.Module):
    """你的实现方式：使用nn.Linear作为LoRA组件"""
    def __init__(self, in_features, out_features, lora_rank=4, layer_name="query"):
        super().__init__()
        self.layer_name = layer_name
        # 基础权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA部分 - 使用nn.Linear作为组件
        self.lora_A = nn.Linear(in_features, lora_rank, bias=False)  # ❌ 关键区别
        self.add_module('lora_A', self.lora_A)
        self.lora_B = nn.Linear(lora_rank, out_features, bias=False) # ❌ 关键区别
        self.add_module('lora_B', self.lora_B)  # 明确注册为子Module
        
        # 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = self.lora_B(self.lora_A(x))  # 通过两个Linear
        return base + lora
    
    def __repr__(self):
        return f"YourLoRALayer({self.layer_name}, lora_A={type(self.lora_A)}, lora_A.weight_id={id(self.lora_A.weight)})"

# 方法2：ShareLoRA实现方式（直接使用Parameter）
print("\n" + "="*70)
print("2. ShareLoRA实现方式（直接使用Parameter）")
print("="*70)

class ShareLoRALayer(nn.Module):
    """ShareLoRA方式：直接使用Parameter"""
    def __init__(self, in_features, out_features, lora_rank=4, layer_name="query"):
        super().__init__()
        self.layer_name = layer_name
        # 基础权重
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA部分 - 直接使用Parameter
        self.lora_A = nn.Parameter(torch.randn(lora_rank, in_features))  # ✅ 关键区别
        self.lora_B = nn.Parameter(torch.randn(out_features, lora_rank)) # ✅ 关键区别
        
        # 初始化
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x):
        base = F.linear(x, self.weight, self.bias)
        lora = (x @ self.lora_A.t()) @ self.lora_B.t()  # 直接使用Parameter
        return base + lora
    
    def __repr__(self):
        return f"ShareLoRALayer({self.layer_name}, lora_A={type(self.lora_A)}, lora_A_id={id(self.lora_A)})"

# 创建实例对比
print("\n" + "="*70)
print("3. 创建实例对比")
print("="*70)

# 创建你的方式实例
your_query = YourLoRALayer(10, 8, layer_name="query")
your_key = YourLoRALayer(10, 8, layer_name="key")
your_value = YourLoRALayer(10, 8, layer_name="value")

print("你的实现方式创建:")
print(f"  query层: {your_query}")
print(f"  key层: {your_key}")
print(f"  value层: {your_value}")

# 创建ShareLoRA方式实例
share_query = ShareLoRALayer(10, 8, layer_name="query")
share_key = ShareLoRALayer(10, 8, layer_name="key")
share_value = ShareLoRALayer(10, 8, layer_name="value")

print("\nShareLoRA实现方式创建:")
print(f"  query层: {share_query}")
print(f"  key层: {share_key}")
print(f"  value层: {share_value}")

# 测试参数共享
print("\n" + "="*70)
print("4. 测试参数共享")
print("="*70)

def test_your_sharing():
    """测试你的方式参数共享"""
    print("\n测试你的方式参数共享:")
    
    # 方法A：直接赋值（可能不工作）
    print("方法A：直接赋值 module.lora_A = shared_module")
    your_key.lora_A = your_query.lora_A
    print(f"  your_key.lora_A is your_query.lora_A: {your_key.lora_A is your_query.lora_A}")
    print(f"  your_key.lora_A.weight is your_query.lora_A.weight: {your_key.lora_A.weight is your_query.lora_A.weight}")
    
    # 检查_parameters
    print(f"\n  检查_parameters:")
    print(f"    your_query._parameters keys: {list(your_query._parameters.keys())}")
    print(f"    your_key._parameters keys: {list(your_key._parameters.keys())}")
    
    # 方法B：直接操作_parameters
    print("\n方法B：直接操作 _parameters")
    your_value.lora_A = your_query.lora_A
    your_value._parameters['lora_A'] = your_query.lora_A.weight  # 直接设置Parameter
    print(f"  your_value.lora_A类型: {type(your_value.lora_A)}")
    if isinstance(your_value.lora_A, nn.Parameter):
        print(f"  your_value.lora_A is your_query.lora_A.weight: {your_value.lora_A is your_query.lora_A.weight}")
    elif hasattr(your_value.lora_A, 'weight'):
        print(f"  your_value.lora_A.weight is your_query.lora_A.weight: {your_value.lora_A.weight is your_query.lora_A.weight}")

def test_sharelora_sharing():
    """测试ShareLoRA方式参数共享"""
    print("\n测试ShareLoRA方式参数共享:")
    
    # 直接赋值Parameter
    print("方法：直接赋值 module.lora_A = shared_parameter")
    share_key.lora_A = share_query.lora_A
    print(f"  share_key.lora_A is share_query.lora_A: {share_key.lora_A is share_query.lora_A}")
    print(f"  share_key.lora_A ID: {id(share_key.lora_A)}")
    print(f"  share_query.lora_A ID: {id(share_query.lora_A)}")
    
    # 也共享给value
    share_value.lora_A = share_query.lora_A
    print(f"  share_value.lora_A is share_query.lora_A: {share_value.lora_A is share_query.lora_A}")

# 执行测试
test_your_sharing()
test_sharelora_sharing()

# 测试计算图
print("\n" + "="*70)
print("5. 测试计算图差异")
print("="*70)

def analyze_computation_graph(layer1, layer2, layer1_name, layer2_name):
    """分析两个层的计算图"""
    print(f"\n分析 {layer1_name} 和 {layer2_name} 的计算图:")
    
    x = torch.randn(2, 10, requires_grad=True)
    
    # 分别前向传播
    out1 = layer1(x)
    out2 = layer2(x)
    
    print(f"  {layer1_name}输出grad_fn: {type(out1.grad_fn).__name__ if out1.grad_fn else 'None'}")
    print(f"  {layer2_name}输出grad_fn: {type(out2.grad_fn).__name__ if out2.grad_fn else 'None'}")
    
    # 检查lora_A是否在计算图中
    def find_param_in_output(output, param, param_name):
        """检查参数是否在计算图中"""
        found = False
        def search(grad_fn, target, visited=None):
            nonlocal found
            if visited is None:
                visited = set()
            if grad_fn in visited:
                return
            visited.add(grad_fn)
            
            # 检查是否是目标参数
            if hasattr(grad_fn, 'variable') and grad_fn.variable is target:
                found = True
                return
            
            # 继续搜索
            if hasattr(grad_fn, 'next_functions'):
                for next_fn, _ in grad_fn.next_functions:
                    if next_fn is not None:
                        search(next_fn, target, visited)
        
        if output.grad_fn is not None:
            search(output.grad_fn, param)
        return found
    
    # 获取lora_A参数
    if hasattr(layer1, 'lora_A'):
        lora_A_1 = layer1.lora_A
        if hasattr(lora_A_1, 'weight'):
            param1 = lora_A_1.weight  # 你的方式
        else:
            param1 = lora_A_1  # ShareLoRA方式
        
        in_graph1 = find_param_in_output(out1, param1, f"{layer1_name}.lora_A")
        in_graph2 = find_param_in_output(out2, param1, f"{layer1_name}.lora_A")
        
        print(f"  {layer1_name}.lora_A在{layer1_name}计算图中: {in_graph1}")
        print(f"  {layer1_name}.lora_A在{layer2_name}计算图中: {in_graph2}")
        
        if in_graph1 and in_graph2 and layer1.lora_A is layer2.lora_A:
            print(f"  ⚠️ 警告: 同一个lora_A参数在两个独立计算图中!")

# 分析计算图
print("\n你的方式 - 共享前:")
analyze_computation_graph(your_query, your_key, "query", "key")

print("\n你的方式 - 共享后（测试共享的value和query）:")
your_value.lora_A = your_query.lora_A
analyze_computation_graph(your_query, your_value, "query", "value")

print("\nShareLoRA方式 - 共享后:")
analyze_computation_graph(share_query, share_key, "query", "key")

# 测试梯度计算
print("\n" + "="*70)
print("6. 测试梯度计算")
print("="*70)

def test_gradients(layer1, layer2, layer1_name, layer2_name):
    """测试两个层的梯度计算"""
    print(f"\n测试 {layer1_name} 和 {layer2_name} 的梯度计算:")
    
    # 清零梯度
    for layer in [layer1, layer2]:
        for param in layer.parameters():
            if param.grad is not None:
                param.grad.zero_()
    
    # 分别前向和反向
    x1 = torch.randn(1, 10, requires_grad=True)
    x2 = torch.randn(1, 10, requires_grad=True)
    
    out1 = layer1(x1)
    out2 = layer2(x2)
    
    loss1 = out1.sum()
    loss2 = out2.sum()
    
    print(f"  分别计算梯度...")
    
    # 第一次反向
    loss1.backward(retain_graph=True)
    
    # 获取第一次梯度
    if hasattr(layer1, 'lora_A'):
        if hasattr(layer1.lora_A, 'weight'):
            param = layer1.lora_A.weight
        else:
            param = layer1.lora_A
        
        if param.grad is not None:
            grad1 = param.grad.clone()
            print(f"  第一次反向后梯度范数: {grad1.norm().item():.6f}")
    
    # 第二次反向
    try:
        loss2.backward()
        print(f"  第二次反向成功")
        
        if param.grad is not None:
            grad2 = param.grad.clone()
            print(f"  第二次反向后梯度范数: {grad2.norm().item():.6f}")
            if hasattr(grad1, 'norm'):
                print(f"  梯度是否累加: {grad2.norm().item() > grad1.norm().item()}")
    
    except Exception as e:
        print(f"  第二次反向失败: {e}")

# 测试梯度
print("\n你的方式（未共享）梯度测试:")
test_gradients(your_query, your_key, "query", "key")

print("\nShareLoRA方式（共享后）梯度测试:")
test_gradients(share_query, share_key, "query", "key")

# 展示正确的前向传播方式
print("\n" + "="*70)
print("7. 正确的训练方式（避免计算图冲突）")
print("="*70)

def correct_forward_pass(layer1, layer2, layer1_name, layer2_name):
    """展示正确的前向传播方式"""
    print(f"\n{layer1_name}和{layer2_name}的正确前向传播:")
    
    x = torch.randn(2, 10, requires_grad=True)
    
    # 在同一个前向传播中处理
    out1 = layer1(x)
    out2 = layer2(x)
    
    # 合并计算损失
    combined = torch.cat([out1, out2], dim=1)
    total_loss = combined.sum()
    
    print(f"  总损失: {total_loss.item():.6f}")
    
    # 一次反向传播
    total_loss.backward()
    
    # 检查共享参数梯度
    if hasattr(layer1, 'lora_A') and layer1.lora_A is layer2.lora_A:
        if hasattr(layer1.lora_A, 'weight'):
            param = layer1.lora_A.weight
        else:
            param = layer1.lora_A
        
        if param.grad is not None:
            print(f"  共享参数梯度范数: {param.grad.norm().item():.6f}")
            print(f"  ✅ 梯度计算成功（一个计算图）")

# 展示正确方式
correct_forward_pass(share_query, share_key, "query", "key")

print("\n" + "="*70)
print("总结")
print("="*70)

print("""
关键区别总结：

1. 你的实现方式：
   - lora_A 是 nn.Linear Module（包含 weight Parameter）
   - 共享时需要共享整个Module，或者共享内部的weight Parameter
   - 问题：Module出现在多个计算图中，导致底层Parameter被多次访问
   - DeepSpeed可能报错："参数已被reduce"

2. ShareLoRA实现方式：
   - lora_A 直接是 nn.Parameter
   - 直接共享Parameter对象，更简单直接
   - 但仍需注意计算图管理

3. 共同问题：
   - 共享参数出现在多个独立的计算图中
   - 每个计算图都认为自己"拥有"这个参数
   - 梯度规约时产生冲突

4. 解决方案：
   - 确保所有使用共享参数的层在同一个前向传播中
   - 一次计算所有输出，然后一次backward
   - 避免共享参数出现在多个计算图中

你的选择：
1. 修改实现：使用直接Parameter而不是nn.Linear
2. 修改训练：统一前向传播流程
3. 或者：接受DeepSpeed Stage 1的限制
""")