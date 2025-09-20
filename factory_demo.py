#!/usr/bin/env python3
"""
工厂模式调试演示 - 展示为什么调试时会"跳来跳去"
"""

# 模拟 object_factory.py
class ObjectFactory:
    def __init__(self):
        self._builders = {}
    
    def register_builder(self, name, builder_func):
        print(f"📝 注册阶段: 存储 {name} 的创建函数")
        self._builders[name] = builder_func  # 只是存储，不执行！
    
    def create(self, builder_name, **kwargs):
        print(f"🏭 工厂调用: 开始创建 {builder_name}")
        builder = self._builders.get(builder_name) # builder_name=amp_continuous, _builders.get怎么理解, _builder不是一个字典吗?
        # 上面的用_builders.get(builder_name)就等价与_builderes[builder_name], 只是前者更安全, 有点类似C++的.at()函数. 这些细节好烦啊.
        if not builder:
            raise ValueError(f"未找到 {builder_name} 的构建器")
        
        print(f"🔄 即将调用lambda函数...")
        # ⭐ 关键点：这里会跳回到lambda定义的地方！
        result = builder(**kwargs)  # 在这里打断点，单步调试会跳到train_demo.py
        print(f"✅ 创建完成: {result}")
        return result

# 模拟 amp_continuous.py
class AMPAgent:
    def __init__(self, name="AMP", learning_rate=0.001):
        print(f"🤖 创建AMPAgent: name={name}, lr={learning_rate}")
        self.name = name
        self.lr = learning_rate
    
    def __str__(self):
        return f"AMPAgent({self.name}, lr={self.lr})"

# 模拟 train.py 的逻辑
def train_demo():
    print("=" * 50)
    print("🚀 开始工厂模式演示")
    print("=" * 50)
    
    # 1. 创建工厂
    factory = ObjectFactory()
    
    # 2. 注册阶段 - 对应 train.py:188
    print("\n📋 第一阶段：注册lambda函数")
    print("位置：train_demo() -> factory.register_builder()")
    
    # ⭐ 在这里打断点，这个lambda函数只是被存储，不会执行
    factory.register_builder('amp_continuous', 
                           lambda **kwargs: AMPAgent(**kwargs))  # 这里定义lambda
    
    print("✅ lambda函数已注册，但还没执行")
    
    # 3. 使用阶段 - 对应 runner.run() 内部调用
    print("\n🏭 第二阶段：工厂创建对象")
    print("位置：factory.create() -> 调用之前注册的lambda")
    
    # ⭐ 在这里打断点，单步调试时会发生跳转：
    # factory.create() -> builder(**kwargs) -> 跳回上面的lambda定义处
    agent = factory.create('amp_continuous', name="演示AMP", learning_rate=0.01)
    
    print(f"\n🎉 最终结果: {agent}")

if __name__ == "__main__":
    train_demo()
    
    print("\n" + "=" * 50)
    print("🔍 调试跳转说明:")
    print("1. 在 factory.register_builder() 打断点")
    print("2. 在 factory.create() 打断点") 
    print("3. 单步调试 create() 时会跳回 lambda 定义处")
    print("4. 然后跳到 AMPAgent.__init__()")
    print("5. 这就是你看到的'跳来跳去'！")
    print("=" * 50)
