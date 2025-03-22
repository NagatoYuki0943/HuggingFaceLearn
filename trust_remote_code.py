import importlib.util
import os
import sys
from typing import Any


def load_remote_code(file_path: str, class_name: str, **kwargs) -> Any:
    """
    加载外部Python文件并实例化指定的类

    参数:
        file_path: Python文件的路径
        class_name: 要实例化的类名
        kwargs: 传递给类构造函数的参数

    返回:
        实例化的对象
    """
    # 获取文件名（不带扩展名）作为模块名
    module_name = os.path.basename(file_path).replace(".py", "")

    # 检查文件是否存在
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    # 加载模块规范
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载模块规范: {file_path}")

    # 创建模块
    module = importlib.util.module_from_spec(spec)

    # 将模块添加到sys.modules
    sys.modules[module_name] = module

    # 执行模块
    spec.loader.exec_module(module)

    # 检查类是否存在于模块中
    if not hasattr(module, class_name):
        raise AttributeError(f"在模块 {module_name} 中找不到类 {class_name}")

    # 获取类并实例化
    cls = getattr(module, class_name)
    return cls(**kwargs)


class RemoteCodeLoader:
    """模拟transformers的from_pretrained功能的简化版本"""

    @classmethod
    def from_pretrained(
        cls, model_path: str, class_name: str, trust_remote_code: bool = False, **kwargs
    ) -> Any:
        """
        从指定路径加载模型

        参数:
            model_path: 模型文件路径
            class_name: 要实例化的类名
            trust_remote_code: 是否信任并执行远程代码
            kwargs: 传递给模型构造函数的参数

        返回:
            实例化的模型对象
        """
        if not trust_remote_code:
            raise ValueError("必须设置trust_remote_code=True才能加载外部代码")

        return load_remote_code(model_path, class_name, **kwargs)


# 示例用法
if __name__ == "__main__":
    # 方法1：直接使用load_remote_code函数
    model1 = load_remote_code(
        file_path="custom_model.py",
        class_name="CustomModel",
        model_size=768,
        vocab_size=50000,
    )

    print(model1.generate("你好，世界！"))

    # 方法2：使用RemoteCodeLoader类
    model2 = RemoteCodeLoader.from_pretrained(
        model_path="custom_model.py",
        class_name="CustomModel",
        trust_remote_code=True,
        model_size=1024,
        vocab_size=30000,
    )

    print(model2.generate("这是一个测试"))
