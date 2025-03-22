class CustomModel:
    def __init__(self, model_size=None, vocab_size=None):
        self.model_size = model_size
        self.vocab_size = vocab_size
        print(f"初始化CustomModel: model_size={model_size}, vocab_size={vocab_size}")

    def generate(self, text):
        return f"处理文本: {text} (使用大小为{self.model_size}的模型)"
