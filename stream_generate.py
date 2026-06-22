# 流式生成 token 示范
# https://chatgpt.com/c/6a38c4fb-3ad0-83e8-ad59-fd704504d896

from __future__ import annotations

import queue
import threading
import time
from collections.abc import Iterator


_END = object()


class MiniGenerator:
    def generate(self, prompt: str, *, stream: bool = False) -> str | Iterator[str]:
        """
        stream=False: 返回完整字符串
        stream=True: 返回一个 Iterator[str]，逐段产生文本
        """

        # 非流式：当前线程直接跑完，然后返回完整结果
        if not stream:
            return self._run_generate(prompt, output_queue=None)

        # 流式：worker 线程负责生成，主线程负责从 queue 里取结果并 yield
        output_queue: queue.Queue[str | BaseException | object] = queue.Queue()

        def worker() -> None:
            try:
                # 生产者
                self._run_generate(prompt, output_queue=output_queue)
            except BaseException as error:
                output_queue.put(error)
            finally:
                output_queue.put(_END)

        # 消费者
        def iterator() -> Iterator[str]:
            thread = threading.Thread(target=worker, daemon=True)
            thread.start()

            try:
                while True:
                    item = output_queue.get()

                    if item is _END:
                        break

                    if isinstance(item, BaseException):
                        raise item

                    yield item
            finally:
                thread.join()

        return iterator()

    def _run_generate(
        self,
        prompt: str,
        *,
        output_queue: queue.Queue[str | BaseException | object] | None,
    ) -> str:
        """
        最小化的“生成函数”。

        output_queue is None:
            非流式模式，只收集结果，最后 return 完整字符串。

        output_queue is not None:
            流式模式，每生成一段就放进 queue。
        """

        # 假装这是模型生成出来的 token / text chunk
        fake_tokens = [
            "收到：",
            prompt,
            "\n",
            "这是",
            "一个",
            "最小",
            "流式",
            "生成",
            "示例。",
        ]

        parts: list[str] = []

        for token in fake_tokens:
            time.sleep(0.5)  # 假装模型正在推理，GPU 风扇正在表达哲学观点

            parts.append(token)

            if output_queue is not None:
                output_queue.put(token)

        return "".join(parts)


g = MiniGenerator()

# 非流式调用
text = g.generate("你好", stream=False)
print(text)
print()

# 流式调用
for piece in g.generate("你好", stream=True):
    print(piece, end="", flush=True)
