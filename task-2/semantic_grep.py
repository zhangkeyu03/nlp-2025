import sys
import gzip
import logging
import inspect  # 1. 引入 inspect
from typing import List, Set

# --- 关键修复开始：针对 Python 3.11+ 的完美补丁 ---
# 解决 pymorphy2 在 Python 3.11+ 下报错的问题
# 同时也解决了 PyCharm 报红线的问题 (使用 setattr)
if not hasattr(inspect, 'getargspec'):
    def _getargspec_stub(func):
        """
        模拟旧版 getargspec 的行为，只返回前4个参数。
        getfullargspec 返回7个参数，直接赋值会导致解包错误。
        """
        spec = inspect.getfullargspec(func)
        # 旧版 getargspec 返回: (args, varargs, keywords, defaults)
        # 新版 getfullargspec 前4项对应: args, varargs, varkw, defaults
        return (spec.args, spec.varargs, spec.varkw, spec.defaults)


    setattr(inspect, 'getargspec', _getargspec_stub)
# --- 关键修复结束 ---

# 引入必要的 NLP 库
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import pymorphy2  # <--- 必须在补丁之后导入
import nltk
from nltk.corpus import stopwords
import os

# 配置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class SemanticGrep:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.model = None
        self.sentences = []

        # 初始化形态分析器
        self.morph = pymorphy2.MorphAnalyzer()

        # 初始化停用词
        try:
            self.stop_words = set(stopwords.words('russian'))
        except LookupError:
            print("正在下载 NLTK 停用词数据...")
            nltk.download('stopwords')
            self.stop_words = set(stopwords.words('russian'))

        # 手动添加一些在新闻数据中常见的干扰词
        self.stop_words.update(['эти', 'это', 'который', 'свой', 'весь', 'наш', 'ваш'])

    def preprocess_text(self, text: str) -> List[str]:
        """
        对文本进行：分词 -> 去停用词 -> 词形还原
        """
        # 1. 简单分词和去标点 (gensim)
        tokens = simple_preprocess(text)

        clean_tokens = []
        for token in tokens:
            # 2. 过滤停用词和过短的词
            if token not in self.stop_words and len(token) > 2:
                # 3. 词形还原 (Lemmatization)
                # 例如: 'денег' -> 'деньги', 'футболу' -> 'футбол'
                normal_form = self.morph.parse(token)[0].normal_form
                clean_tokens.append(normal_form)

        return clean_tokens

    def load_and_preprocess(self):
        """读取文件并预处理用于训练"""
        print(f"正在读取并预处理数据: {self.data_path} ...")
        self.sentences = []

        # 兼容 .gz 和普通文本
        open_func = gzip.open if self.data_path.endswith('.gz') else open
        mode = 'rt' if self.data_path.endswith('.gz') else 'r'

        try:
            with open_func(self.data_path, mode, encoding='utf-8') as f:
                for i, line in enumerate(f):
                    parts = line.strip().split('\t')
                    if len(parts) >= 3:
                        text = parts[2]  # 获取正文
                    else:
                        text = line

                    # 使用改进后的预处理
                    tokens = self.preprocess_text(text)
                    if tokens:
                        self.sentences.append(tokens)

                    if i % 10000 == 0 and i > 0:
                        print(f"已处理 {i} 行...")

        except Exception as e:
            print(f"读取文件出错: {e}")
            sys.exit(1)

    def train_model(self):
        """训练 Word2Vec 模型"""
        if not self.sentences:
            self.load_and_preprocess()

        print("开始训练 Word2Vec 模型...")
        self.model = Word2Vec(
            sentences=self.sentences,
            vector_size=100,
            window=5,
            min_count=5,
            workers=4,
            epochs=10
        )
        print("模型训练完成！")

    def get_synonyms(self, word: str, topn: int = 3) -> List[str]:
        """获取同义词"""
        if not self.model:
            raise ValueError("模型未训练")

        # 对查询词也进行同样的预处理（还原为原形）
        processed = self.preprocess_text(word)
        if not processed:
            print(f"警告: 查询词 '{word}' 被停用词过滤器忽略。")
            return []

        target_word = processed[0]  # 取处理后的第一个词

        if target_word not in self.model.wv:
            print(f"警告: 单词 '{target_word}' (原词: {word}) 不在词汇表中 (OOV)。")
            return []

        # 获取最相似的词
        similar_words = self.model.wv.most_similar(target_word, topn=topn)
        return [w[0] for w in similar_words]

    def grep(self, query_word: str):
        """执行语义搜索"""
        if not self.model:
            self.train_model()

        print(f"\n执行搜索: '{query_word}'")

        # 1. 处理查询词并扩展
        processed_query = self.preprocess_text(query_word)
        if not processed_query:
            print("查询词无效（可能是停用词）。")
            return

        target_word = processed_query[0]
        synonyms = self.get_synonyms(query_word, topn=4)

        # 搜索集合包含：处理后的查询词 + 同义词
        search_terms = set([target_word] + synonyms)

        print("-" * 50)
        print(f"原始查询: '{query_word}' -> 还原形式: '{target_word}'")
        print(f"扩展语义词: {synonyms}")
        print(f"正在搜索包含以下词根的行: {search_terms}")
        print("-" * 50)

        # 2. 再次读取文件进行搜索
        open_func = gzip.open if self.data_path.endswith('.gz') else open
        mode = 'rt' if self.data_path.endswith('.gz') else 'r'

        match_count = 0

        with open_func(self.data_path, mode, encoding='utf-8') as f:
            for line_idx, line in enumerate(f):
                line_tokens = set(self.preprocess_text(line))

                # 检查交集
                found_terms = list(search_terms.intersection(line_tokens))

                if found_terms:
                    match_count += 1
                    display_line = line.strip()
                    if len(display_line) > 100:
                        display_line = display_line[:100] + "..."

                    print(f"[Line {line_idx}] [匹配词根: {', '.join(found_terms)}]: {display_line}")

                    if match_count >= 10:
                        print("... (达到显示上限，停止输出)")
                        break


if __name__ == '__main__':
    DATA_FILE = r'D:\WorkSoftware\pycharm\PyCharm 2024.1.3\projects\dancepose\nlp-2025\task1\news.txt'
    if not os.path.exists(DATA_FILE):
        print(f"错误: 找不到文件 {DATA_FILE}")
        sys.exit(1)

    grep_tool = SemanticGrep(DATA_FILE)
    grep_tool.train_model()

    # --- 测试案例 ---
    grep_tool.grep("футбол")
    grep_tool.grep("деньги")