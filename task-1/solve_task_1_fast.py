import sys
from dataclasses import dataclass
from typing import Optional
import gzip

# 引入 yargy 相关库
from yargy import Parser, rule, or_, and_
from yargy.predicates import gram, is_capitalized, type, dictionary, gte, lte
from yargy.pipelines import morph_pipeline
from yargy.interpretation import fact
from yargy.tokenizer import MorphTokenizer


# --- 1. 定义数据结构 ---
@dataclass
class Entry:
    name: str
    birth_date: Optional[str]
    birth_place: Optional[str]


BirthFact = fact(
    'BirthFact',
    ['name', 'birth_date', 'birth_place']
)

# --- 2. 定义规则 (保持不变) ---

# 姓名
NAME = rule(
    and_(gram('Name'), is_capitalized()).repeatable()
).interpretation(BirthFact.name)

# 日期
MONTH_NAME = dictionary({
    'январь', 'февраль', 'март', 'апрель', 'май', 'июнь',
    'июль', 'август', 'сентябрь', 'октябрь', 'ноябрь', 'декабрь'
})
DAY = and_(type('INT'), gte(1), lte(31))  # 简单限制日期
YEAR = and_(type('INT'), gte(1000), lte(2100))  # 简单限制年份
YEAR_WORD = morph_pipeline(['год', 'г.'])

DATE = or_(
    rule(DAY, MONTH_NAME, YEAR, YEAR_WORD.optional()),
    rule(DAY, MONTH_NAME, YEAR),
    rule(DAY, MONTH_NAME),
    rule(YEAR, YEAR_WORD)
).interpretation(BirthFact.birth_date)

# 地点
PLACE = rule(
    and_(gram('Geox'), is_capitalized())
).interpretation(BirthFact.birth_place)

# 辅助词
IN = morph_pipeline(['в', 'во', 'из'])
BIRTH_VERB = morph_pipeline(['родился', 'родилась', 'родились', 'уроженец', 'уроженка'])

# 组合规则
PATTERN_1 = rule(NAME, BIRTH_VERB, DATE, IN, PLACE)
PATTERN_2 = rule(NAME, BIRTH_VERB, IN, PLACE)
PATTERN_3 = rule(BIRTH_VERB, PLACE, NAME)
# 增加一种常见语序：Name (Date) - Place (例如括号中的日期，处理稍微复杂，这里先加一个简单变体)
# PATTERN_4 = rule(NAME, IN, PLACE, BIRTH_VERB) # 可能会误报，暂不加

BIRTH_RULE = or_(PATTERN_1, PATTERN_2, PATTERN_3).interpretation(BirthFact)

# 初始化 Parser
parser = Parser(BIRTH_RULE)

# --- 3. 核心优化：预过滤关键词 ---
# 这些词根必须出现在句子中，否则不可能触发规则
# 注意全部小写
KEYWORDS = ['родил', 'урожен']


def extract_birth_info_fast(file_path):
    entries = []

    # 兼容 .gz 和普通文本
    open_func = gzip.open if file_path.endswith('.gz') else open
    mode = 'rt' if file_path.endswith('.gz') else 'r'

    count = 0
    skipped = 0

    try:
        with open_func(file_path, mode, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # --- 数据清洗 ---
                # 你的文件可能是 category <tab> title <tab> text
                # 或者是 text (如果是多行文本)
                # 简单处理：如果包含 tab，取最后一部分；如果不包含，取全行。
                parts = line.split('\t')
                if len(parts) >= 3:
                    text = parts[-1]  # 取最后一部分通常是正文
                else:
                    text = line

                # --- 性能优化关键点 ---
                # 预先检查：如果文本中不包含 'родил' (birth) 或 'урожен' (native of)，则直接跳过
                text_lower = text.lower()
                if not any(kw in text_lower for kw in KEYWORDS):
                    skipped += 1
                    continue

                # 只有包含关键词的行才送入 Yargy 解析
                for match in parser.findall(text):
                    fact = match.fact
                    entry = Entry(
                        name=fact.name,
                        birth_date=getattr(fact, 'birth_date', None),
                        birth_place=getattr(fact, 'birth_place', None)
                    )
                    entries.append(entry)

                    # 实时打印结果
                    print(f"[Line {count}] Found: {entry.name} | Date: {entry.birth_date} | Place: {entry.birth_place}")

                count += 1
                if count % 1000 == 0:
                    print(f"Processed {count} candidate lines (Skipped {skipped} irrelevant lines)...")

    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

    return entries


if __name__ == '__main__':
    # 你的 news.txt 路径
    data_path = 'news.txt'
    print(f"Fast extracting from {data_path}...")

    results = extract_birth_info_fast(data_path)

    print("-" * 30)
    print(f"Done! Total entries found: {len(results)}")