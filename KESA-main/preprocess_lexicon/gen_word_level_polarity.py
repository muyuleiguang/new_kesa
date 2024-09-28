# generate word level sentiment polarity from SWN 3.0
from collections import defaultdict
from ast import literal_eval


def read_write_SWN():
    r_file = "../../dataset/lexicon/SentiWordNet_3.0.0.txt"

    word_sentiment_dic = defaultdict(dict)
    with open(r_file, 'r', encoding='utf-8') as f:
        for line in f:
            # 去除行尾的换行符和空白字符
            line = line.strip()

            # 跳过空行或注释行
            if not line or line.startswith('#'):
                continue

            # 使用制表符拆分行内容
            parts = line.split('\t')

            # 检查是否有足够的值
            if len(parts) != 6:
                print(f"Warning: Invalid line format: {line}")
                continue

            POS, ID, PosScore, NegScore, SynsetTerms, Gloss = parts
            PosScore, NegScore = float(PosScore), float(NegScore)
            words = SynsetTerms.split()
            for word in words:
                word, num = word.split('#')
                word_sentiment_dic[word].update({num: PosScore - NegScore})

    return word_sentiment_dic


def extract_word_polarity(word_sentiment_dic):
    # 抽取每个word的最强极性
    word_polarity = {}
    for word, polarity_list in word_sentiment_dic.items():
        values = list(polarity_list.values())
        max_value = max(values)
        min_value = abs(min(values))
        if max_value > 0 and max_value > min_value:
            word_polarity[word] = 0
        elif min_value > 0 and min_value > max_value:
            word_polarity[word] = 1

    # fw = open('../../datasets/lexicon/SWN.word.polarity', 'w', encoding='utf-8')
    # fw.write(str(word_polarity))

    return word_polarity


if __name__ == "__main__":
    t1 = read_write_SWN()
    result = extract_word_polarity(t1)
    print(result)  # 输出处理结果
