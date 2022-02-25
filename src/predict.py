# -*- coding: utf-8 -*-
import os
import warnings

import hydra
import pandas as pd
from transformers import AutoTokenizer

from models.MainModels import EncoderModel

warnings.filterwarnings(action="ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


@hydra.main(config_name="config.yml")
def main(cfg):
    print(f"load tokenizer...", end=" ")
    tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_dir)
    print("done!")

    model = EncoderModel.load(cfg.ETC.output_dir2)

    sentences = pd.read_csv("./data/category_initial.csv")
    for sentence in sentences["content"][10:27]:
        data = tokenizer(
            sentence,
            max_length=cfg.MODEL.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="np",
        )

        labels = [
            "인공지능",
            "로봇",
            "스마트팜",
            "에너지",
            "서버",
            "투자",
            "정부지원",
            "증강현실",
            "이동수단",
            "개발",
            "통신",
            "과학",
            "드론",
            "블록체인",
            "핀테크",
            "커머스",
            "여행",
            "미디어",
            "헬스케어",
            "의약",
            "식품",
            "교육",
            "직업",
            "경제",
            "광고",
            "제약",
            "O2O",
            "뷰티",
            "부동산",
            "etc",
        ]
        pred = model(data)[0]

        result = {}
        
        for (key, l), p in zip(enumerate(labels), pred.numpy()):
            if p >= 0.8:
                result[str(key) + " " + str(l)] = format(p, ".4f")
        sorted_result = dict(
            sorted(result.items(), key=lambda item: item[1], reverse=True)
        )

        print("--------------predict--------------")
        print(sentence)
        print(sorted_result)


if __name__ == "__main__":
    main()
