import json
import logging
import os
import re
import time
from logging import handlers

import numpy as np
import pymysql
import requests

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


import hydra
import pandas as pd
import tensorflow as tf
from pandas import json_normalize
from transformers import AutoTokenizer

from models.MainModels import EncoderModel

# definition
# tensorflow 메모리 증가를 허용
gpus = tf.config.experimental.list_physical_devices("GPU")

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


# log setting
carLogFormatter = logging.Formatter("%(asctime)s,%(message)s")

carLogHandler = handlers.TimedRotatingFileHandler(
    filename="/home/ubuntu/news-label-serving/log/predict.log",
    when="midnight",
    interval=1,
    encoding="utf-8",
)
carLogHandler.setFormatter(carLogFormatter)
carLogHandler.suffix = "%Y%m%d"

scarp_logger = logging.getLogger()
scarp_logger.setLevel(logging.INFO)
scarp_logger.addHandler(carLogHandler)


def processing(content):
    result = re.sub(r"[a-zA-Z가-힣]+뉴스", "", str(content))
    result = re.sub(r"[a-zA-Z가-힣]+ 뉴스", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+newskr", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+Copyrights", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+ Copyrights", "", result)
    result = re.sub(r"\s+Copyrights", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+com", "", result)
    result = re.sub(r"[가-힣]+ 기자", "", result)
    result = re.sub(r"[가-힣]+기자", "", result)
    result = re.sub(r"[가-힣]+ 신문", "", result)
    result = re.sub(r"[가-힣]+신문", "", result)
    result = re.sub(r"데일리+[가-힣]", "", result)
    result = re.sub(r"[가-힣]+투데이", "", result)
    result = re.sub(r"[가-힣]+미디어", "", result)
    result = re.sub(r"[가-힣]+ 데일리", "", result)
    result = re.sub(r"[가-힣]+데일리", "", result)
    result = re.sub(r"[가-힣]+ 콘텐츠 무단", "", result)
    result = re.sub(r"전재\s+변형", "전재", result)
    result = re.sub(r"[가-힣]+ 전재", "", result)
    result = re.sub(r"[가-힣]+전재", "", result)
    result = re.sub(r"[가-힣]+배포금지", "", result)
    result = re.sub(r"[가-힣]+배포 금지", "", result)
    result = re.sub(r"\s+배포금지", "", result)
    result = re.sub(r"\s+배포 금지", "", result)
    result = re.sub(r"[a-zA-Z가-힣]+.kr", "", result)
    result = re.sub(r"/^[a-z0-9_+.-]+@([a-z0-9-]+\.)+[a-z0-9]{2,4}$/", "", result)
    result = re.sub(r"[\r|\n]", "", result)
    result = re.sub(r"\[[^)]*\]", "", result)
    result = re.sub(r"\([^)]*\)", "", result)
    result = re.sub(r"[^ ㄱ-ㅣ가-힣A-Za-z0-9]", "", result)
    result = (
        result.replace("뉴스코리아", "")
        .replace("및", "")
        .replace("Copyright", "")
        .replace("저작권자", "")
        .replace("ZDNET A RED VENTURES COMPANY", "")
        .replace("\n", "")
    )
    result = result.strip()

    return result


# label
labels = {
    "0": "인공지능",
    "1": "로봇",
    "2": "스마트팜",
    "3": "에너지",
    "4": "서버",
    "5": "투자",
    "6": "정부지원",
    "7": "증강현실",
    "8": "이동수단",
    "9": "개발",
    "10": "통신",
    "11": "과학",
    "12": "드론",
    "13": "블록체인",
    "14": "핀테크",
    "15": "커머스",
    "16": "여행",
    "17": "미디어",
    "18": "헬스케어",
    "19": "의약",
    "20": "식품",
    "21": "교육",
    "22": "직업",
    "23": "경제",
    "24": "광고",
    "25": "제약",
    "26": "O2O",
    "27": "뷰티",
    "28": "부동산",
    "29": "etc",
}


def data_load(**kwargs):
    try:
        logging.info("dataload start")
        conn = pymysql.connect(
            user=kwargs.get("user"),
            passwd=kwargs.get("passwd"),
            db=kwargs.get("db"),
            host=kwargs.get("host"),
            port=kwargs.get("port"),
            charset="utf8",
            use_unicode=True,
        )

        cursor = conn.cursor(pymysql.cursors.DictCursor)
        
        cursor.execute(kwargs.get("query"))

        data = pd.DataFrame(cursor.fetchall(), columns=["id", "content", "create_date"])
        data = data[["id", "content", "create_date"]]
        logging.info(data.head())

        data["content"] = data.content.apply(processing)
        df = data.drop_duplicates()
        logging.info("dataload end")

        return df

    except Exception as e:
        logging.info(e)


def curl(param, api_url):
    logging.info("curl start")
    headers = {
        "content-type": "application/json",
    }
    param = json.dumps(param)

    response = requests.post(
        api_url, headers=headers, data=param
    )
    logging.info("curl end")

    return response


@hydra.main(config_name="config.yml")
def predict(cfg):
    try:
        # start time
        start = time.time()

        # dataload
        df = data_load(**cfg.AWS)
        
        logging.info("model load start")
        tokenizer = AutoTokenizer.from_pretrained(cfg.ETC.tokenizer_dir)
        model = EncoderModel.load(cfg.ETC.output_dir)

        data = tokenizer(
            df["content"].to_list(),
            max_length=model.seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="tf",
        )

        logging.info("start predict")
        pred = model.predict(dict(data))

        label = [
            [j for j in r if pred[i, j] >= 0.8]
            for i, r in enumerate(np.argsort(pred)[:, :-4:-1])
        ]
        df["label"] = label
        df["label"] = df.label.apply(lambda x: [29] if len(x) == 0 else x)
        df["predict"] = df.label.apply(
            lambda x: ", ".join(labels.get(str(e)) for e in x)
        )

        # dataframe to update parameter
        param_dict = dict()
        # db info
        conn_dict = dict()
        conn_dict["user"] = cfg.DATABASE.user
        conn_dict["passwd"] = cfg.DATABASE.passwd
        conn_dict["db"] = cfg.DATABASE.db
        conn_dict["host"] = cfg.DATABASE.host
        conn_dict["port"] = cfg.DATABASE.port
        conn_dict["charset"] = "utf8"
        conn_dict["use_unicode"] = "True"
        conn_dict["query"] = cfg.DATABASE.query
        

        # where value
        param_list = list()
        predict_list = list()
        for i in range(len(df)):
            predict_list = list()
            predict_list.append(df["predict"][i])
            predict_list.append(str(df["id"][i]))
            param_list.append(predict_list)
        param_dict["param"] = str(param_list)
        param_dict["conn"] = conn_dict

        logging.info(param_list[:10])
        logging.info("end predict")

        response = curl(param_dict, cfg.AWS.api)
        logging.info(response)

        # end time
        logging.info("time :" + str(time.time() - start))

    except Exception as e:
        logging.info(e)
        return 200


if __name__ == "__main__":
    predict()
