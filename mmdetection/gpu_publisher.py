import time
import redis
import json
import os
import pickle

CONFIG_QUEUE_PATH = 'configs/config_queue/'
CONFIG_END_PATH = 'configs/config_end/'

# Redis 데이터베이스 연결 정보
config = {
    'host': '49.50.173.166',
    'port': 30003,
    'password': 'foobared'
}

# Redis 클라이언트 생성
r = redis.Redis(**config)

r.delete("work_queue_filename")
r.delete("work_queue_config")
i = 0
while True:
    configs = [file for file in os.listdir(CONFIG_QUEUE_PATH)]
    if configs : 
        print('len configs', len(configs))
        print(f"Config file {configs[0]} is being read")

        with open(os.path.join(CONFIG_QUEUE_PATH, configs[0]), "r") as f:
            config_file = f.read()

            work_queue = r.lrange("work_queue_config", 0, -1)
            print('len q1', len(work_queue))

            r.lpush("work_queue_filename", configs[0])
            r.lpush("work_queue_config", pickle.dumps(config_file))

            work_queue = r.lrange("work_queue_config", 0, -1)
            print('len q2', len(work_queue))

            work_queue = r.lrange("work_queue_config", 0, -1)
            for item in work_queue:
                config_filee =pickle.loads(item)
                print('work queue ', config_filee)
            os.system(f"mv {os.path.join(CONFIG_QUEUE_PATH, configs[0])} {os.path.join(CONFIG_END_PATH, configs[0])}")



