import time
import redis
import json
import os
import pickle

CONFIG_QUEUE_PATH = 'configs/config_queue/'
CONFIG_END_PATH = 'configs/config_end/'

# Redis 데이터베이스 연결 정보
config = {
    'host': '127.0.0.1',
    'port': 6379
    # 'password': 'redis_password'
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




# r.publish('gpu_task_queue', json.dumps(train_config))




# # 작업 큐에 config 파일 추가
# r.lpush('work_queue', json.dumps(train_config))



######################################################################
# # GPU 작업 생성 및 전송 함수
# def send_task(task):
#     # GPU 작업 전송
#     r.publish('gpu_task_queue', task)
#     print(f"Task {task} sent")

# # GPU 작업 생성 및 전송
# for i in range(10):
#     task = f"Task_{i}"
#     send_task(task)