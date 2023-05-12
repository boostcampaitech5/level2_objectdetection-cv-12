
import redis
import pickle
import os
import subprocess

# Redis 연결 설정
config = {
    'host': '127.0.0.1',
    'port': 6379
    # 'password': 'redis_password'
}

CONFIG_QUEUE_PATH = 'configs/work_queue_client'

r = redis.Redis(**config)

config_file = None
config_filename = None

while True:
    if config_file is None :
        config_file = r.rpop('work_queue_config')
    if config_filename is None:
        config_filename = r.rpop('work_queue_filename')

    if config_file is not None and config_filename is not None:

        # Pickle 데이터를 Python 객체로 변환
        config = pickle.loads(config_file)
        config_filename = config_filename.decode()

        with open(os.path.join(CONFIG_QUEUE_PATH, config_filename), 'w') as f:
            f.write(config)
            
        # 실행할 명령어를 정의합니다.
        command = "python3 train.py --config_dir {}".format(os.path.join(CONFIG_QUEUE_PATH, config_filename))
        
        # 명령어를 실행합니다.
        subprocess.run(command, shell=True)

        # 결과를 Redis 서버로 다시 전송합니다.
        result = {"config": config, "result": "gpu_task_done"}
        r.publish("gpu_task_result", str(result))

        config_file = None
        config_filename = None
