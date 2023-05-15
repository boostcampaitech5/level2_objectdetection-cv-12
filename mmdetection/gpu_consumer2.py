
import redis
import pickle
import os
import subprocess
from pathlib import Path

# Redis 데이터베이스 연결 정보
config = {
    'host': '49.50.173.166',
    'port': 30003,
    'password': 'ganddddiganddddi'
}

CONSUMER_FILES = 'configs/publisher_files'
# CONFIG_QUEUE_PATH = 'configs/work_queue_client'
STREAM_KEY = 'work_queue'
GROUP_NAME = 'ConsumerGroup'
CONSUMER_NAME = 'consumer-hm'

r = redis.Redis(**config)


config_file = None
config_filename = None
work = None

while True:

    # pending을 읽는다. 만약 consumer에 해당하는 정보가 있으면 처리한다. 

    # group 값 읽기
    work = r.xreadgroup(
        groupname=GROUP_NAME, 
        consumername=CONSUMER_NAME,
        count=1, 
        streams={STREAM_KEY:'>'}, 
        block=0)
    
    work_id, work_info_dict = work[0][1][0]            
    work_info = list(work_info_dict.items())[0]
    filename = work_info[0].decode()
    config = pickle.loads(work_info[1])
    
    print(f"#### Start Training {filename} ####")
    print(config)
    print()

    total_filename = os.path.join(CONSUMER_FILES, filename)
    file_path = Path(total_filename)

    # 기존에 동일한 파일명 있는 경우 1씩 더함
    if file_path.is_file():
        count = 1
        while True:
            new_filename = f'{Path(file_path.stem)}_{count}{file_path.suffix}'
            new_file_path = file_path.with_name(new_filename)
            if not new_file_path.is_file():
                break
            count += 1
        file_path = new_file_path
    
    with open(file_path, 'w') as f:
        f.write(config)
    
    # 실행할 명령어를 정의합니다.
    command = "python3 train.py --config_dir {}".format(file_path)

    # 명령어를 실행합니다.
    subprocess.run(command, shell=True)
    
    print(f"#### Finished Training {file_path} ####")
        
    # 해당 work 처리 완료 message를 redis에게 전송 
    r.xack(STREAM_KEY, 
            GROUP_NAME, 
            work_id)
    
    # work에서 
    r.xdel(STREAM_KEY, work_id)
    
    work = None
                
    
