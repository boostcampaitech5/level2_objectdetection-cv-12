import time
import redis
import json
import os
import pickle

PUBLISHER_FILES = 'configs/publisher_files'
# CONFIG_QUEUE_PATH = 'configs/config_queue/'

# Redis 데이터베이스 연결 정보
config = {
    'host': '49.50.173.166',
    'port': 30003,
    'password': 'ganddddiganddddi'
}

# Redis 클라이언트 생성
r = redis.Redis(**config)

# Redis stream key 

STREAM_KEY = 'work_queue'

# work_queue에 있는 file name들 출력
work_queue = r.xrange(STREAM_KEY)
print(f'#### Current works (Uncompleted works) in queue: {len(work_queue)-1} ####')
for work_id, work_info_dict in work_queue:
    # work_info_dict : dict_items([(b'filename', b'config')])
    work_info = list(work_info_dict.items())[0]
    filename = work_info[0].decode()
    # filename = filename[0]
    print(filename)
print()

# 해당 경로 파일 이름
configs = [file for file in os.listdir(PUBLISHER_FILES)]

result = 0
# 파일 하나씩 접근해서 read 
for config in configs:
    push_work = {}
    
    with open(os.path.join(PUBLISHER_FILES, config), "r") as f:
        config_file = f.read()
    
    print(f"PUSH {config}")

    # 형식 = {filename : configs 내용}     
    push_work[config] = pickle.dumps(config_file)
    
    # work_queue에 push할 config 저장
    result = r.xadd(STREAM_KEY, push_work)

print()
if result:
    print('success push')
else:
    print('failed push')

