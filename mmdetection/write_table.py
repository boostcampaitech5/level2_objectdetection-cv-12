import mysql.connector
import json

# MySQL 데이터베이스 연결 정보
config = {
    'user': 'user_name',
    'password': 'password',
    'host': 'localhost',
    'database': 'database_name'
}

# MySQL 데이터베이스 연결
conn = mysql.connector.connect(**config)

# 데이터베이스 커서 생성
cursor = conn.cursor()

# task_queue 테이블에서 is_done이 False인 레코드 중에서 model_config 필드 추출
query = "SELECT model_config FROM task_queue WHERE is_done = False"
cursor.execute(query)
model_configs = cursor.fetchall()

# 추출한 model_config 필드를 이용하여 모델 학습 수행
for model_config in model_configs:
    # model_config 필드를 JSON 형태로 변환
    model_config = json.loads(model_config[0])
    
    # 모델 학습 수행
    train_model(model_config)
    
    # 학습이 완료된 경우, is_done을 True로 업데이트
    query = "UPDATE task_queue SET is_done = True WHERE model_config = %s"
    cursor.execute(query, (json.dumps(model_config),))
    conn.commit()

# 데이터베이스 연결 종료
cursor.close()
conn.close()