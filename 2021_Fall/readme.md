# Voice-Profiling

--------------------------------------------------

성균관대학교 2021 가을학기 Co-Deep Learning 프로젝트   
SKKU 2021 Fall Semester Co-Deep Learning Project
우수상 수상  

2021-2 P-SAT 딥러닝팀
Advisor : JinYeong Bak (SKKU)

--------------------------------------------------

입력 : 음성이 담겨 있는 wav file  
출력 : 화자의 성별, 연령, 방언 정보  


사용 모델
 - Hard Parameter Sharning CNN  
  - Shared_CNN class in train.py
 ![HPS](https://github.com/victolee0/Voice-Profiling/blob/main/static/asset/Para.png)
 
 
 - CLSTM  
  - CLSTM in train.py
 ![CLSTM](https://github.com/victolee0/Voice-Profiling/blob/main/static/asset/CLSTM.png)
