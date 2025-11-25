# main

from logFetcher import *

log_data = VehicleLog()



# 2차 사전테스트
# fname = setfilename('2025-08-29 08-56-47.log'); flg_cnti = 0; # Laps: 0/0/0, Remarks: 가속/제동
# fname = setfilename('2025-08-29 09-01-23.log'); flg_cnti = 0; # Laps: 0/0/0, Remarks: 동적 성능 1
# fname = setfilename('2025-08-29 09-04-01.log'); flg_cnti = 1; # Laps: 0/0/0, Remarks: 동적 성능 1(이어짐)
# fname = setfilename('2025-08-29 09-05-08.log'); flg_cnti = 1; # Laps: 0/0/0, Remarks: 동적 성능 1(이어짐)
# fname = setfilename('2025-08-29 09-09-27.log'); flg_cnti = 0; # Laps: 0/0/0, Remarks: 동적 성능 2
# fname = setfilename('2025-08-29 09-11-42.log'); flg_cnti = 1; # Laps: 0/0/0, Remarks: 동적 성능 2(이어짐)
# fname = setfilename('.log') % Laps: 0/0/0, Remarks:

file_list = [setfilename('2025-08-29 09-01-23.log'),
             setfilename('2025-08-29 09-04-01.log'),
             setfilename('2025-08-29 09-05-08.log')]
# 이어지는 데이터는 함께 넣을 것.

for idx, fname in enumerate(file_list):
    # ... (파일 열고 n_points 계산하는 로직) ...
    print(f"\nINFO: entering size search loop{idx}... ")
    cnt_tot_sub = 0 #count total 몇 줄인지
    try:
        with open(fname, 'rb') as fid:
            while True:
                read_tmp1 = fid.read(8) #8 bytes 읽기
                
                if not read_tmp1: #비어있는 경우 false. 따라서 read_tmp1(현재 읽고 있는 줄)이 비어있다면 중단한다는 의미
                    break
                
                read_tmp2 = fid.read(8) 

                cnt_tot_sub += 1

        print(f"INFO: size search loop finished. {cnt_tot_sub}")

    except FileNotFoundError:
        print(f"Error: File {fname} not found.")
        continue # 파일 없으면 다음 파일로 넘어감

    is_first = (idx == 0)
    log_data.allocate_or_extend(cnt_tot_sub, is_first)
    log_data.parse_file(fname)



if __name__ == "__main__":    
    print("\n" + "="*30)
    print(" [최종 데이터 확인] ")
    print("="*30)
    
    # 데이터가 잘 들어갔는지 Shape 확인
    if log_data.CAN_set is not None:
        print(f"CAN_set shape: {log_data.CAN_set.shape}")
        # 예시: 첫 번째 데이터의 시간값 출력
        print(f"First timestamp: {log_data.currAct_set[0, 0]}")
    else:
        print("CAN_set is empty.")

    if log_data.currAct_set is not None:
        print(f"currAct_set shape: {log_data.currAct_set.shape}")
    
    # 카운터 확인
    print(f"Total Sources Counted: {log_data.cnt_source}")
    print(log_data.currAct_set)
