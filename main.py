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


    cnt_tot_sub
    
    # MATLAB의 if/else 로직이 이 한 줄로 끝납니다!
    is_first = (idx == 0)
    log_data.allocate_or_extend(cnt_tot_sub, is_first)


if __name__ == "__main__":    
    print(len(log_data.CAN_set[0]))
    
