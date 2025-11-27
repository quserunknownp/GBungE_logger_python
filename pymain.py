# main

from pylogFetcher import *
from pylogPostProcessor import LogVisualizer

log_data = VehicleLog()



# 2차 사전테스트
# fname = setfilename('2025-08-29 08-56-47.log'); flg_cnti = 0; # Laps: 0/0/0, Remarks: 가속/제동
# fname = setfilename('2025-08-29 09-01-23.log'); flg_cnti = 0; # Laps: 0/0/0, Remarks: 동적 성능 1
# fname = setfilename('2025-08-29 09-04-01.log'); flg_cnti = 1; # Laps: 0/0/0, Remarks: 동적 성능 1(이어짐)
# fname = setfilename('2025-08-29 09-05-08.log'); flg_cnti = 1; # Laps: 0/0/0, Remarks: 동적 성능 1(이어짐)
# fname = setfilename('2025-08-29 09-09-27.log'); flg_cnti = 0; # Laps: 0/0/0, Remarks: 동적 성능 2
# fname = setfilename('2025-08-29 09-11-42.log'); flg_cnti = 1; # Laps: 0/0/0, Remarks: 동적 성능 2(이어짐)
# fname = setfilename('.log') % Laps: 0/0/0, Remarks:
# % fname = '2025-08-30 01-58-39.log'; % 2일차 온도 터진거(테스트주행)
# % fname = '2025-08-30 06-08-15.log'; % 오토크로스
# % fname = '2025-08-30 08-32-10.log'; % 예선
# % fname = '2025-08-31 01-13-33.log'; % 본선1 ( 충격으로 꺼짐)
# % fname = '2025-08-31 01-33-25.log'; % 본선2 (재시작 후 피트인)
# % fname = '2025-08-31 01-48-43.log'; % 본선3 (김경민 10분)
#  fname = '2025-08-31 02-03-58.log'; % 본선4 (임동윤, 임동윤, 김경민) 
# % fname = '2025-08-31 03-07-25.log'; %

file_list = [setfilename('2025-08-31 02-03-58.log')]
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
    # uncomment plot you want to view.

    visualizer = LogVisualizer(log_data)
    
    # visualizer.plot_gps_only()
    
    # print("Graph 1: 토크 응답성 확인")
    # visualizer.plot_torque_performance()
    
    # print("Graph 2: 벡터 제어(Id/Iq) 상태 확인")
    # visualizer.plot_vector_control()

    # print("Graph 3: 약계자 제어")
    # visualizer.plot_field_weakening()

    # print("gps 속도 slip ratio")
    # visualizer.plot_gps_velocity_and_slip()

    # visualizer.plot_torque_vs_rpm()

    # visualizer.plot_temperature_profile()

    # visualizer.plot_torque_vs_temperature()

    # visualizer.plot_current_vs_torque_efficiency()

    # visualizer.plot_current_efficiency()

    # visualizer.plot_advanced_id_iq_analysis()

    # visualizer.plot_vehicle_dynamics()

    # visualizer.plot_vehicle_dynamics_lpf()
    
    # visualizer.plot_vehicle_dynamics_mv_avg()

    # log_data.split_laps()

    # visualizer.plot_gps_gforce_map()   

    # visualizer.plot_laps_slideshow()

    # visualizer.plot_power_and_temp()
    
    # visualizer.analyze_moving_rms()

    # visualizer.plot_temp_rise_vs_power()

    # visualizer.plot_temp_slope_trend()

    # visualizer.plot_thermal_path()

    # visualizer.plot_thermal_path_v2()

    visualizer.plot_power_vs_temp_slope()

    # visualizer.plot_cooling_trend_regression()

    # visualizer.plot_cooling_intercept()
