import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt  # [필수] 신호처리 라이브러리

class LogVisualizer:
    def __init__(self, log_data):
        self.data = log_data

    def plot_gps_only(self):
        """기본 GPS 주행 궤적 그리기 (Path Only)"""
        
        # 1. 데이터 존재 확인
        if self.data.gpsPos_set is None:
            print("GPS 데이터가 없습니다.")
            return

        # 2. 데이터 추출
        # Row 1: Longitude (경도, X축)
        # Row 2: Latitude (위도, Y축)
        lon = self.data.gpsPos_set[1, :]
        lat = self.data.gpsPos_set[2, :]

        # (옵션) GPS 초기화 전 0,0 튀는 값 제거 (필요시 주석 해제)
        # mask = (lon > 1) & (lat > 1) # 0 근처 값 제외
        # lon = lon[mask]
        # lat = lat[mask]

        # 3. 그래프 그리기
        plt.figure(figsize=(10, 10))

        # 궤적 그리기 (파란색 실선)
        plt.plot(lon, lat, 'b-', linewidth=2, label='Vehicle Path')

        # 시작점(Start)과 끝점(End) 표시
        # 주행 방향을 헷갈리지 않게 도와줍니다.
        if len(lon) > 0:
            plt.plot(lon[0], lat[0], 'go', markersize=12, label='Start')
            plt.plot(lon[-1], lat[-1], 'rx', markersize=12, markeredgewidth=3, label='End')

        # 꾸미기
        plt.title('GPS Tracking Path')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        
        # [중요] 지도 비율 고정 (안 하면 찌그러져 보임)
        plt.axis('equal') 
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_torque_performance(self):
        """1. 토크 추종성 및 속도 분석 (Torque Tracking)"""
        if self.data.Tdmd_set is None: return

        # 데이터 추출
        t_cmd   = self.data.Tdmd_set[0, :]
        trq_cmd = self.data.Tdmd_set[1, :]   # Demand Torque
        
        t_act   = self.data.torqueAct_set[0, :]
        trq_act = self.data.torqueAct_set[1, :] # Actual Torque
        
        t_vel   = self.data.vel_set[0, :]
        vel     = self.data.vel_set[1, :]       # Velocity (RPM 등)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 상단: 토크 비교
        ax1.plot(t_cmd, trq_cmd, 'r--', label='Target (Tdmd)', linewidth=1.5)
        ax1.plot(t_act, trq_act, 'b-',  label='Actual (Tact)', linewidth=1)
        ax1.set_ylabel('Torque (Nm)')
        ax1.set_title('Torque Response Analysis')
        ax1.legend()
        ax1.grid(True)

        # 하단: 속도
        ax2.plot(t_vel, vel, 'k-')
        ax2.set_ylabel('Motor Speed') # 단위(RPM) 확인 필요
        ax2.set_xlabel('Time (sec)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_vector_control(self):
        """2. 벡터 제어 상태 분석 (Id, Iq Currents) - 가장 전문적인 분석"""
        if self.data.Idq_set is None: return

        # Idq_set 구조: [Time, Id, Iq] 라고 가정 (3행)
        t_dq = self.data.Idq_set[0, :]
        id_curr = self.data.Idq_set[1, :]
        iq_curr = self.data.Idq_set[2, :]
        
        # DC Link Voltage (Vcap)
        t_vcap = self.data.Vcap_set[0, :]
        vcap   = self.data.Vcap_set[1, :]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # 상단: d-q축 전류
        ax1.plot(t_dq, iq_curr, 'r', label='Iq (Torque)', alpha=0.8)
        ax1.plot(t_dq, id_curr, 'b', label='Id (Flux)', alpha=0.8)
        ax1.set_ylabel('Current (A)')
        ax1.set_title('FOC Vector Control Analysis (Id, Iq)')
        ax1.legend()
        ax1.grid(True)
        # 팁: 고속에서 Id가 음수로 떨어지면 '약계자 제어' 중인 것임

        # 하단: DC Link 전압
        ax2.plot(t_vcap, vcap, 'g-')
        ax2.set_ylabel('DC Link Voltage (V)')
        ax2.set_xlabel('Time (sec)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()

    def plot_power_flow(self):
        """3. 전력 흐름 및 배터리 전류 분석"""
        if self.data.Ibatt_set is None: return
        
        t_batt = self.data.Ibatt_set[0, :]
        i_batt = self.data.Ibatt_set[1, :]
        
        # 기계적 출력 파워 계산 (P = Torque * w)
        # 시간축을 맞춰야 하므로 간단히 interpolation하거나, 
        # 샘플링이 같다면 인덱스로 접근 (여기서는 길이 체크 후 계산 추천)
        
        # 예시로 배터리 전류만 도시 (파워 계산은 단위 변환 필요)
        plt.figure(figsize=(10, 5))
        plt.plot(t_batt, i_batt, 'm-')
        plt.title('Battery Current Consumption')
        plt.ylabel('Current (A)')
        plt.xlabel('Time (sec)')
        plt.grid(True)
        plt.show()
    
    def plot_field_weakening(self):
        """약계자 제어(Field Weakening) 정밀 분석"""
        # 데이터가 없으면 리턴
        if self.data.Idq_set is None or self.data.Vmod_set is None:
            print("약계자 분석에 필요한 데이터(Idq, Vmod)가 부족합니다.")
            return

        # 1. 데이터 추출
        t_vel = self.data.vel_set[0, :]
        vel   = self.data.vel_set[1, :]       # RPM
        
        t_dq  = self.data.Idq_set[0, :]
        id_curr = self.data.Idq_set[1, :]     # d-axis Current (Flux)
        iq_curr = self.data.Idq_set[2, :]     # q-axis Current (Torque)

        t_vmod = self.data.Vmod_set[0, :]
        vmod   = self.data.Vmod_set[1, :]     # Modulation Voltage
        
        # (옵션) DC Link 전압이 있다면 MI(변조지수) 계산 가능
        if self.data.Vcap_set is not None:
            # 시간축이 다를 수 있으므로 여기서는 단순 비교용으로 따로 그림
            vcap = self.data.Vcap_set[1, :] # DC Link Voltage
        
        # 2. 그래프 그리기 (3단 구성)
        fig, axs = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

        # [Graph 1] 속도 (RPM)
        axs[0].plot(t_vel, vel, 'k-', linewidth=2)
        axs[0].set_ylabel('Speed (RPM)')
        axs[0].set_title('1. Motor Speed (High Speed Check)')
        axs[0].grid(True)
        
        # [Graph 2] d-q 전류 (핵심!)
        # Id가 음수로 떨어지는지 봐야 하므로 0점 기준선을 그어줌
        axs[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5) 
        axs[1].plot(t_dq, iq_curr, 'r-', label='Iq (Torque)', alpha=0.7)
        axs[1].plot(t_dq, id_curr, 'b-', label='Id (Flux Weakening)', linewidth=2)
        axs[1].set_ylabel('Current (A)')
        axs[1].set_title('2. Vector Currents (Check Negative Id)')
        axs[1].legend(loc='upper right')
        axs[1].grid(True)

        # [Graph 3] 전압 사용률 (Vmod)
        axs[2].plot(t_vmod, vmod, 'g-', label='Vmod (Output Voltage)')
        
        # 만약 Vcap 데이터가 있다면 전압 한계선(Vdc_link)을 같이 그려줌
        if self.data.Vcap_set is not None:
             # 샘플링 개수가 맞다고 가정하고 그림 (다르면 보간 필요)
            axs[2].plot(self.data.Vcap_set[0, :], self.data.Vcap_set[1, :], 
                        'm--', label='Vdc (Limit)', alpha=0.6)
            
        axs[2].set_ylabel('Voltage (V)')
        axs[2].set_xlabel('Time (sec)')
        axs[2].set_title('3. Voltage Utilization (Saturation Check)')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        plt.show()
    
    def plot_gps_velocity_and_slip(self):
        """GPS 속도와 슬립율(Slip Ratio) 이중축 그래프"""
        
        # 1. 데이터 확인
        if self.data.gpsVec_set is None or self.data.CAN_set is None:
            print("데이터가 부족하여 슬립율 그래프를 그릴 수 없습니다.")
            return

        # 2. 데이터 추출
        # GPS Speed (Time, Speed)
        t_gps = self.data.gpsVec_set[0, :]
        gps_kph = self.data.gpsVec_set[1, :]

        # Motor RPM (Time, RPM) -> Wheel Speed로 변환 필요
        t_rpm = self.data.CAN_set[0, :]
        rpm = self.data.CAN_set[4, :]

        # --- [슬립율 계산 로직] ---
        # 실제 차량 파라미터가 필요합니다. (임의값 적용)
        # Wheel Speed (kph) = RPM * (2*pi/60) * r_tire * 3.6 / Gear_Ratio
        r_tire = 0.3      # 타이어 반지름 (m) 예시
        gear_ratio = 9.0  # 감속비 예시
        
        # RPM 데이터를 GPS 시간축에 맞춰야 정확한 계산이 가능하지만,
        # 여기서는 샘플링이 비슷하다고 가정하고 계산하거나, 
        # 단순히 RPM 자체를 비교용으로 씁니다. (정석은 interp1로 시간 동기화 필요)
        
        # 간단한 변환 (RPM -> KPH)
        wheel_kph = rpm * (2 * np.pi / 60) * r_tire * (1 / gear_ratio) * 3.6
        
        # 슬립율 계산: (Wheel - Vehicle) / Vehicle * 100
        # 분모가 0일 때 에러 방지를 위해 1e-5 더함
        slip_ratio = (wheel_kph - gps_kph) / (np.abs(gps_kph) + 1.0) * 100
        
        # 노이즈가 심할 수 있으므로 범위 제한 (MATLAB의 ylim 역할)
        slip_ratio = np.clip(slip_ratio, -20, 50) 


        # --- [그래프 그리기] ---
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # [Left Axis] GPS Velocity
        # MATLAB: plot(..., 'LineStyle','none','Marker','.','Color','k')
        color_1 = 'black'
        ax1.set_xlabel('Time (sec)')
        ax1.set_ylabel('GPS Speed (km/h)', color=color_1)
        ax1.plot(t_gps, gps_kph, color=color_1, linestyle='none', marker='.', markersize=3, label='GPS Speed')
        ax1.tick_params(axis='y', labelcolor=color_1)
        ax1.grid(True) # 격자는 왼쪽 축 기준

        # [Right Axis] Slip Ratio (핵심: twinx 사용)
        ax2 = ax1.twinx()  # X축을 공유하는 쌍둥이 축 생성
        
        color_2 = 'blue'
        ax2.set_ylabel('Slip Ratio (%)', color=color_2)
        # MATLAB: plot(..., 'LineStyle','-','Marker','none')
        # RPM 시간축(t_rpm)을 사용
        ax2.plot(t_rpm, slip_ratio, color=color_2, linestyle='-', linewidth=1, alpha=0.6, label='Slip Ratio')
        ax2.tick_params(axis='y', labelcolor=color_2)
        
        # MATLAB: yline(0, 'b') -> 수평선 0
        ax2.axhline(0, color='blue', linestyle='--', linewidth=0.8)

        # MATLAB: xline(...) -> 수직선 (예: 이벤트 발생 지점)
        # if ~isempty(tValSeg_LS) 구현 (예시로 10초, 20초에 선 그리기)
        event_times = [10, 20] # 예시 데이터
        for et in event_times:
            ax1.axvline(x=et, color='red', linestyle='--', linewidth=1.5)

        # 제목 및 마무리
        plt.title('GPS Velocity vs Slip Ratio Analysis')
        fig.tight_layout()  # 레이아웃 자동 정렬
        plt.show()

    def plot_torque_vs_rpm(self):
        """RPM에 따른 토크 분포 (Torque Map Analysis)"""
        
        # 1. 데이터 확인
        if self.data.vel_set is None or self.data.Tdmd_set is None:
            print("데이터가 부족하여 토크 맵을 그릴 수 없습니다.")
            return

        # 2. 데이터 추출
        # 기준이 되는 X축 데이터 (Velocity)
        t_vel = self.data.vel_set[0, :]  # 시간 (Time reference)
        rpm   = self.data.vel_set[1, :]  # RPM (X-axis value)

        # Y축 데이터 1 (Demand Torque)
        t_dmd   = self.data.Tdmd_set[0, :]
        val_dmd = self.data.Tdmd_set[1, :]

        # Y축 데이터 2 (Actual Torque)
        t_act   = self.data.torqueAct_set[0, :]
        val_act = self.data.torqueAct_set[1, :]

        # --- [핵심] 데이터 시간 동기화 (Interpolation) ---
        # MATLAB: interp1(x, y, x_new)
        # Python: np.interp(x_new, x, y)  <-- 순서가 다릅니다! 주의!
        
        # "RPM이 측정된 그 시간(t_vel)에 토크는 몇이었니?"를 계산
        dmd_interp = np.interp(t_vel, t_dmd, val_dmd)
        act_interp = np.interp(t_vel, t_act, val_act)

        # 3. 그래프 그리기
        plt.figure(figsize=(10, 8))
        
        # Demand Torque (파란 점)
        # MATLAB: LineStyle='none', Marker='.'
        plt.plot(rpm, dmd_interp, color='blue', linestyle='None', marker='.', 
                 markersize=3, label='Dmd (Target)')
        
        # Actual Torque (주황 점)
        plt.plot(rpm, act_interp, color='orange', linestyle='None', marker='.', 
                 markersize=3, label='Act (Actual)')

        # 꾸미기
        plt.grid(True)
        plt.legend(loc='best')
        plt.xlabel('Speed (RPM)')   # MATLAB: rpm_{CAN}
        plt.ylabel('Torque (Nm)')
        plt.title('Torque vs Speed Distribution')
        
        plt.show()

    def plot_temperature_profile(self):
        """시간에 따른 모터 온도 변화 그래프"""
        
        # 1. 데이터 확인
        if self.data.motorTemp_set is None:
            print("모터 온도 데이터(motorTemp_set)가 없습니다.")
            return

        # 2. 데이터 추출
        # motorTemp_set: [Time, Temperature]
        t_temp = self.data.motorTemp_set[0, :]
        temp_val = self.data.motorTemp_set[1, :]

        # 3. 그래프 그리기
        plt.figure(figsize=(12, 6))
        
        # 메인 온도 그래프
        plt.plot(t_temp, temp_val, color='tab:red', linewidth=2, label='Motor Temp')
        
        # [분석 팁] ME1616 / Sevcon의 일반적인 제한 온도 표시
        # 보통 120도~130도 부근에서 토크 컷이 시작됩니다.
        plt.axhline(y=120, color='orange', linestyle='--', label='Warning (120°C)')
        plt.axhline(y=140, color='red', linestyle='--', label='Limit (140°C)')

        # 꾸미기
        plt.title('Motor Temperature Profile over Time')
        plt.xlabel('Time (sec)')
        plt.ylabel('Temperature (°C)')
        plt.grid(True)
        plt.legend()
        
        # 현재 최고 온도 표시 (텍스트)
        max_temp = np.max(temp_val)
        plt.text(t_temp[-1], max_temp, f' Max: {max_temp:.1f}°C', 
                 verticalalignment='bottom', color='red', fontweight='bold')

        plt.tight_layout()
        plt.show()

    def plot_torque_vs_temperature(self):
        """안전 로직 확인: 온도에 따른 토크 제한(Derating) 확인"""
        if self.data.motorTemp_set is None or self.data.torqueAct_set is None:
            return

        # 데이터 추출 및 시간 동기화
        t_vel = self.data.vel_set[0, :] # 기준 시간축
        
        t_temp = self.data.motorTemp_set[0, :]
        temp_val = self.data.motorTemp_set[1, :]
        
        t_trq = self.data.torqueAct_set[0, :]
        trq_val = self.data.torqueAct_set[1, :]
        
        # 인터폴레이션
        temp_interp = np.interp(t_vel, t_temp, temp_val)
        trq_interp = np.interp(t_vel, t_trq, trq_val)

        plt.figure(figsize=(10, 8))
        
        # 산점도 그리기
        plt.scatter(temp_interp, trq_interp, alpha=0.3, s=3, c='orange')
        
        plt.xlabel('Motor Temperature (°C)')
        plt.ylabel('Actual Torque (Nm)')
        plt.title('Thermal Derating Check (Torque vs Temp)')
        plt.grid(True)
        
        # 가이드라인 (예상되는 제한선)
        plt.axvline(x=110, color='r', linestyle='--', label='Derating Start (Expected)')
        plt.legend()
        
        plt.show()

    def plot_current_vs_torque_efficiency(self):
        """효율 분석: 토크 대비 소모 전류량"""
        if self.data.torqueAct_set is None or self.data.Idq_set is None:
            return

        # 데이터 추출
        t_trq = self.data.torqueAct_set[0, :]
        trq = self.data.torqueAct_set[1, :]
        
        t_iq = self.data.Idq_set[0, :]
        iq = self.data.Idq_set[2, :] # 토크 생성에 쓰인 전류
        
        # 시간 동기화
        iq_interp = np.interp(t_trq, t_iq, iq)

        plt.figure(figsize=(10, 6))
        
        # X축: 실제 토크, Y축: 소모 전류 (Iq)
        plt.scatter(trq, iq_interp, alpha=0.1, s=3, c='purple')
        
        plt.xlabel('Actual Torque (Nm)')
        plt.ylabel('Current Iq (A)')
        plt.title('Current Consumption per Torque (Lower is Better)')
        plt.grid(True)
        
        # 이상적인 선 (Reference)
        # ME1616 Kt ~= 0.23 Nm/A  =>  Current = Torque / 0.23
        x_ref = np.linspace(0, 100, 100)
        plt.plot(x_ref, x_ref / 0.23, 'g--', label='Ideal Efficiency Line')
        plt.legend()
        
        plt.show()

    def plot_current_efficiency(self):
        """배터리 전류 vs 모터 상전류 비교 (인버터 효율 확인)"""
        # 데이터 존재 확인
        if self.data.Idq_set is None or self.data.Ibatt_set is None:
            print("전류 분석에 필요한 데이터(Idq, Ibatt)가 없습니다.")
            return

        # 1. 데이터 추출
        # 배터리 전류 (DC)
        t_batt = self.data.Ibatt_set[0, :]
        i_batt = self.data.Ibatt_set[1, :]
        
        # 모터 전류 (Vector Control -> Phase Current Magnitude)
        t_dq = self.data.Idq_set[0, :]
        id_curr = self.data.Idq_set[1, :]
        iq_curr = self.data.Idq_set[2, :]
        
        # 상전류 크기 계산 (Magnitude = sqrt(Id^2 + Iq^2))
        i_phase_calc = np.sqrt(id_curr**2 + iq_curr**2)
        
        # 시간 동기화 (배터리 전류를 모터 전류 시간축에 맞춤)
        i_batt_interp = np.interp(t_dq, t_batt, i_batt)

        # 2. 그래프 그리기
        plt.figure(figsize=(12, 6))
        
        # 모터로 들어가는 전류 (힘)
        plt.plot(t_dq, i_phase_calc, 'm-', label='Phase Current (Motor Force)', alpha=0.8, linewidth=1.5)
        
        # 배터리에서 나가는 전류 (비용)
        plt.plot(t_dq, i_batt_interp, 'k-', label='Battery Current (Energy Cost)', linewidth=2)
        
        plt.ylabel('Current (A)')
        plt.xlabel('Time (sec)')
        plt.title('Inverter Current Multiplication (Phase vs Battery)')
        plt.legend()
        plt.grid(True)
        
        # 3. 뻥튀기 비율 텍스트 표시 (최대 부하 구간)
        max_idx = np.argmax(i_phase_calc)
        max_phase = i_phase_calc[max_idx]
        max_batt = i_batt_interp[max_idx]
        
        if max_batt > 1: # 0으로 나누기 방지
            ratio = max_phase / max_batt
            plt.text(t_dq[max_idx], max_phase, f' Max Ratio: {ratio:.1f}x', 
                     verticalalignment='bottom', color='purple', fontweight='bold')

        plt.show()

    def plot_advanced_id_iq_analysis(self):
        """
        [고급 분석] Id/Iq 데이터를 통한 기어비, 토크맵, 주행전략 분석
        1. Gear Ratio Check: RPM vs Id (약계자 진입 시점 확인)
        2. Saturation Check: Iq vs Torque (자기 포화 확인)
        3. Operating Point: Id vs Iq Circle (주행 전략 확인)
        """
        # 데이터 존재 확인
        if self.data.Idq_set is None or self.data.vel_set is None or self.data.torqueAct_set is None:
            print("데이터 부족: Idq, Vel, TorqueAct가 모두 필요합니다.")
            return

        # --- 1. 데이터 추출 및 동기화 ---
        # 기준 시간축: 속도(Vel) 데이터 사용
        t_ref = self.data.vel_set[0, :]
        rpm   = self.data.vel_set[1, :]

        # Id, Iq 동기화
        t_idq = self.data.Idq_set[0, :]
        id_raw = self.data.Idq_set[1, :]
        iq_raw = self.data.Idq_set[2, :]
        
        id_sync = np.interp(t_ref, t_idq, id_raw)
        iq_sync = np.interp(t_ref, t_idq, iq_raw)

        # Torque 동기화
        t_trq = self.data.torqueAct_set[0, :]
        trq_raw = self.data.torqueAct_set[1, :]
        trq_sync = np.interp(t_ref, t_trq, trq_raw)

        # 노이즈 제거 (정차 중 데이터 제외)
        mask = (np.abs(rpm) > 100) 
        rpm_m = rpm[mask]
        id_m = id_sync[mask]
        iq_m = iq_sync[mask]
        trq_m = trq_sync[mask]

        # --- 2. 그래프 그리기 (1x3 Layout) ---
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))

        # [Graph 1] 기어비 분석: RPM vs Id (Flux Current)
        # 목적: 약계자(Id < 0)가 너무 낮은 RPM에서 시작되는지 확인
        axs[0].scatter(rpm_m, id_m, c='blue', s=3, alpha=0.3)
        axs[0].axhline(0, color='k', linestyle='--', linewidth=1)
        axs[0].set_xlabel('Motor Speed (RPM)')
        axs[0].set_ylabel('Flux Current Id (A)')
        axs[0].set_title('1. Gear Ratio Check\n(RPM vs Id)')
        axs[0].grid(True)
        # 가이드: Id가 음수로 떨어지는 RPM 지점이 "기저속도(Base Speed)"

        # [Graph 2] 토크맵 효율: Iq vs Torque
        # 목적: 전류를 퍼부어도 토크가 안 느는 "포화 구간" 확인
        axs[1].scatter(iq_m, trq_m, c=np.abs(id_m), cmap='coolwarm', s=3, alpha=0.5)
        axs[1].set_xlabel('Torque Current Iq (A)')
        axs[1].set_ylabel('Actual Torque (Nm)')
        axs[1].set_title('2. Saturation Check\n(Iq vs Torque)')
        axs[1].grid(True)
        # 색상(Color)은 Id 전류량 (빨갈수록 약계자 심함)
        
        # 기준선 (Ideal Kt Line, ME1616 approx 0.23)
        x_ref = np.linspace(0, np.max(iq_m), 100)
        axs[1].plot(x_ref, 0.23 * x_ref, 'g--', label='Linear Ref (Kt=0.23)')
        axs[1].legend()

        # [Graph 3] 주행 전략: Id vs Iq (Current Vector Trajectory)
        # 목적: 운전자가 전류원을 어떻게 쓰고 있는지 분포 확인
        sc = axs[2].scatter(id_m, iq_m, c=rpm_m, cmap='viridis', s=3, alpha=0.5)
        axs[2].set_xlabel('Flux Current Id (A)')
        axs[2].set_ylabel('Torque Current Iq (A)')
        axs[2].set_title('3. Driving Strategy\n(Current Vector Trajectory)')
        axs[2].grid(True)
        axs[2].axis('equal') # 원형 유지를 위해 비율 고정
        
        # 전류 제한원(Current Limit Circle) 가이드 (예: 400A)
        theta = np.linspace(0, np.pi, 100)
        axs[2].plot(400*np.cos(theta), 400*np.sin(theta), 'r--', label='400A Limit')
        
        cbar = plt.colorbar(sc, ax=axs[2])
        cbar.set_label('Speed (RPM)')

        plt.tight_layout()
        plt.show()

    def plot_vehicle_dynamics(self):
        """[수정됨] 가속도 센서를 활용한 차량 거동 분석 (센서 누워있음)"""
        if self.data.acc_set is None:
            print("가속도 센서 데이터가 없습니다.")
            return

        # 데이터 추출 (Raw Data)
        t_acc = self.data.acc_set[0, :]
        raw_x = self.data.acc_set[1, :] 
        raw_y = self.data.acc_set[2, :] 
        raw_z = self.data.acc_set[3, :] 

        # --- [축 매핑 수정] ---
        # 사용자 설정: Z=앞뒤, X=위아래
        long_g = raw_z  # 전후 (가속/감속) -> Z축
        lat_g  = raw_y  # 좌우 (코너링)    -> Y축 (기존 유지)
        vert_g = raw_x  # 상하 (진동/점프) -> X축

        fig = plt.figure(figsize=(14, 6))

        # [Graph 1] G-G Diagram (Longitudinal vs Lateral)
        # 이제 Z축(전후)과 Y축(좌우)을 그려야 합니다.
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(lat_g, long_g, 'k.', markersize=2, alpha=0.2)
        
        # 가이드라인 (0.5G, 1.0G)
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(0.5*np.cos(theta), 0.5*np.sin(theta), 'r--', label='0.5G')
        ax1.plot(1.0*np.cos(theta), 1.0*np.sin(theta), 'b--', label='1.0G')
        
        ax1.axhline(0, color='gray', linewidth=0.5)
        ax1.axvline(0, color='gray', linewidth=0.5)
        ax1.set_xlabel('Lateral G (Y-axis: Cornering)')
        ax1.set_ylabel('Longitudinal G (Z-axis: Accel/Brake)')
        ax1.set_title('1. G-G Diagram (Z vs Y)')
        ax1.axis('equal') 
        ax1.legend()
        ax1.grid(True)

        # [Graph 2] 시계열 진동 분석 (Z축 & X축)
        ax2 = fig.add_subplot(1, 2, 2)
        
        # 전후(Z) 가속도: 파란색
        ax2.plot(t_acc, long_g, 'b-', label='Longitudinal (Z: Go/Stop)', alpha=0.6, linewidth=1)
        
        # 상하(X) 가속도: 빨간색 (이제 X축이 충격입니다!)
        ax2.plot(t_acc, vert_g, 'r-', label='Vertical (X: Bump/Jump)', alpha=0.4, linewidth=0.5)
        
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Acceleration (g)')
        ax2.set_title('2. Vibration & Impact Analysis')
        ax2.legend(loc='upper right')
        ax2.grid(True)
        ax2.set_ylim(-2, 2) # 노이즈 제거용 범위 제한

        plt.tight_layout()
        plt.show()

    def plot_vehicle_dynamics_lpf(self):
        """
        [Bug Fix] NaN 데이터 제거 후 필터링 적용
        """
        if self.data.acc_set is None:
            return

        # 1. 데이터 추출
        t_acc = self.data.acc_set[0, :]
        raw_x = self.data.acc_set[1, :] # 수직
        raw_y = self.data.acc_set[2, :] # 좌우
        raw_z = self.data.acc_set[3, :] # 전후

        # --- [핵심 수정] NaN(빈 값) 제거 ---
        # 하나라도 NaN이 있으면 그 행은 쓰지 않습니다.
        mask = ~np.isnan(raw_x) & ~np.isnan(raw_y) & ~np.isnan(raw_z)
        
        t_acc = t_acc[mask]
        raw_x = raw_x[mask]
        raw_y = raw_y[mask]
        raw_z = raw_z[mask]

        if len(t_acc) == 0:
            print("유효한 가속도 데이터가 없습니다.")
            return
        # ----------------------------------

        # 2. 샘플링 주파수 자동 계산
        if len(t_acc) > 1:
            fs = 1.0 / np.mean(np.diff(t_acc))
        else:
            fs = 100.0

        # 3. 자동 영점 보정 (Calibration)
        # 정차 구간(초반 1초) 평균을 0으로 잡음
        N_cal = min(100, len(raw_x))
        
        offset_x = np.mean(raw_x[:N_cal]) 
        offset_y = np.mean(raw_y[:N_cal])
        offset_z = np.mean(raw_z[:N_cal])

        calib_x = raw_x - offset_x
        calib_y = raw_y - offset_y
        calib_z = raw_z - offset_z

        # 4. 필터링 (LPF)
        def apply_lpf(data, cutoff=1.0, order=2): #필터링 강도. 1.0Hz이상 컷오프.
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = butter(order, normal_cutoff, btype='low', analog=False)
            return filtfilt(b, a, data)

        filt_x = apply_lpf(calib_x)
        filt_y = apply_lpf(calib_y)
        filt_z = apply_lpf(calib_z)

        # 5. 축 매핑 (Z=전후, Y=좌우)
        final_long = filt_z
        final_lat  = filt_y
        final_vert = filt_x

        # --- 그래프 그리기 ---
        fig = plt.figure(figsize=(14, 7))

        # [Graph 1] G-G Diagram
        ax1 = fig.add_subplot(1, 2, 1)
        # 배경: 보정된 Raw (회색)
        ax1.plot(calib_y, calib_z, 'k.', markersize=1, alpha=0.05, label='Raw Noise')
        # 전경: 필터링 된 데이터 (빨간색) -> 이제 보일 겁니다!
        ax1.scatter(final_lat, final_long, s=5, c='red', alpha=0.2, label='Vehicle Motion')        
        
        # 중심 표시
        ax1.plot(0, 0, 'g+', markersize=15, linewidth=2)

        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(0.5*np.cos(theta), 0.5*np.sin(theta), 'b--', label='0.5G')
        
        ax1.set_xlabel('Lateral G')
        ax1.set_ylabel('Longitudinal G')
        ax1.set_title('1. G-G Diagram (Calibrated & Filtered)')
        ax1.axis('equal')
        ax1.legend()
        ax1.grid(True)

        # [Graph 2] Time Series
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(t_acc, final_long, 'b-', linewidth=1.5, label='Longitudinal (Go/Stop)')
        ax2.plot(t_acc, final_vert, 'r-', linewidth=1, alpha=0.6, label='Vertical (Suspension)')
        
        ax2.set_title('2. Motion Analysis (Filtered)')
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Acceleration (g)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(-1.5, 1.5)

        plt.tight_layout()
        plt.show()
    
    def plot_vehicle_dynamics_mv_avg(self):
        """
        [최종 수정 v3] 3초 이동 평균(Moving Average) 적용
        """
        if self.data.acc_set is None:
            return

        # 1. 데이터 추출 & NaN 제거
        t_acc = self.data.acc_set[0, :]
        raw_x = self.data.acc_set[1, :] 
        raw_y = self.data.acc_set[2, :] 
        raw_z = self.data.acc_set[3, :] 

        mask = ~np.isnan(raw_x) & ~np.isnan(raw_y) & ~np.isnan(raw_z)
        t_acc = t_acc[mask]
        raw_x = raw_x[mask]
        raw_y = raw_y[mask]
        raw_z = raw_z[mask]

        if len(t_acc) == 0:
            print("유효한 가속도 데이터가 없습니다.")
            return

        # 2. 샘플링 주파수(fs) 및 윈도우 사이즈 계산
        # fs: 1초에 데이터가 몇 개 찍히는가?
        if len(t_acc) > 1:
            fs = 1.0 / np.mean(np.diff(t_acc))
        else:
            fs = 100.0 # 기본값

        # [핵심] 3초 동안의 데이터 개수 계산
        target_sec = 3.0
        window_size = int(fs * target_sec)
        
        # 윈도우가 너무 작으면 최소 1로 설정
        if window_size < 1: window_size = 1
        
        print(f"INFO: 3초 이동평균 적용 (Window Size: {window_size} samples)")

        # 3. 자동 영점 보정 (Calibration)
        # 정차 구간(초반 1초) 평균을 0으로 잡음
        N_cal = min(int(fs), len(raw_x)) # 1초 분량
        
        calib_x = raw_x - np.mean(raw_x[:N_cal])
        calib_y = raw_y - np.mean(raw_y[:N_cal])
        calib_z = raw_z - np.mean(raw_z[:N_cal])

        # 4. 이동 평균 필터 함수 (Moving Average)
        def apply_moving_average(data, w):
            # np.convolve를 사용하여 'same' 모드로 길이를 유지합니다.
            return np.convolve(data, np.ones(w)/w, mode='same')

        # 필터 적용 (3초 평균)
        avg_x = apply_moving_average(calib_x, window_size)
        avg_y = apply_moving_average(calib_y, window_size)
        avg_z = apply_moving_average(calib_z, window_size)

        # 5. 축 매핑 (Z=전후, Y=좌우, X=상하)
        final_long = avg_z
        final_lat  = avg_y
        final_vert = avg_x

        # --- 그래프 그리기 ---
        fig = plt.figure(figsize=(14, 7))

        # [Graph 1] G-G Diagram
        ax1 = fig.add_subplot(1, 2, 1)
        
        # 배경: 원본 노이즈 (아주 흐리게)
        ax1.plot(calib_y, calib_z, 'k.', markersize=1, alpha=0.02, label='Raw Noise')
        
        # 전경: 3초 평균 데이터 (빨간색 실선)
        # 평균을 냈으므로 점보다는 선(Path)으로 보는 게 더 직관적입니다.
        ax1.plot(final_lat, final_long, 'r-', linewidth=2, alpha=0.9, label='3s Moving Avg')
        
        # 시작점 표시
        ax1.plot(final_lat[0], final_long[0], 'go', label='Start')

        # 가이드라인
        ax1.plot(0, 0, 'g+', markersize=15, linewidth=2)
        theta = np.linspace(0, 2*np.pi, 100)
        ax1.plot(0.5*np.cos(theta), 0.5*np.sin(theta), 'b--', label='0.5G')
        ax1.plot(1.0*np.cos(theta), 1.0*np.sin(theta), 'k--', alpha=0.5, label='1.0G')
        
        ax1.set_xlabel('Lateral G')
        ax1.set_ylabel('Longitudinal G')
        ax1.set_title(f'1. G-G Diagram ({target_sec}s Avg)')
        ax1.axis('equal')
        ax1.legend(loc='upper right')
        ax1.grid(True)
        ax1.set_xlim(-1.0, 1.0) # 범위는 데이터에 맞춰 조절하세요
        ax1.set_ylim(-1.0, 1.0)

        # [Graph 2] Time Series
        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(t_acc, final_long, 'b-', linewidth=2, label='Longitudinal (Z)')
        ax2.plot(t_acc, final_vert, 'r-', linewidth=2, alpha=0.7, label='Vertical (X)')
        
        ax2.set_title(f'2. Motion Analysis ({target_sec}s Avg)')
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Acceleration (g)')
        ax2.legend()
        ax2.grid(True)
        ax2.set_ylim(-0.5, 0.5) # 평균을 내면 값이 작아지므로 범위를 좁혀서 봅니다.

        plt.tight_layout()
        plt.show()

    def plot_gps_gforce_map(self):
        """
        [Final v3] GPS G-Force Heatmap (단순 거리 기반 필터링 적용)
        """
        if self.data.gpsPos_set is None or self.data.acc_set is None:
            return

        # 1. 데이터 추출
        t_gps = self.data.gpsPos_set[0, :]
        lon = self.data.gpsPos_set[1, :]
        lat = self.data.gpsPos_set[2, :]

        # --- [수정됨] 노이즈 제거: Median + Fixed Radius ---
        # 통계(IQR) 대신 물리적인 거리로 자릅니다.
        
        # (1) 0,0 제거
        mask_valid = (np.abs(lon) > 1.0) & (np.abs(lat) > 1.0)
        
        if np.sum(mask_valid) > 0:
            # (2) 트랙의 중심(Median) 찾기 - 튀는 값의 영향을 안 받음
            center_lon = np.median(lon[mask_valid])
            center_lat = np.median(lat[mask_valid])
            
            # (3) 반경 설정 (약 0.02도 ~= 2km)
            # 트랙이 아무리 커도 이 안에는 들어옵니다.
            radius = 0.02 
            
            mask_dist = (np.abs(lon - center_lon) < radius) & \
                        (np.abs(lat - center_lat) < radius)
            
            final_mask = mask_valid & mask_dist
        else:
            final_mask = mask_valid

        # 필터 적용
        t_gps = t_gps[final_mask]
        lon = lon[final_mask]
        lat = lat[final_mask]
        
        print(f"GPS Data Points: {len(lon)} (Filtered)") # 디버깅용 출력

        if len(t_gps) == 0:
            print("유효한 GPS 데이터가 없습니다.")
            return
        # ---------------------------------------------------

        # 2. 가속도 데이터 준비 & 필터링
        t_acc = self.data.acc_set[0, :]
        raw_y = self.data.acc_set[2, :] # Lateral
        raw_z = self.data.acc_set[3, :] # Longitudinal
        
        # (중요) 가속도 데이터도 NaN 제거 안 하면 색깔이 안 나옴
        acc_mask = ~np.isnan(raw_y) & ~np.isnan(raw_z)
        t_acc = t_acc[acc_mask]
        raw_y = raw_y[acc_mask]
        raw_z = raw_z[acc_mask]

        # 영점 보정
        if len(raw_y) > 100:
            calib_y = raw_y - np.mean(raw_y[:100])
            calib_z = raw_z - np.mean(raw_z[:100])
        else:
            calib_y, calib_z = raw_y, raw_z

        # LPF (1Hz) - 부드러운 색상 변화
        fs = 100.0
        if len(t_acc) > 1: fs = 1.0 / np.mean(np.diff(t_acc))
            
        def apply_lpf(data, cutoff=1.0):
            nyq = 0.5 * fs
            b, a = butter(2, cutoff/nyq, btype='low')
            return filtfilt(b, a, data)

        filt_lat = apply_lpf(calib_y)
        filt_long = apply_lpf(calib_z)

        # 3. 데이터 동기화
        # GPS 시간축에 맞춰 가속도 값을 가져옴
        lat_g_sync = np.interp(t_gps, t_acc, filt_lat)
        long_g_sync = np.interp(t_gps, t_acc, filt_long)

        # 4. Total G 계산
        total_g = np.sqrt(lat_g_sync**2 + long_g_sync**2)

        # 5. 그래프 그리기
        plt.figure(figsize=(12, 10))
        
        # s=20 (점 크기를 좀 키움), cmap='turbo' (색상 대비 명확하게)
        sc = plt.scatter(lon, lat, c=total_g, cmap='turbo', s=20, alpha=1.0,
                         vmin=0, vmax=0.6) # Baja니까 0.6G면 빨간색(Max)으로 설정

        cbar = plt.colorbar(sc)
        cbar.set_label('Total G-Force (g)')
        
        plt.title('Vehicle G-Force Heatmap (Fixed Radius Filter)')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.axis('equal')
        plt.grid(True)
        
        # 시작/끝 표시
        plt.plot(lon[0], lat[0], 'gx', markersize=12, markeredgewidth=3, label='Start')
        plt.plot(lon[-1], lat[-1], 'rx', markersize=12, markeredgewidth=3, label='End')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot_laps_slideshow(self):
        """
        [Interactive] 키보드 좌우 방향키로 랩을 넘겨보는 기능
        """
        # 1. 랩 분할
        laps = self.data.split_laps(start_radius_m=20, min_lap_time_sec=30)
        if not laps:
            print("감지된 랩이 없습니다.")
            return

        # 2. 전체 데이터 준비 (한 번만 계산)
        # (매번 계산하면 넘길 때 버벅거리므로 미리 다 해둡니다)
        t_gps = self.data.gpsPos_set[0, :]
        lon_raw = self.data.gpsPos_set[1, :]
        lat_raw = self.data.gpsPos_set[2, :]

        # NMEA 변환
        def nmea_to_decimal(nmea_val):
            deg = np.floor(nmea_val / 100.0)
            min = nmea_val - (deg * 100.0)
            return deg + (min / 60.0)

        if np.mean(lon_raw) > 180:
            lon_all = nmea_to_decimal(lon_raw)
            lat_all = nmea_to_decimal(lat_raw)
        else:
            lon_all, lat_all = lon_raw, lat_raw

        # 가속도 & Total G 준비
        t_acc = self.data.acc_set[0, :]
        ay = self.data.acc_set[2, :]
        az = self.data.acc_set[3, :]
        
        # NaN 제거
        mask = ~np.isnan(ay) & ~np.isnan(az)
        t_acc, ay, az = t_acc[mask], ay[mask], az[mask]

        # 필터링 (1Hz)
        from scipy.signal import butter, filtfilt
        fs = 100.0
        if len(t_acc) > 1: fs = 1.0 / np.mean(np.diff(t_acc))
        b, a = butter(2, 1.0/(0.5*fs), btype='low')
        ay_filt = filtfilt(b, a, ay - np.mean(ay[:100]))
        az_filt = filtfilt(b, a, az - np.mean(az[:100]))

        # 동기화
        ay_sync = np.interp(t_gps, t_acc, ay_filt)
        az_sync = np.interp(t_gps, t_acc, az_filt)
        total_g_all = np.sqrt(ay_sync**2 + az_sync**2)

        # 3. 인터랙티브 뷰어 설정
        fig, ax = plt.subplots(figsize=(12, 10))
        plt.subplots_adjust(bottom=0.2) # 아래쪽에 설명 문구 공간 확보

        current_lap_idx = [0] # 리스트로 감싸서 내부 함수에서 접근 가능하게 함

        def update_plot():
            ax.clear() # 이전 그림 지우기
            
            idx = current_lap_idx[0]
            lap_info = laps[idx]
            
            s = lap_info['idx_start']
            e = lap_info['idx_end']
            duration = lap_info['time_duration']

            # 데이터 슬라이싱
            lon_lap = lon_all[s:e]
            lat_lap = lat_all[s:e]
            g_lap = total_g_all[s:e]

            # 그리기
            sc = ax.scatter(lon_lap, lat_lap, c=g_lap, cmap='turbo', s=30, 
                            vmin=0, vmax=0.6) # 스케일 고정
            
            # 시작점
            ax.plot(lon_lap[0], lat_lap[0], 'gx', markersize=15, markeredgewidth=3, label='Start')

            # 꾸미기
            ax.set_title(f"Lap {idx + 1} / {len(laps)}  (Time: {duration:.2f}s)", fontsize=20, fontweight='bold')
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.axis('equal')
            ax.grid(True)
            ax.legend(loc='upper right')

            # 컬러바는 처음에 한 번만 그리거나, 없으면 추가
            if len(fig.axes) > 1: 
                # 이미 컬러바가 있으면 업데이트 안 함 (복잡도 줄임)
                pass 
            else:
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label('Total G-Force (g)')

            fig.canvas.draw_idle()

        # 4. 키보드 이벤트 핸들러
        def on_key(event):
            if event.key == 'right':
                current_lap_idx[0] = (current_lap_idx[0] + 1) % len(laps)
                update_plot()
            elif event.key == 'left':
                current_lap_idx[0] = (current_lap_idx[0] - 1) % len(laps)
                update_plot()

        # 이벤트 연결
        fig.canvas.mpl_connect('key_press_event', on_key)

        # 초기 실행
        update_plot()
        
        # 안내 문구
        plt.figtext(0.5, 0.05, "Use [Left] / [Right] Arrow Keys to Switch Laps", 
                    ha="center", fontsize=14, color='blue')
        plt.show()
