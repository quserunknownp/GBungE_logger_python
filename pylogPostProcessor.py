import matplotlib.pyplot as plt
import numpy as np

class LogVisualizer:
    def __init__(self, log_data):
        self.data = log_data

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
