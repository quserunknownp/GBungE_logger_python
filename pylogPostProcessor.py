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