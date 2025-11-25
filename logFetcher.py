# log fetcher

# import module

import numpy as np
"""This module does mathmatical calculation"""
import os
"""Used to read file"""


# setting const

rawAcc2g = 4/512
mls2sec = 1e-3
rawSpd2kph = 1.852 / 100


# setting var

class VehicleLog:
    """Class for data"""
    def __init__(self):
        """변수 초기화"""
        self.source_set = None
        self.CAN_set = None
        self.key_set = None
        self.torqueAct_set = None
        self.currAct_set = None
        self.vel_set = None
        
        self.ud_set = None
        self.uq_set = None
        self.Vmod_set = None
        self.Vcap_set = None
            
        self.L_set = None
        self.Vlim_set = None
        self.Iflux_set = None
        self.Iqmax_set = None
            
        self.motorTemp_set = None
        self.Ibatt_set = None
        self.Tdmd_set = None
            
        self.Vtgt_set = None
        self.Idq_set = None
        self.acc_set = None
        self.gpsPos_set = None
        self.gpsVec_set = None
    def allocate_or_extend(self, n_points, is_first_file):
        """
        n_points: 이번 파일에서 읽을 데이터 개수 (cnt_tot_sub)
        is_first_file: 첫 번째 파일인지 여부 (0이 첫 파일) (True/False)
        """



# file select

def setfilename(fnametmp):
    '''set file path'''
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'log')
    return os.path.join(log_dir, fnametmp)


# uncomment file you want to analize

# 2차 사전테스트
# fname = setfilename('2025-08-29 08-56-47.log'); flg_cnti = 0; # Laps: 0/0/0, Remarks: 가속/제동
# fname = setfilename('2025-08-29 09-01-23.log'); flg_cnti = 0; # Laps: 0/0/0, Remarks: 동적 성능 1
# fname = setfilename('2025-08-29 09-04-01.log'); flg_cnti = 1; # Laps: 0/0/0, Remarks: 동적 성능 1(이어짐)
# fname = setfilename('2025-08-29 09-05-08.log'); flg_cnti = 1; # Laps: 0/0/0, Remarks: 동적 성능 1(이어짐)
fname = setfilename('log/2025-08-29 09-09-27.log'); flg_cnti = 0; # Laps: 0/0/0, Remarks: 동적 성능 2
# fname = setfilename('2025-08-29 09-11-42.log'); flg_cnti = 1; # Laps: 0/0/0, Remarks: 동적 성능 2(이어짐)
# fname = setfilename('.log') % Laps: 0/0/0, Remarks:


# code
# count line:
print("\nINFO: entering size search loop... ")
cnt_tot_sub = 0 #count total 몇 줄인지
try:
    with open(fname, 'rb') as fid:
        while True:
            read_tmp1 = fid.read(8) #8 bytes 읽기
            
            if not read_tmp1: #비어있는 경우 false. 따라서 read_tmp1(현재 읽고 있는 줄)이 비어있다면 중단한다는 의미
                break
            
            read_tmp2 = fid.read(8) 

            cnt_tot_sub += 1

    print("INFO: size search loop finished. ")

except FileNotFoundError:
    print(f"Error: File {fname} not found.")

# set offset :

if( flg_cnti != 1 ):
    timestamp_offset = 0

    source_set = np.full((1, cnt_tot_sub), np.nan)
    cnt_source = 1
    
    CAN_set = np.full((8, cnt_tot_sub), np.nan)
    cnt_CAN = 1
    
    key_set = np.full((1, cnt_tot_sub), np.nan)
    cnt_key = 1
    
    
    # 물리량별 저장 공간 선언
    torqueAct_set   = np.full((2, cnt_tot_sub), np.nan); cnt_torqueAct = 1
    currAct_set     = np.full((2, cnt_tot_sub), np.nan); cnt_currAct = 1
    vel_set         = np.full((2, cnt_tot_sub), np.nan); cnt_vel = 1
    
    ud_set          = np.full((2, cnt_tot_sub), np.nan); cnt_ud = 1
    uq_set          = np.full((2, cnt_tot_sub), np.nan); cnt_uq = 1
    Vmod_set        = np.full((2, cnt_tot_sub), np.nan); cnt_Vmod = 1
    Vcap_set        = np.full((2, cnt_tot_sub), np.nan); cnt_Vcap = 1
    
    L_set           = np.full((2, cnt_tot_sub), np.nan); cnt_L = 1
    Vlim_set        = np.full((2, cnt_tot_sub), np.nan); cnt_Vlim = 1
    Iflux_set       = np.full((2, cnt_tot_sub), np.nan); cnt_Iflux = 1
    Iqmax_set       = np.full((2, cnt_tot_sub), np.nan); cnt_Iqmax = 1
    
    motorTemp_set   = np.full((2, cnt_tot_sub), np.nan); cnt_motorTemp = 1
    Ibatt_set       = np.full((2, cnt_tot_sub), np.nan); cnt_Ibatt = 1
    Tdmd_set        = np.full((2, cnt_tot_sub), np.nan); cnt_Tdmd = 1
    
    Vtgt_set        = np.full((2, cnt_tot_sub), np.nan); cnt_Vtgt = 1
    Idq_set        = np.full((3, cnt_tot_sub), np.nan); cnt_Idq = 1
    
    
    acc_set = np.full((4, cnt_tot_sub), np.nan)
    cnt_acc = 1
    
    gpsPos_set = np.full((3, cnt_tot_sub), np.nan)
    cnt_gpsPos = 1
    
    gpsVec_set = np.full((3, cnt_tot_sub), np.nan)
    cnt_gpsVec = 1

else :
    # timestamp_offset = timestamp_mem

    source_set = [source_set, np.full((1, cnt_tot_sub), np.nan)]
    CAN_set    = [CAN_set, np.full((8, cnt_tot_sub), np.nan)]
    key_set    = [key_set, np.full((1, cnt_tot_sub), np.nan)]


    # 물리량별 저장 공간 확장
    torqueAct_set   = [torqueAct_set, np.full((2, cnt_tot_sub), np.nan)]
    currAct_set     = [currAct_set, np.full((2, cnt_tot_sub), np.nan)]
    vel_set         = [vel_set, np.full((2, cnt_tot_sub), np.nan)]

    ud_set          = [ud_set, np.full((2, cnt_tot_sub), np.nan)]
    uq_set          = [uq_set, np.full((2, cnt_tot_sub), np.nan)]
    Vmod_set        = [Vmod_set, np.full((2, cnt_tot_sub), np.nan)]
    Vcap_set        = [Vcap_set, np.full((2, cnt_tot_sub), np.nan)]

    L_set           = [L_set, np.full((2, cnt_tot_sub), np.nan)]
    Vlim_set        = [Vlim_set, np.full((2, cnt_tot_sub), np.nan)]
    Iflux_set       = [Iflux_set, np.full((2, cnt_tot_sub), np.nan)]
    Iqmax_set       = [Iqmax_set, np.full((2, cnt_tot_sub), np.nan)]

    motorTemp_set   = [motorTemp_set, np.full((2, cnt_tot_sub), np.nan)]
    Ibatt_set       = [Ibatt_set, np.full((2, cnt_tot_sub), np.nan)]
    Tdmd_set        = [Tdmd_set, np.full((2, cnt_tot_sub), np.nan)]

    Vtgt_set        = [Vtgt_set, np.full((2, cnt_tot_sub), np.nan)]
    Idq_set         = [Idq_set, np.full((3, cnt_tot_sub), np.nan)]


    acc_set         = [acc_set, np.full((4, cnt_tot_sub), np.nan)]
    gpsPos_set      = [gpsPos_set, np.full((3, cnt_tot_sub), np.nan)]
    gpsVec_set      = [gpsVec_set, np.full((3, cnt_tot_sub), np.nan)]

#---------------test---------------

if __name__ == "__main__":
    print("\ntestcode :")
    print(f"cnt_tot_sub:{cnt_tot_sub}")


def djkajfkdsjflsj():
    """this function to show docstring
    this is poop"""

print(djkajfkdsjflsj.__doc__)