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
        self.source_set     = None
        self.CAN_set        = None
        self.key_set        = None

        self.torqueAct_set  = None
        self.currAct_set    = None
        self.vel_set        = None
        
        self.ud_set         = None
        self.uq_set         = None
        self.Vmod_set       = None
        self.Vcap_set       = None
            
        self.L_set          = None
        self.Vlim_set       = None
        self.Iflux_set      = None
        self.Iqmax_set      = None
            
        self.motorTemp_set  = None
        self.Ibatt_set      = None
        self.Tdmd_set       = None

        self.Vtgt_set       = None
        self.Idq_set        = None
        self.acc_set        = None
        self.gpsPos_set     = None
        self.gpsVec_set     = None
    def allocate_or_extend(self, n_points, is_first_file):
        """
        n_points: 이번 파일에서 읽을 데이터 개수 (cnt_tot_sub)
        is_first_file: 첫 번째 파일인지 여부 (0이 첫 파일) (True/False)
        """
        new_source      = np.full((1, n_points), np.nan)
        new_CAN         = np.full((8, n_points), np.nan)
        new_key         = np.full((1, n_points), np.nan)

        new_torqueAct   = np.full((2, n_points), np.nan)
        new_currAct     = np.full((2, n_points), np.nan)
        new_vel         = np.full((2, n_points), np.nan)
        
        new_ud          = np.full((2, n_points), np.nan)
        new_uq          = np.full((2, n_points), np.nan)
        new_Vmod        = np.full((2, n_points), np.nan)
        new_Vcap        = np.full((2, n_points), np.nan)
        
        new_L           = np.full((2, n_points), np.nan)
        new_Vlim        = np.full((2, n_points), np.nan)
        new_Iflux       = np.full((2, n_points), np.nan)
        new_Iqmax       = np.full((2, n_points), np.nan)
        
        new_motorTemp   = np.full((2, n_points), np.nan)
        new_Ibatt       = np.full((2, n_points), np.nan)
        new_Tdmd        = np.full((2, n_points), np.nan)
        
        new_Vtgt        = np.full((2, n_points), np.nan)
        new_Idq         = np.full((3, n_points), np.nan)
        new_acc         = np.full((4, n_points), np.nan)
        new_gpsPos      = np.full((3, n_points), np.nan)
        new_gpsVec      = np.full((3, n_points), np.nan)

        if (is_first_file == True):
            self.source_set     = new_source   
            self.CAN_set        = new_CAN
            self.key_set        = new_key
            self.torqueAct_set  = new_torqueAct
            self.currAct_set    = new_currAct
            self.vel_set        = new_vel

            self.ud_set         = new_ud
            self.uq_set         = new_uq
            self.Vmod_set       = new_Vmod
            self.Vcap_set       = new_Vcap

            self.L_set          = new_L
            self.Vlim_set       = new_Vlim
            self.Iflux_set      = new_Iflux
            self.Iqmax_set      = new_Iqmax

            self.motorTemp_set  = new_motorTemp
            self.Ibatt_set      = new_Ibatt
            self.Tdmd_set       = new_Tdmd

            self.Vtgt_set       = new_Vtgt
            self.Idq_set        = new_Idq
            self.acc_set        = new_acc
            self.gpsPos_set     = new_gpsPos
            self.gpsVec_set     = new_gpsVec
        else:
            self.source_set     = np.concatenate((self.source_set, new_source),axis=1)
            self.CAN_set        = np.concatenate((self.CAN_set, new_CAN),axis=1)
            self.key_set        = np.concatenate((self.key_set, new_key),axis=1)
            self.torqueAct_set  = np.concatenate((self.torqueAct_set, new_torqueAct),axis=1)
            self.currAct_set    = np.concatenate((self.currAct_set, new_currAct),axis=1)
            self.vel_set        = np.concatenate((self.vel_set, new_vel),axis=1)

            self.ud_set         = np.concatenate((self.ud_set, new_ud),axis=1)
            self.uq_set         = np.concatenate((self.uq_set, new_uq),axis=1)
            self.Vmod_set       = np.concatenate((self.Vmod_set, new_Vmod),axis=1)
            self.Vcap_set       = np.concatenate((self.Vcap_set, new_Vcap),axis=1)

            self.L_set          = np.concatenate((self.L_set, new_L),axis=1)
            self.Vlim_set       = np.concatenate((self.Vlim_set, new_Vlim),axis=1)
            self.Iflux_set      = np.concatenate((self.Iflux_set, new_Iflux),axis=1)
            self.Iqmax_set      = np.concatenate((self.Iqmax_set, new_Iqmax),axis=1)

            self.motorTemp_set  = np.concatenate((self.motorTemp_set, new_motorTemp),axis=1)
            self.Ibatt_set      = np.concatenate((self.Ibatt_set, new_Ibatt),axis=1)
            self.Tdmd_set       = np.concatenate((self.Tdmd_set, new_Tdmd),axis=1)

            self.Vtgt_set       = np.concatenate((self.Vtgt_set, new_Vtgt),axis=1)
            self.Idq_set        = np.concatenate((self.Idq_set, new_Idq),axis=1)
            self.acc_set        = np.concatenate((self.acc_set, new_acc),axis=1)
            self.gpsPos_set     = np.concatenate((self.gpsPos_set, new_gpsPos),axis=1)
            self.gpsVec_set     = np.concatenate((self.gpsVec_set, new_gpsVec),axis=1)


# file select

def setfilename(fnametmp):
    '''set file path'''
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_dir = os.path.join(base_dir, 'log')
    return os.path.join(log_dir, fnametmp)


#---------------test---------------

if __name__ == "__main__":
    testsett=VehicleLog()


    def exfunc():
        """this function to show docstring
        this is poop"""
    print(exfunc.__doc__)
