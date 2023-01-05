# importierend er benötigten Bibliotheken
import pandas as pd
import rtde_control
import rtde_receive
import rtde_io
import script_client
import time
import matplotlib.pyplot as plt
import sklearn.preprocessing as prepro

def check_connection(IP:str):
    # check if connection of the interfaces was succesfull
    if rtde_c.isConnected():  # Entspricht if (rtde_s.isConnected() == True)
        # rtde_c.osConnected() liefert liefert eine Bool-variable zurück: True wenn verbunden, False wenn nicht verbunden
        print("rtde_c ist verbunden:  192.168.1.12 \n")
    else:
        print("Verbinden rtde_c fehlgeschlagen")

    if rtde_r.isConnected():
        print("rtde_r ist verbunden:  192.168.1.12 \n")
    else:
        print("Verbinden rtde_r fehlgeschlagen")

    if rtde_s.isConnected():
        print("rtde_s ist verbunden:  192.168.1.12 \n")
    else:
        print("Verbinden rtde_s fehlgeschlagen \n")

    # Prüfen ob der Roboter durch Sicherheitsmaßnahmen gestoppt wurde:

    if rtde_r.isEmergencyStopped():
        print("wurde per Not-Aus gestoppt.\n")
    else:
        print("Not-Aus deaktiviert.\n")

    if rtde_r.isProtectiveStopped():
        print("Sicherheitsstopp aktiviert\n")
    else:
        print("Sicherheitsstopp deaktiviert\n")


def disconnect_robot():
    # disconnect all connections
    rtde_c.disconnect()
    rtde_r.disconnect()
    rtde_s.disconnect()
    print("Verbindungen getrennt.\n")


def check_and_move_home(home:list):
    # check if robot is not moving
    actual_q = rtde_r.getActualQ()
    actual_tcp = rtde_r.getActualTCPPose()
    if rtde_c.isSteady():
        print("Roboter ist in Ruhe.\n")

        # Prüft ob die momentane Position mit der Home-Position übereinstimmt, wenn ja geht das restliche Programm weiter,
        # wenn nein wird zuerst die Home-Position angefahren
        if (actual_tcp[0] < home[0]-0.001 or actual_tcp[1] < home[1]-0.01 or actual_tcp[2] < home[2]-0.01) or \
                (actual_tcp[0] < home[0]+0.001 or actual_tcp[1] < home[1]+0.01 or actual_tcp[2] < home[2]+0.01):
            print("Nicht in Home-Position:\n")
            rtde_c.moveL(home, 0.2, 0.2)
            time.sleep(3)
            actual_tcp_h = rtde_r.getActualTCPPose()

        else:
            print("In Home-Position\n")
    else:
        print("Roboter ist nicht in Ruhe.\n")

def move_J(pos1, tool_speed, tool_acceleration, collected_data:list):
    rtde_c.moveJ(pos1,tool_speed,tool_acceleration, True)
    # ensure robot finishes movement
    while rtde_c.isSteady() == False:   #maybe here is a fault? maybe change to while actual_pos != target_pos:
        collect_data(collected_data)
        time.sleep(0.01)
    print("movement finished\n")

def move_L(pos1, tool_speed, tool_acceleration, collected_data:list):
    rtde_c.moveL(pos1,tool_speed,tool_acceleration, True)
    # ensure robot finishes movement
    while rtde_c.isSteady() == False:
        collect_data(collected_data)
        time.sleep(0.01)
    print("movement finished\n")

def move_C(pos1, tool_speed, tool_acceleration, collected_data:list):
    rtde_c.moveC(pos1,tool_speed,tool_acceleration, True)
    # ensure robot finishes movement
    while rtde_c.isSteady() == False:
        collect_data(collected_data)
        time.sleep(0.01)
    print("movement finished\n")

def collect_data(collected_data:list):
    act_q = rtde_r.getActualQ() # returns 6 doubles for position in rad
    collected_data[0].append(act_q[0])
    collected_data[1].append(act_q[1])
    collected_data[2].append(act_q[2])
    collected_data[3].append(act_q[3])
    collected_data[4].append(act_q[4])
    collected_data[5].append(act_q[5])
    act_qd = rtde_r.getActualQd() # returns 6 doubles for velocity in rad/s
    collected_data[6].append(act_qd[0])
    collected_data[7].append(act_qd[1])
    collected_data[8].append(act_qd[2])
    collected_data[9].append(act_qd[3])
    collected_data[10].append(act_qd[4])
    collected_data[11].append(act_qd[5])
    act_curr = rtde_r.getActualCurrent() # returns actual current for joints
    collected_data[12].append(act_curr[0])
    collected_data[13].append(act_curr[1])
    collected_data[14].append(act_curr[2])
    collected_data[15].append(act_curr[3])
    collected_data[16].append(act_curr[4])
    collected_data[17].append(act_curr[5])
    act_curr_contr = rtde_r.getJointControlOutput() # returns current of joint controllers
    collected_data[18].append(act_curr_contr[0])
    collected_data[19].append(act_curr_contr[1])
    collected_data[20].append(act_curr_contr[2])
    collected_data[21].append(act_curr_contr[3])
    collected_data[22].append(act_curr_contr[4])
    collected_data[23].append(act_curr_contr[5])
    act_TCP_pose = rtde_r.getActualTCPPose() # returns actual tcp pose in x,y,z,rx,ry,rz in m or rad
    collected_data[24].append(act_TCP_pose[0])
    collected_data[25].append(act_TCP_pose[1])
    collected_data[26].append(act_TCP_pose[2])
    collected_data[27].append(act_TCP_pose[3])
    collected_data[28].append(act_TCP_pose[4])
    collected_data[29].append(act_TCP_pose[5])
    act_TCP_speed = rtde_r.getActualTCPSpeed() # returns actual tcp speed in x,y,z,rx,ry,rz in m/s or rad/s
    collected_data[30].append(act_TCP_speed[0])
    collected_data[31].append(act_TCP_speed[1])
    collected_data[32].append(act_TCP_speed[2])
    collected_data[33].append(act_TCP_speed[3])
    collected_data[34].append(act_TCP_speed[4])
    collected_data[35].append(act_TCP_speed[5])
    act_TCP_force = rtde_r.getActualTCPForce() # force in joints in Nm
    collected_data[36].append(act_TCP_force[0])
    collected_data[37].append(act_TCP_force[1])
    collected_data[38].append(act_TCP_force[2])
    collected_data[39].append(act_TCP_force[3])
    collected_data[40].append(act_TCP_force[4])
    collected_data[41].append(act_TCP_force[5])
    act_temp = rtde_r.getJointTemperatures() # returns joint temps
    collected_data[42].append(act_temp[0])
    collected_data[43].append(act_temp[1])
    collected_data[44].append(act_temp[2])
    collected_data[45].append(act_temp[3])
    collected_data[46].append(act_temp[4])
    collected_data[47].append(act_temp[5])
    act_acc = rtde_r.getActualToolAccelerometer() # returns x,y,z acceleration in rad/s^2
    collected_data[48].append(act_acc[0])
    collected_data[49].append(act_acc[1])
    collected_data[50].append(act_acc[2])
    act_momentum = rtde_r.getActualMomentum() # returns double of linearised force in Nm
    collected_data[51].append(act_momentum)
    act_rob_vol = rtde_r.getActualRobotVoltage() # returns double of voltage consumed by robot in V
    collected_data[52].append(act_rob_vol)
    act_rob_curr = rtde_r.getActualRobotCurrent()  # returns double of current consumed by robot in A
    collected_data[53].append(act_rob_curr)
    act_vol = rtde_r.getActualCurrent() # returns actual voltage for joints
    collected_data[54].append(act_vol[0])
    collected_data[55].append(act_vol[1])
    collected_data[56].append(act_vol[2])
    collected_data[57].append(act_vol[3])
    collected_data[58].append(act_vol[4])
    collected_data[59].append(act_vol[5])

    return collected_data

if __name__ == "__main__":
    IP = "139.321.3123.323"
    save_path = r""
    data_list = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],
                 [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    # Aufbauend der Verbindungen  zu den jeweiligen Ports
    rtde_r = rtde_receive.RTDEReceiveInterface(IP)
    rtde_c = rtde_control.RTDEControlInterface(IP)
    rtde_io = rtde_io.RTDEIOInterface(IP)
    rtde_s = script_client.ScriptClient(IP, 30002, False)
    check_connection(IP)
    # set standard params
    tool_speed = 0.2
    tool_acceleration = 0.2
    blend = 0
    # home should be start_pos
    home = [953.90, -163.9, 700.5, -90, -0.379, 0.379]

    robot_status = rtde_r.getRobotStatus()
    robot_mode = rtde_r.getRobotMode()
    safety_mode = rtde_r.getSafetyMode()

    print("Roboterstatus:", robot_status, "\n")
    print("Robotermodus:", robot_mode, "\n")
    print("Sicherheitsmodus:", safety_mode, "\n")

    check_and_move_home(home)
    # targets in x,y,z,rx,ry,rz | x,y,z in m
    target_1 = [953.90, -163.9, 700.5, -90, -0.379, 0.379]
    target_2 = [953.70, -163.9, 429.5, -90, -0.504, 0.504]
    target_3 = [537.90, -163.9, 421.9, -90, 2.246, -2.246]
    target_4 = [547.8, 127, 421.967, -80, 46.75, -46.75]
    target_5 = [907.92, 334.9, 429.59, -80.9, 46.731, -46.731]
    target_6 = [908.090, 335.012, 700.67, -80.916, 46.731, -46.731]

    #Pfad definieren
    move_J(target_2, tool_speed, tool_acceleration, data_list)
    move_L(target_3, tool_speed, tool_acceleration, data_list)
    move_J(target_4, tool_speed, tool_acceleration, data_list)
    move_L(target_5, tool_speed, tool_acceleration, data_list)
    move_J(target_6, tool_speed, tool_acceleration, data_list)
    move_J(target_1, tool_speed, tool_acceleration, data_list)

    # save data
    data_dict = {
                    "j1_degree": data_list[0], "j1_speed": data_list[6], "j1_current": data_list[12],
                    "j1_controll_current": data_list[18],"j1_temp": data_list[42], "j1_voltage": data_list[54],
                    'j2_degree': data_list[1], "j2_speed": data_list[7], "j2_current": data_list[13], "j2_controll_current": data_list[19],
                    "j2_temp": data_list[43], "j2_voltage": data_list[55], 'j3_degree': data_list[2], "j3_speed": data_list[8],
                    "j3_current": data_list[14], "j3_controll_current": data_list[20], "j3_temp": data_list[44], "j3_voltage": data_list[56],
                    "j4_degree": data_list[3], "j4_speed": data_list[9], "j4_current": data_list[15], "j4_controll_current": data_list[21],
                    "j4_temp": data_list[45],  "j4_voltage": data_list[57], "j5_degree": data_list[4], "j5_speed": data_list[10],
                    "j5_current": data_list[16], "j5_controll_current": data_list[22], "j5_temp": data_list[46], "j5_voltage": data_list[58],
                    "j6_degree": data_list[5], "j6_speed": data_list[11], "j6_current": data_list[17], "j6_controll_current": data_list[23],
                    "j6_temp": data_list[47], "j6_voltage": data_list[59], "tcp_x": data_list[24],"tcp_y": data_list[25],"tcp_z": data_list[26],
                    "tcp_rx": data_list[27], "tcp_ry": data_list[28],"tcp_rz": data_list[29], "tcp_speed_x": data_list[30],
                    "tcp_speed_y": data_list[31],"tcp_speed_z": data_list[32],"tcp_speed_rx": data_list[33], "tcp_speed_ry": data_list[34],
                    "tcp_speed_rz": data_list[35], "tcp_force_x": data_list[36], "tcp_force_y": data_list[37],"tcp_force_z": data_list[38],
                    "tcp_force_rx": data_list[39], "tcp_force_ry": data_list[40],"tcp_force_rz": data_list[41], "tool_accel_x": data_list[48],
                    "tool_accel_y": data_list[49], "tool_accel_z": data_list[50], "robot_momentum": data_list[51],
                    "robot_voltage": data_list[52],"robot_current": data_list[53]
                     }


    test = data_dict["j1_degree"]
    plt.plot(prepro.minmax_scale(test, feature_range=(-1, 1)))
    plt.show()

    df = pd.DataFrame(data_dict)
    df.to_csv(save_path)

    disconnect_robot()