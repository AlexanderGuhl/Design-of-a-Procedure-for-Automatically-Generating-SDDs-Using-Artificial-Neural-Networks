import torch
import pandas as pd
import matplotlib.pyplot as plt
import json
from Baseline import CompressNet
from matplotlib.pyplot import figure
import math


def plot_kuka(data:pd.DataFrame, variante):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(1, 7):
        new_data = data["Joint " + str(j)][:5250]*(math.pi/180)
        plt.plot(new_data)

    plt.title("Gelenkwinkel KR 22 R1610")
    plt.legend(["Gelenk 1", "Gelenk 2", "Gelenk 3", "Gelenk 4", "Gelenk 5", "Gelenk 6"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)
    plt.ylabel('Gelenkwinkel [rad]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots\Kuka/var_" + str(variante) + r"\Kuka_var_" + str(variante) + "_Gelenkwinkel.svg")
    plt.show()


def plot_angles(data:pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(0, 6):
        if robot == "UR10_MTL":
            new_data = data["actual_q_" + str(j)][:6600]
            plt.title("Gelenkwinkel UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_q_" + str(j)][:5125]
            plt.title("Gelenkwinkel UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_q_" + str(j)][:8800]
            plt.title("Gelenkwinkel UR10e (f)")
            plt.plot(new_data)
    plt.legend(["Gelenk 1", "Gelenk 2", "Gelenk 3", "Gelenk 4", "Gelenk 5", "Gelenk 6"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)
    plt.ylabel('Gelenkwinkel [rad]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot +"/var_" + str(variante) + r"/" + robot + "_var_" + str(variante) + "_Gelenkwinkel.svg")
    plt.show()


def plot_angle_velocities(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(0, 6):
        if robot == "UR10_MTL":
            new_data = data["actual_qd_" + str(j)][:6600]
            plt.title("Gelenkwinkelgeschwindigkeiten UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_qd_" + str(j)][:5125]
            plt.title("Gelenkwinkelgeschwindigkeiten UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_qd_" + str(j)][:8800]
            plt.title("Gelenkwinkelgeschwindigkeiten UR10e (f)")
            plt.plot(new_data)

    plt.legend(["Gelenk 1", "Gelenk 2", "Gelenk 3", "Gelenk 4", "Gelenk 5", "Gelenk 6"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)
    plt.ylabel('Gelenkwinkelgeschwindigkeit [rad/s]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_Gelenkgeschwindigkeit.svg")
    plt.show()


def plot_joint_temperatures(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(0, 6):
        if robot == "UR10_MTL":
            new_data = data["joint_temperatures_" + str(j)][:6600]
            plt.title("Gelenktemperaturen UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["joint_temperatures_" + str(j)][:5125]
            plt.title("Gelenktemperaturen UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["joint_temperatures_" + str(j)][:8800]
            plt.title("Gelenktemperaturen UR10e (f)")
            plt.plot(new_data)

    plt.legend(["Gelenk 1", "Gelenk 2", "Gelenk 3", "Gelenk 4", "Gelenk 5", "Gelenk 6"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)
    plt.ylabel('Gelenktemperaturen [°C]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_Gelenktemperaturen.svg")
    plt.show()


def plot_angle_accel(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(0, 6):
        if robot == "UR10_MTL":
            new_data = data["target_qdd_" + str(j)][:6600]
            plt.title("Gelenkwinkelbeschleunigungen UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["target_qdd_" + str(j)][:5125]
            plt.title("Gelenkwinkelbeschleunigungen UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["target_qdd_" + str(j)][:8800]
            plt.title("Gelenkwinkelbeschleunigungen UR10e (f)")
            plt.plot(new_data)

    plt.legend(["Gelenk 1", "Gelenk 2", "Gelenk 3", "Gelenk 4", "Gelenk 5", "Gelenk 6"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)
    plt.ylabel('Gelenkwinkelbeschleunigung [rad/s²]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_Gelenkbeschleunigung.svg")
    plt.show()


def plot_tcp_cartesian_pose(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(0, 3):
        if robot == "UR10_MTL":
            new_data = data["actual_TCP_pose_" + str(j)][:6600]
            plt.title("kartesische TCP Position UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_TCP_pose_" + str(j)][:5125]
            plt.title("kartesische TCP Position UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_TCP_pose_" + str(j)][:8800]
            plt.title("kartesische TCP Position UR10e (f)")
            plt.plot(new_data)

    plt.legend(["x", "y", "z"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)
    plt.ylabel('kartesische TCP Position [m]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_TCP_cartesian_pose.svg")
    plt.show()


def plot_tcp_rot_pose(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(3, 6):
        if robot == "UR10_MTL":
            new_data = data["actual_TCP_pose_" + str(j)][:6600]
            plt.title("rotatorische TCP Position UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_TCP_pose_" + str(j)][:5125]
            plt.title("rotatorische TCP Position UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_TCP_pose_" + str(j)][:8800]
            plt.title("rotatorische TCP Position UR10e (f)")
            plt.plot(new_data)

    plt.legend(["rx", "ry", "rz"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)
    plt.ylabel(' TCP Winkelposition [rad]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_TCP_rot_pose.svg")
    plt.show()


def plot_momemtum(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(0, 1):
        if robot == "UR10_MTL":
            new_data = data["actual_momentum"][:6600]
            plt.title("Betrag des Impulses UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_momentum"][:5125]
            plt.title("Betrag des Impulses UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_momentum"][:8800]
            plt.title("Betrag des Impulses UR10e (f)")
            plt.plot(new_data)

    plt.ylabel('Betrag des Impulses [Ns]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_Impuls.svg")
    plt.show()


def plot_robot_voltage(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    for j in range(0, 1):
        figure(figsize=(10, 5), dpi=100)
        if robot == "UR10_MTL":
            new_data = data["actual_robot_voltage"][:6600]
            plt.title("Am Industrieroboter anliegende Spannung UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_robot_voltage"][:5125]
            plt.title("Am Industrieroboter anliegende Spannung UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_robot_voltage"][:8800]
            plt.title("Am Industrieroboter anliegende Spannung UR10e (f)")
            plt.plot(new_data)

    plt.ylabel('Am Industrieroboter anliegende Spannung [V]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_Roboter_Spannung.svg")
    plt.show()


def plot_robot_current(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    for j in range(0, 1):
        figure(figsize=(10, 5), dpi=100)
        if robot == "UR10_MTL":
            new_data = data["actual_robot_current"][:6600]
            plt.title("Am Industrieroboter anliegender Strom UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_robot_current"][:5125]
            plt.title("Am Industrieroboter anliegender Strom UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_robot_current"][:8800]
            plt.title("Am Industrieroboter anliegender Strom UR10e (f)")
            plt.plot(new_data)

    plt.ylabel('Am Industrieroboter anliegender Strom [A]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_Roboter_Strom.svg")
    plt.show()


def plot_main_voltage(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    for j in range(0, 1):
        figure(figsize=(10, 5), dpi=100)
        if robot == "UR10_MTL":
            new_data = data["actual_main_voltage"][:6600]
            plt.title("An der Steuerung anliegene Spannung UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_main_voltage"][:5125]
            plt.title("An der Steuerung anliegene Spannung UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_main_voltage"][:8800]
            plt.title("An der Steuerung anliegene Spannung UR10e (f)")
            plt.plot(new_data)

    plt.ylabel('An der Steuerung anliegende Spannung [A]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_Hauptspannung.svg")
    plt.show()


def plot_tcp_accel(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(0, 3):
        if robot == "UR10_MTL":
            new_data = data["actual_tool_accelerometer_" + str(j)][:6600]
            plt.title("kartesische TCP Beschleunigung UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_tool_accelerometer_" + str(j)][:5125]
            plt.title("kartesische TCP Beschleunigung UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_tool_accelerometer_" + str(j)][:8800]
            plt.title("kartesische TCP Beschleunigung UR10e (f)")
            plt.plot(new_data)
    plt.legend(["x", "y", "z"],bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)

    plt.ylabel('kartesische Beschleunigung am TCP [m/s²]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_TCP_Beschleunigung.svg")
    plt.show()


def plot_tcp_trans_speed(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(0, 3):
        if robot == "UR10_MTL":
            new_data = data["actual_TCP_speed_" + str(j)][:6600]
            plt.title("translatorische TCP Geschwindigkeit UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_TCP_speed_" + str(j)][:5125]
            plt.title("translatorische TCP Geschwindigkeit UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_TCP_speed_" + str(j)][:8800]
            plt.title("translatorische TCP Geschwindigkeit UR10e (f)")
            plt.plot(new_data)
    plt.legend(["x", "y", "z"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)

    plt.ylabel('translatorische Geschwindigkeit am TCP [m/s]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_TCP_trans_speed.svg")
    plt.show()


def plot_tcp_angular_speed(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(3, 6):
        if robot == "UR10_MTL":
            new_data = data["actual_TCP_speed_" + str(j)][:6600]
            plt.title("TCP Winkelgeschwindigkeit UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_TCP_speed_" + str(j)][:5125]
            plt.title("TCP Winkelgeschwindigkeit UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_TCP_speed_" + str(j)][:8800]
            plt.title("TCP Winkelgeschwindigkeit UR10e (f)")
            plt.plot(new_data)
    plt.legend(["rx", "ry", "rz"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)

    plt.ylabel('Winkelgeschwindigkeit am TCP [rad/s]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_TCP_angular_speed.svg")
    plt.show()


def plot_tcp_force(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(0, 3):
        if robot == "UR10_MTL":
            new_data = data["actual_TCP_force_" + str(j)][:6600]
            plt.title("Im TCP wirkende Kräfte UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_TCP_force_" + str(j)][:5125]
            plt.title("Im TCP wirkende Kräfte UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_TCP_force_" + str(j)][:8800]
            plt.title("Im TCP wirkende Kräfte UR10e (f)")
            plt.plot(new_data)
    plt.legend(["Fx", "Fy", "Fz"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)

    plt.ylabel('Kräfte am TCP [N]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_TCP_Kraefte.svg")
    plt.show()


def plot_tcp_torque(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(6, 5), dpi=100)
    for j in range(3, 6):
        if robot == "UR10_MTL":
            new_data = data["actual_TCP_force_" + str(j)][:6600]
            plt.title("Im TCP wirkende Drehmomente UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_TCP_force_" + str(j)][:5125]
            plt.title("Im TCP wirkende Drehmomente UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_TCP_force_" + str(j)][:8800]
            plt.title("Im TCP wirkende Drehmomente UR10e (f)")
            plt.plot(new_data)
    plt.legend(["Mx", "My", "Mz"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)

    plt.ylabel('Drehmoment am TCP [Nm]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_TCP_Drehmoment.svg")
    plt.show()


def plot_joint_current(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    figure(figsize=(10, 7), dpi=100)
    plt.rc('legend', fontsize=14)
    plt.rcParams.update({'font.size': 18})
    for j in range(0, 6):
        if robot == "UR10_MTL":
            new_data = data["actual_current_" + str(j)][:6600]
            plt.title("Gelenkströme UR10e (w)")
            plt.plot(new_data)
        if robot == "UR3":
            new_data = data["actual_current_" + str(j)][:5125]
            plt.title("Gelenkströme UR3")
            plt.plot(new_data)
        if robot == "UR10":
            new_data = data["actual_current_" + str(j)][:8800]
            plt.title("Gelenkströme UR10e (f)")
            plt.plot(new_data)
    plt.legend(["Gelenk 1", "Gelenk 2", "Gelenk 3", "Gelenk 4", "Gelenk 5", "Gelenk 6"],
               bbox_to_anchor=(0.95, 0.7), fancybox=True, shadow=True)

    plt.ylabel('Gelenk Strom [A]')
    plt.xlabel('Messpunkte')
    plt.tight_layout()
    plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
        variante) + "_Gelenkstrom.svg")
    plt.show()


def plot_joint_voltage(data: pd.DataFrame, variante, robot):
    plt.style.use(style="ggplot")
    if robot == "UR10_MTL":
        for j in range(0,6):
            figure(figsize=(10, 5), dpi=100)
            new_data = data["actual_joint_voltage_" + str(j)][:367]
            plt.plot(new_data)
            plt.ylabel('Gelenkspannung [V]')
            plt.xlabel('Messpunkte')
            title = "Gelenkspannung Gelenk " + str(j) + " UR10 (w)"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
                variante) + "_Gelenk_" + str(j) + "_Spannung.svg")
            plt.show()
            break


    if robot == "UR3":
        for k in range(0, 6):
            figure(figsize=(10, 5), dpi=100)
            new_data = data["actual_joint_voltage_" + str(k)][:320]
            plt.plot(new_data)
            plt.ylabel('Gelenkspannung [V]')
            plt.xlabel('Messpunkte')
            title = "Gelenkspannung Gelenk " + str(k) + " UR3"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
                variante) + "_Gelenk_" + str(k) + "_Spannung.svg")
            plt.show()
            break


    if robot == "UR10":
        for l in range(0,6):
            figure(figsize=(10, 5), dpi=100)
            new_data = data["actual_joint_voltage_" + str(l)][:550]
            plt.plot(new_data)
            plt.ylabel('Gelenkspannung [V]')
            plt.xlabel('Messpunkte')
            title = "Gelenkspannung Gelenk " + str(l) + " UR10 (f)"
            plt.title(title)
            plt.tight_layout()
            plt.savefig(r"I:\BA\Plots/" + robot + "/var_" + str(variante) + r"/" + robot + "_var_" + str(
                variante) + "_Gelenk_" + str(l) + "_Spannung.svg")
            plt.show()
            break


def plot_exp_results(folder_path:str):
    metric = []
    exp_num = []
    for i in range(72):
        with open(folder_path + "/" + str(i) + '/results.json') as json_file:
            results = json.load(json_file)
        metric.append(results["results"]["acc_micro_test"])
        exp_num.append(results["setup"]["EXPERIMENT_ID"])

    fig, ax = plt.subplots()
    bars = ax.barh(exp_num, metric)
    plt.title("Accuracy Micro multi_2")
    ax.bar_label(bars)
    plt.show()


def plot_all(data, version, robot):
    plot_angles(data, version, robot)
    plot_angle_velocities(data, version, robot)
    plot_angle_accel(data, version, robot)
    plot_joint_temperatures(data, version, robot)
    plot_tcp_cartesian_pose(data, version, robot)
    plot_tcp_rot_pose(data, version, robot)
    plot_momemtum(data, version, robot)
    plot_robot_voltage(data, version, robot)
    plot_robot_current(data, version, robot)
    plot_main_voltage(data, version, robot)
    plot_tcp_accel(data, version, robot)
    plot_tcp_trans_speed(data, version, robot)
    plot_tcp_angular_speed(data, version, robot)
    plot_tcp_force(data, version, robot)
    plot_tcp_torque(data, version, robot)
    plot_joint_current(data, version, robot)
    plot_joint_voltage(data, version, robot)


def compress_data(path):
    data = pd.read_csv(path)
    # length of UR3 dataset
    model = CompressNet(1, 50000)
    for i in data.columns:
        data2 = data[i].astype(float)
        y = model.forward(torch.tensor(data2))
        y_np = y.numpy()
        y_df = pd.DataFrame(y_np)
        data[i] = y_df
    data.dropna().to_csv(r"I:\BA\Code\datasets\ur10_MTL_var_1_comp_test.csv", index=False)


if __name__ == "__main__":
    file_list_kuka = [r"I:\BA\Plots\Kuka\var_1\kuka_var_1.csv",r"I:\BA\Plots\Kuka\var_2\kuka_var_2.csv",r"I:\BA\Plots\Kuka\var_3\kuka_var_3.csv",
                      r"I:\BA\Plots\Kuka\var_4\kuka_var_4.csv",r"I:\BA\Plots\Kuka\var_5\kuka_var_5.csv",r"I:\BA\Plots\Kuka\var_6\kuka_var_6.csv"]

    file_list_ur3 = [r"I:\BA\Plots\UR3\var_1\ur3_var_1.csv", r"I:\BA\Plots\UR3\var_2\ur3_var_2.csv",
                      r"I:\BA\Plots\UR3\var_3\ur3_var_3.csv",
                      r"I:\BA\Plots\UR3\var_4\ur3_var_4.csv", r"I:\BA\Plots\UR3\var_5\ur3_var_5.csv",
                      r"I:\BA\Plots\UR3\var_6\ur3_var_6.csv"]

    file_list_ur10_MTL = [r"I:\BA\Plots\UR10_MTL\var_corr.csv"]
    file_list_ur10 = [r"I:\BA\Plots\UR10\var_1\ur10_var_1.csv",r"I:\BA\Plots\UR10\var_2\ur10_var_2.csv",r"I:\BA\Plots\UR10\var_3\ur10_var_3.csv",
                      r"I:\BA\Plots\UR10\var_4\ur10_var_4.csv",r"I:\BA\Plots\UR10\var_5\ur10_var_5.csv",r"I:\BA\Plots\UR10\var_6\ur10_var_6.csv"]

    a = 1
    for i in file_list_ur10_MTL:
        data = pd.read_csv(i)
        #plot_all(data, a, "UR10_MTL")
        plot_joint_current(data, a, "UR10_MTL")
        a = a + 1



    #compress_data(r"I:\BA\Code\datasets\ur10_MTL_var_1.csv")

