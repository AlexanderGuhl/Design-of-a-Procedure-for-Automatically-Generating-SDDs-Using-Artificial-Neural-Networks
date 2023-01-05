import json
from odio_urdf import *
import pandas as pd
import xmltodict
from anytree import Node, RenderTree
from anytree.exporter import DotExporter


def map_model_outputs(path: str):
    """
    function that uses the csv file, used for training the model,to assign numerical labels (0 to n) to semantic labels
    Assignment is saved in a json with dict structure
    only creates correct assignments for Datamodule of this project

    Params:
        path: path to csv file used for training the model
    """
    df = pd.read_csv(path)
    output_dict = {}

    # map neuron activation to semantic label
    for i in range(len(df.columns)):
        output_dict[i] = df.columns[i]

    with open("model_neuron_semantic_label_assignment.json", "w") as json_file:
        json.dump(output_dict, json_file, indent=4)


def create_json(sdd_dict: dict):
    """
    function to create a json file from dict

    Params:
        sdd_dict: dictionary containing ssd of robot
    """
    sdd = sdd_dict
    with open("SGB.json", "w") as json_file:
        json.dump(sdd, json_file, indent=4)


def create_xml(save_path_xml: str,mapping_socket_name: dict, robot_base_data: dict = None):
    """
    library taken from: https://github.com/hauptmech/odio_urdf

    create tree diagram of ros by switching to correct conda env and folder --> urdf_to_graphviz SGB.xml
    Problem: sockets don't get included in plot

    function that creates a xml file in URDF style, which can be parsed by ROS
    modified xml includes sockets for signal output of sensors
    """
    None_in_tcp = False
    # define single parts of robot
    if robot_base_data == None:
        my_robot = Robot("UR10", robot_sockets(1, 2, 3, 4))
    else:
        my_robot = Robot("UR10", robot_sockets(1, 2, 3, 4), Vendor_Name(robot_base_data["Vendor_name"]),Model_Name(robot_base_data["Model_name"]),
                     Ident_Number(robot_base_data["ident_number"]))
    link1 = Link("link 1")
    link2 = Link("link 2")
    link3 = Link("link 3")
    link4 = Link("link 4")
    link5 = Link("link 5")
    link6 = Link("link 6")
    # add parts to robot
    my_robot(link1, link2, link3, link4, link5, link6)

    # add sockets in odio_urdf init under sockets | add more categories in odio_urdf in joint or link class
    joint1 = Joint(Parent("link 1"), Child("link 2"), Origin((5, 3, 1), (0, 0, 0)), j_sockets(mapping_socket_name["actual_q_0"],
                        mapping_socket_name["actual_qd_0"], mapping_socket_name["target_qdd_0"], mapping_socket_name["actual_joint_voltage_0"],
                        mapping_socket_name["actual_current_0"], "Temp Joint 1"), name="Joint1", type="continuous")

    joint2 = Joint(Parent("link 2"), Child("link 3"), Origin((5, 3, 1), (0, 0, 0)), j_sockets(mapping_socket_name["actual_q_1"],
                        mapping_socket_name["actual_qd_1"], mapping_socket_name["target_qdd_0"], mapping_socket_name["actual_joint_voltage_1"],
                        mapping_socket_name["actual_current_0"], "Temp Joint 2"), name="Joint2", type="continuous")

    joint3 = Joint(Parent("link 3"), Child("link 4"), Origin((5, 3, 1), (0, 0, 0)), j_sockets(mapping_socket_name["actual_q_2"],
                        mapping_socket_name["actual_qd_2"], mapping_socket_name["target_qdd_0"], mapping_socket_name["actual_joint_voltage_2"],
                        mapping_socket_name["actual_current_0"], "Temp Joint 3"), name="Joint3", type="continuous")

    joint4 = Joint(Parent("link 4"), Child("link 5"), Origin((5, 3, 1), (0, 0, 0)), j_sockets(mapping_socket_name["actual_q_3"],
                        mapping_socket_name["actual_qd_3"], mapping_socket_name["target_qdd_0"], mapping_socket_name["actual_joint_voltage_3"],
                        mapping_socket_name["actual_current_0"], "Temp Joint 4"), name="Joint4", type="continuous")

    joint5 = Joint(Parent("link 5"), Child("link 6"), Origin((5, 3, 1), (0, 0, 0)), j_sockets(mapping_socket_name["actual_q_4"],
                        mapping_socket_name["actual_qd_4"], mapping_socket_name["target_qdd_0"], mapping_socket_name["actual_joint_voltage_4"],
                        mapping_socket_name["actual_current_0"], "Temp Joint 5"), name="Joint5", type="continuous")
    try:
        joint6 = Joint(Parent("link 6"), Child("None"), Origin((5, 3, 1), (0, 0, 0)), j_sockets(mapping_socket_name["actual_q_5"],
                        mapping_socket_name["actual_qd_5"], mapping_socket_name["target_qdd_0"], mapping_socket_name["actual_joint_voltage_5"],
                        mapping_socket_name["actual_current_0"], "Temp Joint 6"),
                       tcp_pose_sockets(mapping_socket_name["actual_TCP_pose_0"],mapping_socket_name["actual_TCP_pose_1"],mapping_socket_name["actual_TCP_pose_2"],
                        mapping_socket_name["actual_TCP_pose_3"],mapping_socket_name["actual_TCP_pose_4"],mapping_socket_name["actual_TCP_pose_5"]),
                       tcp_speed_sockets(mapping_socket_name["actual_TCP_speed_0"],mapping_socket_name["actual_TCP_speed_1"],mapping_socket_name["actual_TCP_speed_2"],
                       mapping_socket_name["actual_TCP_speed_3"],mapping_socket_name["actual_TCP_speed_4"],mapping_socket_name["actual_TCP_speed_5"]),
                       tcp_accel_sockets(mapping_socket_name["actual_tool_accelerometer_0"], mapping_socket_name["actual_tool_accelerometer_1"], mapping_socket_name["actual_tool_accelerometer_2"]),
                       tcp_force_sockets(mapping_socket_name["actual_TCP_force_0"],mapping_socket_name["actual_TCP_force_1"],mapping_socket_name["actual_TCP_force_2"],
                                         mapping_socket_name["actual_TCP_force_3"],mapping_socket_name["actual_TCP_force_4"],mapping_socket_name["actual_TCP_force_5"]),
                       name="Joint6", type="continuous")
    except:
        None_in_tcp = True

    # add parts to robot
    if None_in_tcp:
        my_robot(joint1, joint2, joint3, joint4, joint5)
    else:
        my_robot(joint1, joint2, joint3, joint4, joint5, joint6)

    # save xml of robot
    with open(save_path_xml + "\SGB.xml", "w") as f:
        f.write(str(my_robot))
        f.close()

    if robot_base_data == None :
        if None_in_tcp == True:
            print("SGB as XML was created without base data and saved in " + save_path_xml + "\SGB.xml\n None Type in TCP")
        else:
            print("SGB as XML was created without base data and saved in " + save_path_xml + "\SGB.xml\n\n")
    else:
        if None_in_tcp == True:
            print("SGB as XML was created with base data and saved in " + save_path_xml + "\SGB.xml\n None Type in TCP")
        else:
            print("SGB as XML was created with base data and saved in " + save_path_xml + "\SGB.xml\n\n")

def dict_create(curr_key, curr_value, prev_dict: dict = None):
    """
    function to simulate recursion
    creates a dictionary with either with or without a previous dict to generate hierarchical nested dictionaries
    created dictionary has recursion depth 1 or 0
    Params:
        curr_key: current key for to be created dictionary
        curr_value: current values for to be created dictionary
        prev_dict: previously created dictionary

    Returns:
        new_dict: from params created dictionary
    """
    new_dict = {}
    if prev_dict is None:
        new_dict[curr_key] = curr_value
    else:
        new_dict[curr_key] = curr_value, prev_dict
    return new_dict


def xml_to_json(path: str):
    """
    function that creates a json file with hierarchical structure from xml file in urdf style
    works with recursion using recur()

    Params:
        path: Path to XML file of SGB
    """
    robot_dict = {}
    # open xml file and parse to string
    with open(path, "r") as file:
        xml_f = file.read()
    sgb_dict = xmltodict.parse(xml_f)
    # get list of dicts for links of robot
    links = sgb_dict["robot"]["link"]
    robot_name = sgb_dict["robot"]["@name"]
    # create list of dictionaries of joints
    joint_dict_list = []
    for i in sgb_dict["robot"]["joint"]:
        if i["@name"] != "Joint6":
            joint_dict = {i["@name"]: (i["@type"], i["origin"], i["j_sockets"])}
            joint_dict_list.append(joint_dict)
        else:
            joint_dict = {i["@name"]: (i["@type"], i["origin"], i["j_sockets"], i["tcp_pose_sockets"], i["tcp_speed_sockets"],
                                       i["tcp_accel_sockets"], i["tcp_force_sockets"])}
            joint_dict_list.append(joint_dict)

    # discount recursion to create hierarchical nested dictionary
    z = len(joint_dict_list)-1
    while True:
        if z == len(joint_dict_list)-1:
            dicto = dict_create(links[z]["@name"], joint_dict_list[z])
            z = z-1
        elif z >= 0:
            dicto = dict_create(links[z]["@name"], joint_dict_list[z], dicto)
            z = z-1
        else:
            robot_dict["robot"] = [robot_name, sgb_dict["robot"]["robot_sockets"]], dicto
            #robot_dict["robot"] = robot_name, dicto
            break
    # create json file from dict
    create_json(robot_dict)

    print("SGB as Json was created and saved in SGB.json\n\n")


def xml_to_plot(path: str):
    """
    function that uses the information in the XML file generated by create_xml() to plot a tree diagram with anytree

    Params:
        path: Path to XML file of SGB
    """
    # open XML file and parse to string
    with open(path, "r") as f:
        xml_sgb = f.read()
    sgb_dict = xmltodict.parse(xml_sgb)

    # get name of robot
    robot_name = sgb_dict["robot"]["@name"]
    # get list of robot links
    links = sgb_dict["robot"]["link"]

    # create list containing dicts of joints with their data
    joint_dict_list = []
    for i in sgb_dict["robot"]["joint"]:
        joint_dict = {i["@name"]: (i["@type"], i["origin"], i["j_sockets"])}
        joint_dict_list.append(joint_dict)

    # create a list of joint names, since key can´t be used as string directly and anytree is sketchy
    joint_name_list = []
    for j in joint_dict_list:
        joint_name_list = joint_name_list + list(j.keys())

    # create anytree start object
    robot = Node(robot_name + "\n" "Current=1")
    robot_parts = [robot]
    for i in range(0, 6):
            robot_parts.append(Node(name=links[i]["@name"], parent=robot_parts[-1]))
            robot_parts.append(Node(joint_name_list[i] + "\n"
              "Force=" + joint_dict_list[i][joint_name_list[i]][2]["@force"] + "\n"
              "Current=" + joint_dict_list[i][joint_name_list[i]][2]["@current"] + "\n"
              "Speed=" + joint_dict_list[i][joint_name_list[i]][2]["@speed"] + "\n"
              "Voltage=" + joint_dict_list[i][joint_name_list[i]][2]["@voltage"] + "\n"
              "Angle=" + joint_dict_list[i][joint_name_list[i]][2]["@angle"] + "\n", parent=robot_parts[-1]))

    # create plot from anytree objects
    DotExporter(robot_parts[0]).to_picture("UR10.png")
    print("SGB was plotted and saved in robot.png\n\n")

    # print tree diagram in python console
    # for pre, fill, node in RenderTree(robot):
    # print("%s%s" % (pre, node.name))


if __name__ == "__main__":
    robot_base_data = {"Vendor_name": "Universal Robot", "Model_name": "UR10 - eSeries", "ident_number": "0x02c6"}
    create_xml("I:\BA\Code\SGB", robot_base_data)
    #xml_to_plot(r"I:\BA\Code\SGB\SGB.xml")
    #xml_to_json(r"I:\BA\Code\SGB\SGB.xml")
    #map_model_outputs(r"I:/BA/Code/datasets/ur10_MTL_var_1.csv")