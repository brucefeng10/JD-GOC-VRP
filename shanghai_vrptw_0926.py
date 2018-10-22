# -*- coding: utf-8 -*-
"""
@Author   : brucefeng10
@Contact  : fengxfcn@163.com
"""


import time
import csv
import numpy as np
import math
import random
import copy
import matplotlib.pyplot as plt



# parameter value setting
run_time = time.strftime("%m%d_%H%M", time.localtime())
veh_spd_kmh = 50  # supposed vehicle speed in km/h
veh_spd = veh_spd_kmh * 1000 / 60.  # supposed vehicle speed in m/min(50km/h, 833m/min)
vehicle_type_dict = {2: '2T', 3: '3T', 5: '5T'}
small_veh = [2, 2, 10, 540, 0.004, 200]  # [veh_type, load weight, effective volume, trvl_dist, trs_cost, fix_cost]
medium_veh = [3, 3, 11, 540, 0.005, 200]  # [veh_type, load weight, effective volume, trvl_dist, trs_cost, fix_cost]
large_veh = [5, 5, 17, 540, 0.006, 300]  # [veh_type, load weight, effective volume, trvl_time, trs_cost, fix_cost]
bskt_vol = 0.65 * 0.44 * 0.16  # volume of fresh basket
trsf_vol = 0.56 * 0.39 * 0.32  # volume of transfer box
milk_vol = 0.001  # volume of both fresh milk and skim milk is 1 litre
paper_bskt = 1.2 * 1 * 0.6  # volume of paper basket
oprt_t = 25  # operation time at each store
start_t = 0  # earliest start time of a vehicle
small_veh_cnt = 60  # setup(initial) small vehicle number
medium_veh_cnt = 39  # setup(initial) medium vehicle number
alp, bet, gam = 0.6, 0.2, 0.2  # weight of t_ij, h_ij and v_ij of Time-oriented Nearest-Neighbor



def read_data():
    pnt = 664
    num_id = {}  # store number: [id, name, address]
    id_num = {}  # id: number
    loc = np.zeros([pnt, 2])  # longitude and latitude
    demd = {}  # store demand: [bskt_num, trsf_num, fresh_milk, skim_milk]
    time_zone = {}  # store service time zone: [earliest start time, latest start time]

    with open(r'C:\Bee\0Huaat\Starbucks\inputdata\Shanghai_nodes.csv', 'rU') as fr:
        reader = csv.reader(fr)
        next(reader)
        itr = 0
        for v in reader:
            itr += 1
            if itr > pnt:
                break
            store_num = int(v[0])
            num_id[store_num] = v[1:4]
            id_num[v[1]] = store_num
            loc[store_num] = v[4:6]
            demd[store_num] = [int(v[8]), int(v[9]), int(v[10]), int(v[11]), int(v[12])]
            first_t = v[6].split(':')
            last_t = v[7].split(':')
            time_zone[store_num] = [int(first_t[0])*60 + int(first_t[1]), int(last_t[0])*60 + int(last_t[1])]

    return num_id, id_num, loc, demd, time_zone


def earth_dist(lng_lat):
    '''Given an array of points(longitude and latitude), this function will return a distance matrix'''
    r = 6371.004 * 1000  # radius of earth(6371km)
    ln = len(lng_lat)  # number of points
    dist_matr = np.zeros([ln, ln])
    time_matr = np.zeros([ln, ln])
    for i in range(ln):
        for j in range(i + 1, ln):
            ax = math.pi * lng_lat[i, 0] / 180.0  # transfer angle to radian
            ay = math.pi * lng_lat[i, 1] / 180.0
            bx = math.pi * lng_lat[j, 0] / 180.0
            by = math.pi * lng_lat[j, 1] / 180.0
            dist = r * math.acos(math.sin(ay) * math.sin(by) + math.cos(ay) * math.cos(by) * math.cos(bx - ax))
            dist_matr[i, j] = round(dist)
            dist_matr[j, i] = dist_matr[i, j]
            time_matr[i, j] = dist_matr[i, j] / veh_spd  # suppose the vehicle driving speed is 50km/h(833.3m/min)
            time_matr[j, i] = time_matr[i, j]

    return dist_matr, time_matr


class GetInitial(object):
    """To get an initial solution."""

    def __init__(self):
        self.a = 5

    def time_nn(self, on_way_time, curr_cust, remain_list, used_resource, rout_len, vehicle_type):
        """Given a vehicle and its current visiting customer, return the next visiting customer.
        Here we use the Time-oriented Nearest-Neighborhood Heuristic proposed by Solomon(1987).
        The closeness between customer i and j: C_ij = alp1*t_ij + alp2*h_ij + alp3*v_ij,
        t_ij: travel time between i and j (distance);
        h_ij = b_j - (b_i + s_i), b is start service time, and s is service time (idle time);
        v_ij = l_j - (b_i + s_i + t_ij), l is the latest admissible service time, t_ij is the travel time (urgency)
        The low value of C_ij, the better.
        """
        if vehicle_type == 2:
            veh_cap = small_veh
        elif vehicle_type == 3:
            veh_cap = medium_veh
        else:
            veh_cap = large_veh
        real_wait_time = 0  # the final wait time after testing all the possible stores
        real_vst_cust = -1  # the final visiting store after testing all the possible stores
        visit_cust = [-1, 100000, 600000, 10000]  # [cust_id, next_start, distance, closeness]
        if rout_len - 1 < 50:  # max number of stores a vehicle visits
            for cust in remain_list:
                # print('checking customer: ', cust)
                if (used_resource[0] + num_demd[cust][0] * bskt_vol + num_demd[cust][1] * trsf_vol + (num_demd[cust][2] +
                        num_demd[cust][3]) * milk_vol + num_demd[cust][4] * paper_bskt) > veh_cap[2]:
                    # print('run out of effective volume')
                    continue  # volume overload
                # elif dist_mat[curr_cust, cust] + dist_mat[cust, 0] > veh_cap[3] - used_resource[3]:
                #     print('run out of distance')
                #     continue
                elif used_resource[2] + time_mat[curr_cust, cust] > num_timez[cust][1]:
                    # print('late than last receive time')
                    continue  # can not arrive before last receive time
                elif time_mat[curr_cust, cust] + oprt_t + time_mat[cust, 0] > veh_cap[3] - on_way_time:
                    # print('run out of work time')
                    continue
                elif (curr_cust > 0 and used_resource[2] + time_mat[curr_cust, cust] < num_timez[cust][0] and
                        num_timez[cust][0] - used_resource[2] + oprt_t + time_mat[cust, 0] > veh_cap[3] - on_way_time):
                    # print('run out of work time - with waiting time')
                    continue
                else:
                    wait_time = num_timez[cust][0] - (used_resource[2] + time_mat[curr_cust, cust])

                    if wait_time < 0:
                        next_start = used_resource[2] + time_mat[curr_cust, cust]
                        h_ij = time_mat[curr_cust, cust]
                    else:  # arrive early
                        next_start = num_timez[cust][0]
                        if curr_cust == 0:
                            h_ij = time_mat[curr_cust, cust]
                            wait_time = 0   # special situation for depot depart
                        else:
                            h_ij = next_start - used_resource[2]
                    v_ij = num_timez[cust][1] - (used_resource[2] + time_mat[curr_cust, cust])
                    close_ij = alp * time_mat[curr_cust, cust] + bet * h_ij + gam * v_ij  # closeness between i and j
                    # print(curr_cust, cust, close_ij)
                    if close_ij < visit_cust[3]:
                        real_wait_time = wait_time
                        real_vst_cust = cust
                        visit_cust[0] = cust
                        visit_cust[1] = next_start
                        visit_cust[2] = dist_mat[curr_cust, cust]
                        visit_cust[3] = close_ij
                    else:
                        continue


        if visit_cust[0] == -1:  # no customer to visit
            visit_cust[0] = 0
            visit_cust[1] = used_resource[-1] + time_mat[curr_cust, 0]
            on_way_time += time_mat[curr_cust, 0]
        else:
            # print(curr_cust, real_vst_cust, real_wait_time)
            if real_wait_time <= 0:
                on_way_time += (oprt_t + time_mat[curr_cust, real_vst_cust])
            else:
                on_way_time += (oprt_t + real_wait_time + time_mat[curr_cust, real_vst_cust])

        return visit_cust, on_way_time


    def greedy_initial(self):
        """
        Generate an initial solution based on the Time-oriented Nearest-neighbor heuristic proposed by Solomon.
        """
        sol = []  # [[0;2;5;0;4;6;0],[],...]
        sol_veh_type = []  # corresponding vehicle type for the solution
        route_way_time = []

        to_vist = [i+1 for i in range(store_num - 1)]  # [1,5,8,...]
        itr = 0

        while len(to_vist) > 0 and itr < 500:
            itr += 1

            if itr <= small_veh_cnt:
                vehicle_type0 = 2
            elif itr <= small_veh_cnt + medium_veh_cnt:
                vehicle_type0 = 3
            else:
                vehicle_type0 = 5

            sol_veh_type.append(vehicle_type0)

            used_res = [0, 0, 0, 0]  # used volume, and travel time of the vehicle, leave time, travel distance
            veh_rout = [0]

            # print '\nA new vehicle will be used.'
            way_time = 0  # travel time of coming to the store + wait time at the store + operation time at this store
            while True:
                curr_cust = veh_rout[-1]

                next_one, way_time = self.time_nn(way_time, curr_cust, to_vist, used_res, len(veh_rout), vehicle_type0)
                next_cust, next_start = next_one[0], next_one[1]
                # print('next start', next_cust, next_start)
                if next_cust == 0:  # next visiting customer is depot
                    # print 'Get back to the depot, and ready for a new round.'
                    veh_rout.append(next_cust)
                    break

                else:  # next visiting customer is a store
                    used_res[0] += (num_demd[next_cust][0] * bskt_vol + num_demd[next_cust][1] * trsf_vol + (num_demd[next_cust][2] + \
                                    num_demd[next_cust][3]) * milk_vol + num_demd[next_cust][4] * paper_bskt)
                    used_res[2] = (next_start + oprt_t)
                    used_res[3] += dist_mat[curr_cust, next_cust]


                veh_rout.append(next_cust)
                # print 'Vehicle used resource: ', used_res
                to_vist.remove(next_cust)

            sol.append(veh_rout)
            route_way_time.append(way_time)

            # print 'Last point 0 earliest leave time: ', int(used_res[-1]) / 60, ':', int(used_res[-1]) % 60
            # print 'Route %s is: ' % itr, veh_rout
            print('*'*10, 'Iteration:', itr, '*'*10)


        if len(to_vist) > 0:
            print('number of stores remained: ', len(to_vist))

        return sol, sol_veh_type, route_way_time



class OutputFormat(object):
    """print result in particular format: route details, route summary, plan summary"""

    def __init__(self):
        self.c = 4


    def print_result(self, solution, vehicle_type, if_write):
        """Given the solution saved in list, calculate the total cost of the solution.
        Write the solution to local in the required format."""

        result = [['Vehicle_ID', 'Vehicle_type', 'Route', 'Leave_Time', 'Back_Time', 'Work_Time', 'Distance',
                   'Load_Volume', 'Wait_Time', 'Fixed_Cost', 'Travel_Cost', 'Total_Cost']]
        total_dist = 0
        total_cost = 0
        for k, veh in enumerate(solution):
            if len(veh) == 2:
                continue

            if vehicle_type[k] == 2:
                trans0 = small_veh[4]
                fix0 = small_veh[5]
            elif vehicle_type[k] == 3:
                trans0 = medium_veh[4]
                fix0 = medium_veh[5]
            else:
                trans0 = large_veh[4]
                fix0 = large_veh[5]

            total_cost += fix0
            departt = check_violation(veh, vehicle_type[k])[3]

            trvl_dist = 0
            veh_load_vol = 0
            wait_time = 0

            # get the output format
            route = [0] * len(result[0])
            route[0] = k + 1  # vehicle name
            route[1] = vehicle_type_dict[vehicle_type[k]]  # vehicle type
            route_ele = []
            for ele in veh:
                if ele == 0:
                    route_ele.append(str(ele))
                else:
                    route_ele.append(num_id[ele][0])
            route[2] = '-'.join(route_ele)  # route

            trvl_dist += (dist_mat[0, veh[1]] + dist_mat[veh[-2], 0])
            veh_load_vol += (num_demd[veh[1]][0] * bskt_vol + num_demd[veh[1]][1] * trsf_vol + (num_demd[veh[1]][2] +
                             num_demd[veh[1]][3]) * milk_vol + num_demd[veh[1]][4] * paper_bskt)
            if departt / 60. < 24.:
                out_time = int(departt)
            else:
                out_time = int(departt - 24 * 60)
            route[3] = str(out_time // 60) + ':' + str(out_time % 60).zfill(2)
            t = departt + time_mat[0, veh[1]] + oprt_t
            for i in range(2, len(veh) - 1):  # can not wait at the first 2 points
                trvl_dist += dist_mat[veh[i - 1], veh[i]]
                veh_load_vol += (num_demd[veh[i]][0] * bskt_vol + num_demd[veh[i]][1] * trsf_vol + (num_demd[veh[i]][2] +
                                 num_demd[veh[i]][3]) * milk_vol + num_demd[veh[i]][4] * paper_bskt)
                wait_t = num_timez[veh[i]][0] - (t + time_mat[veh[i - 1], veh[i]])
                if wait_t > 0 + 1e-5:
                    # print veh[i-1], veh[i], wait_t
                    wait_time += wait_t
                    t = num_timez[veh[i]][0] + oprt_t
                else:
                    t += (time_mat[veh[i - 1], veh[i]] + oprt_t)
            if t + time_mat[veh[-2], 0] < 24. * 60:
                in_time = int(t + time_mat[veh[-2], 0])
            else:
                in_time = int(t + time_mat[veh[-2], 0] - 24 * 60)

            route[4] = str(in_time // 60) + ':' + str(in_time % 60).zfill(2)  # vehicle back time
            route[5] = round((t + time_mat[veh[-2], 0] - departt) / 60., 1)
            route[6] = round(trvl_dist / 1000., 2)  # total distance
            route[7] = veh_load_vol  # vehicle load volume
            route[8] = wait_time  # vehicle wait time
            route[9] = fix0  # vehicle fixed cost
            route[10] = round(trvl_dist * trans0, 2)  # vehicle travel cost
            route[11] = route[9] + route[10]  # total cost

            total_cost += trvl_dist * trans0
            result.append(route)
            # print route
            total_dist += route[6]
            # print 'Last leave time: ', int(t) / 60, ':', int(t) % 60
            # print 'total distances: ', route[5]

        if if_write:
            run_time = time.strftime("%m%d_%H%M", time.localtime())
            with open(r'C:\Bee\0Huaat\Starbucks\results\Route_Plan_%s.csv' % run_time, 'w', newline='') as fw:
                writer = csv.writer(fw)
                for v in result:
                    writer.writerow(v)

        return round(total_cost, 2)


    def print_route_detail(self, solution, vehicle_type, if_write):
        """Given the solution saved in list, calculate the total cost of the solution.
        Write the solution to local in the required format."""

        result = [[
            '线路编号',
            '门店编码',
            '门店名称',
            '门店地址',
            '经度',
            '纬度',
            '车型',
            '额定体积/m3',
            '额定重量/t',
            '到达时间',
            '离开时间',
            '行驶距离/km',
            '累计行驶距离km',
            '行驶时间/min',
            '卸货时间/min',
            '累计工作时间/h',
            '鲜食篮总数',
            '周转箱个数',
            '新绿园鲜奶980ML（罐）',
            '新绿园脱脂牛奶980ML（罐）',
            '纸箱个数',
            '卸货体积',
            '卸货重量']]

        total_dist = 0
        for k, veh in enumerate(solution):
            if vehicle_type[k] == 2:
                trans0 = small_veh[4]
                veh_param = small_veh

            elif vehicle_type[k] == 3:
                trans0 = medium_veh[4]
                veh_param = medium_veh

            else:
                trans0 = large_veh[4]
                veh_param = large_veh


            departt = check_violation(veh, vehicle_type[k])[3]
            t = departt

            trvl_dist = 0
            veh_load_vol = 0
            wait_time = 0

            veh_load_vol += (num_demd[veh[1]][0] * bskt_vol + num_demd[veh[1]][1] * trsf_vol + (num_demd[veh[1]][2] +
                             num_demd[veh[1]][3]) * milk_vol + num_demd[veh[1]][4] * paper_bskt)
            if departt / 60. < 24.:
                out_time = int(math.ceil(departt))
            else:
                out_time = int(math.ceil(departt - 24 * 60))

            # get the output format
            store = [0] * len(result[0])
            store[0] = k + 1  # 线路序号
            store[1] = num_id[0][0]  # 门店编号
            store[2] = num_id[0][1]  # 门店名称
            store[3] = num_id[0][2]  # 门店地址
            store[4] = loc[0][0]  # 经度
            store[5] = loc[0][1]  # 纬度
            store[6] = vehicle_type_dict[vehicle_type[k]]  # 车型
            store[7] = veh_param[2]  # 额定体积
            store[8] = veh_param[1]  # 额定重量
            store[9] = str(out_time // 60) + ':' + str(out_time % 60).zfill(2)  # 到达时间
            store[10] = str(out_time // 60) + ':' + str(out_time % 60).zfill(2)  # 离开时间
            store[11] = 0  # 行驶距离
            store[12] = 0  # 累计行驶距离
            store[13] = 0  # 行驶时间
            store[14] = 0  # 卸货时间
            store[15] = 0  # 累计工作时间
            store[16] = 0  # 鲜食篮件数
            store[17] = 0  # 周转箱个数
            store[18] = 0  # 新绿园鲜奶
            store[19] = 0  # 新绿园脱脂牛奶
            store[20] = 0  # 纸箱
            store[21] = 0  # 卸货体积
            store[22] = 0  # 卸货重量

            store0 = copy.deepcopy(store)
            result.append(store0)

            # t = departt + time_mat[0, veh[1]] + oprt_t  # t is the leaving time
            for i in range(1, len(veh)-1):  # can not wait at the first 2 points
                store[1] = num_id[veh[i]][0]
                store[2] = num_id[veh[i]][1]
                store[3] = num_id[veh[i]][2]
                store[4] = loc[veh[i]][0]
                store[5] = loc[veh[i]][1]
                arr_time = t + time_mat[veh[i-1], veh[i]]
                if arr_time / 60. < 24.:
                    in_time = int(math.ceil(arr_time))
                else:
                    in_time = int(math.ceil(arr_time - 24 * 60))

                trvl_dist += dist_mat[veh[i-1], veh[i]]
                veh_load_vol += (num_demd[veh[i]][0] * bskt_vol + num_demd[veh[i]][1] * trsf_vol + (num_demd[veh[i]][2] +
                                 num_demd[veh[i]][3]) * milk_vol + num_demd[veh[i]][4] * paper_bskt)
                wait_t = num_timez[veh[i]][0] - (t + time_mat[veh[i-1], veh[i]])
                if wait_t > 0 + 1e-5:
                    # t is the leaving time
                    wait_time += wait_t
                    t = num_timez[veh[i]][0] + oprt_t
                else:
                    t += (time_mat[veh[i - 1], veh[i]] + oprt_t)
                if t < 24. * 60:
                    out_time = int(math.ceil(t))
                else:
                    out_time = int(math.ceil(t - 24 * 60))

                store[9] = str(in_time // 60) + ':' + str(in_time % 60).zfill(2)  # 到达时间
                store[10] = str(out_time // 60) + ':' + str(out_time % 60).zfill(2)  # 离开时间
                store[11] = round(dist_mat[veh[i-1], veh[i]] / 1000., 2)  # 行驶距离
                store[12] = round(trvl_dist / 1000., 2)  # 累计行驶距离
                store[13] = round(time_mat[veh[i-1], veh[i]], 1)  # 行驶时间
                store[14] = oprt_t
                store[15] = round((t - departt) / 60., 2)  # 累计工作时间
                store[16] = num_demd[veh[i]][0]  # 鲜食篮件数
                store[17] = num_demd[veh[i]][1]  # 周转箱个数
                store[18] = num_demd[veh[i]][2]  # 新绿园鲜奶
                store[19] = num_demd[veh[i]][3]  # 新绿园脱脂牛奶
                store[20] = num_demd[veh[i]][4]  # 纸箱
                store[21] = (num_demd[veh[i]][0] * bskt_vol + num_demd[veh[i]][1] * trsf_vol + (num_demd[veh[i]][2] +
                            num_demd[veh[i]][3]) * milk_vol + num_demd[veh[i]][4] * paper_bskt)  # 卸货体积
                store[22] = 0  # 卸货重量

                store0 = copy.deepcopy(store)
                result.append(store0)
                # print(result[-1])

            store[1] = num_id[0][0]  # 门店编号
            store[2] = num_id[0][1]  # 门店名称
            store[3] = num_id[0][2]  # 门店地址
            store[4] = loc[0][0]  # 经度
            store[5] = loc[0][1]  # 纬度
            arr_time = t + time_mat[veh[-2], 0]
            if arr_time / 60. < 24.:
                in_time = int(math.ceil(arr_time))
            else:
                in_time = int(math.ceil(arr_time - 24 * 60))
            store[9] = str(in_time // 60) + ':' + str(in_time % 60).zfill(2)  # 到达时间
            store[10] = str(in_time // 60) + ':' + str(in_time % 60).zfill(2)  # 离开时间
            store[11] = round(dist_mat[veh[-2], 0] / 1000., 2)  # 行驶距离
            store[12] = round((trvl_dist + dist_mat[veh[-2], 0]) / 1000., 2)  # 累计行驶距离
            store[13] = round(time_mat[veh[-2], 0], 1)  # 行驶时间
            store[14] = 0  # 卸货时间
            store[15] = round((t - departt + time_mat[veh[-2], 0]) / 60., 2)  # 累计工作时间
            store[16] = 0  # 鲜食篮件数
            store[17] = 0  # 周转箱个数
            store[18] = 0  # 新绿园鲜奶
            store[19] = 0  # 新绿园脱脂牛奶
            store[20] = 0  # 纸箱
            store[21] = 0  # 卸货体积
            store[22] = 0  # 卸货重量

            store0 = copy.deepcopy(store)
            result.append(store0)
            # print(result[-1])

        if if_write:
            # run_time = time.strftime("%m%d_%H%M", time.localtime())
            with open(r'C:\Bee\0Huaat\Starbucks\output\Route_Details_%s_%s.csv' % (veh_spd_kmh, run_time), 'w', newline='') as fw:
                writer = csv.writer(fw)
                for v in result:
                    # print(v)
                    writer.writerow(v)


    def print_route_summary(self, solution, vehicle_type, if_write):
        """Given the solution saved in list, calculate the total cost of the solution.
        Write the solution to local in the required format."""
        result_summary = [[
            '计划编号',
            '门店数',
            '配送总体积/m3',
            '配送总重量/t',
            '设定车速/km/h',
            '总车数',
            '总额定体积/m3',
            '总额定重量/t',
            '体积装载率/%',
            '重量装载率/%',
            '总行驶里程/km',
            '有效里程/km',
            '返空里程/km',
            '工作时间/h',
            '行驶时间/min',
            '卸货时间/min',
            '总成本/元',
            '固定成本/元',
            '运输成本/元',
            '2T车数量',
            '3T车数量',
            '5T车数量',
            '鲜食篮总数',
            '周转箱个数',
            '新绿园鲜奶980ML（罐）',
            '新绿园脱脂牛奶980ML（罐）',
            '纸箱个数']]
        summ_value = [0] * len(result_summary[0])

        result = [[
            '线路编号',
            '出发时间',
            '返回时间',
            '工作时间/h',
            '行驶总时间/min',
            '卸货总时间/min',
            '等待时间/min',
            '总行驶里程/km',
            '有效里程/km',
            '返空里程/km',
            '车型',
            '额定装载体积/m3',
            '额定装载重量/t',
            '实际装载体积/m3',
            '实际装载重量/t',
            '体积装载率/%',
            '重量装载率/%',
            '总成本/元',
            '固定成本/元',
            '运输成本/元',
            '配送门店总数',
            '门店1编号',
            '门店1名称',
            '门店2编号',
            '门店2名称',
            '门店3编号',
            '门店3名称',
            '门店4编号',
            '门店4名称',
            '门店5编号',
            '门店5名称',
            '门店6编号',
            '门店6名称',
            '门店7编号',
            '门店7名称',
            '门店8编号',
            '门店8名称',
            '门店9编号',
            '门店9名称',
            '门店10编号',
            '门店10名称',
            '门店11编号',
            '门店11名称',
            '门店12编号',
            '门店12名称',
            '门店13编号',
            '门店13名称',
            '门店14编号',
            '门店14名称',
            '门店15编号',
            '门店15名称',
            '门店16编号',
            '门店16名称',
            '门店17编号',
            '门店17名称',
            '门店18编号',
            '门店18名称',
            '门店19编号',
            '门店19名称',
            '门店20编号',
            '门店20名称']]

        total_dist = 0
        for k, veh in enumerate(solution):
            if vehicle_type[k] == 2:
                trans0 = small_veh[4]
                veh_param = small_veh
                summ_value[19] += 1
            elif vehicle_type[k] == 3:
                trans0 = medium_veh[4]
                veh_param = medium_veh
                summ_value[20] += 1
            else:
                trans0 = large_veh[4]
                veh_param = large_veh
                summ_value[21] += 1

            departt = check_violation(veh, vehicle_type[k])[3]

            trvl_dist = 0
            veh_load_vol = 0
            wait_time = 0
            trvl_time = 0

            # get the output format
            route = [0] * 21
            route[0] = k + 1  # vehicle name
            route[10] = vehicle_type_dict[vehicle_type[k]]  # 车型


            trvl_dist += (dist_mat[0, veh[1]] + dist_mat[veh[-2], 0])
            trvl_time += (time_mat[0, veh[1]] + time_mat[veh[-2], 0])
            veh_load_vol += (num_demd[veh[1]][0] * bskt_vol + num_demd[veh[1]][1] * trsf_vol + (num_demd[veh[1]][2] +
                             num_demd[veh[1]][3]) * milk_vol + num_demd[veh[1]][4] * paper_bskt)

            summ_value[22] += num_demd[veh[1]][0]
            summ_value[23] += num_demd[veh[1]][1]
            summ_value[24] += num_demd[veh[1]][2]
            summ_value[25] += num_demd[veh[1]][3]
            summ_value[26] += num_demd[veh[1]][4]

            if departt / 60. < 24.:
                out_time = int(departt)
            else:
                out_time = int(departt - 24 * 60)
            route[1] = str(out_time // 60) + ':' + str(out_time % 60).zfill(2)
            t = departt + time_mat[0, veh[1]] + oprt_t
            for i in range(2, len(veh)-1):  # can not wait at the first 2 points
                trvl_dist += dist_mat[veh[i-1], veh[i]]
                trvl_time += time_mat[veh[i-1], veh[i]]
                veh_load_vol += (num_demd[veh[i]][0] * bskt_vol + num_demd[veh[i]][1] * trsf_vol + (num_demd[veh[i]][2] +
                                 num_demd[veh[i]][3]) * milk_vol + num_demd[veh[i]][4] * paper_bskt)

                summ_value[22] += num_demd[veh[i]][0]
                summ_value[23] += num_demd[veh[i]][1]
                summ_value[24] += num_demd[veh[i]][2]
                summ_value[25] += num_demd[veh[i]][3]
                summ_value[26] += num_demd[veh[i]][4]

                wait_t = num_timez[veh[i]][0] - (t + time_mat[veh[i-1], veh[i]])
                if wait_t > 0 + 1e-5:
                    # print veh[i-1], veh[i], wait_t
                    wait_time += wait_t
                    t = num_timez[veh[i]][0] + oprt_t
                else:
                    t += (time_mat[veh[i - 1], veh[i]] + oprt_t)
            if t + time_mat[veh[-2], 0] < 24. * 60:
                in_time = int(t + time_mat[veh[-2], 0])
            else:
                in_time = int(t + time_mat[veh[-2], 0] - 24 * 60)

            route[2] = str(in_time // 60) + ':' + str(in_time % 60).zfill(2)  # 返回时间
            route[3] = round((t + time_mat[veh[-2], 0] - departt) / 60., 1)  # 工作时间
            route[4] = round(trvl_time, 1)  # 行驶时间
            route[5] = round(oprt_t * (len(veh) - 2), 1)  # 操作时间
            route[6] = wait_time
            route[7] = round(trvl_dist / 1000., 2)  # 行驶里程
            route[8] = round((trvl_dist - dist_mat[veh[-2], 0]) / 1000., 2)  # 有效里程
            route[9] = round(dist_mat[veh[-2], 0] / 1000., 2)  # 返空里程
            route[11] = veh_param[2]  # 额定体积
            route[12] = veh_param[1]  # 额定重量
            route[13] = veh_load_vol  # 实际装载体积
            route[14] = 0.  # 实际装载重量
            route[15] = round(veh_load_vol / veh_param[2] * 100, 2)  # 体积装载率
            route[16] = round(route[14] / veh_param[1] * 100, 2)  # 重量装载率
            route[18] = veh_param[-1]  # 固定成本
            route[19] = round(trvl_dist * trans0, 2)  # 运输成本
            route[17] = route[18] + route[19]  # 总成本
            route[20] = len(veh) - 2  # 配送门店总数

            for ele in veh:
                if ele != 0:
                    route.append(num_id[ele][0])
                    route.append(num_id[ele][1])


            result.append(route)
            # print route
            total_dist += route[7]
            # print 'Last leave time: ', int(t) / 60, ':', int(t) % 60
            # print 'total distances: ', route[5]

            summ_value[2] += veh_load_vol
            summ_value[3] += 0
            summ_value[4] = veh_spd_kmh
            summ_value[5] += 1
            summ_value[6] += veh_param[2]
            summ_value[7] += veh_param[1]
            summ_value[10] += round(trvl_dist / 1000., 2)
            summ_value[11] += route[8]
            summ_value[12] += route[9]
            summ_value[13] += route[3]
            summ_value[14] += route[4]
            summ_value[15] += route[5]
            summ_value[16] += route[17]
            summ_value[17] += route[18]
            summ_value[18] += route[19]


        if if_write:
            # run_time = time.strftime("%m%d_%H%M", time.localtime())
            with open(r'C:\Bee\0Huaat\Starbucks\output\Route_Summary_%s_%s.csv' % (veh_spd_kmh, run_time), 'w', newline='') as fw:
                writer = csv.writer(fw)
                for v in result:
                    writer.writerow(v)


            summ_value[0] = run_time
            summ_value[1] = store_num - 1
            summ_value[8] = round(summ_value[2] / summ_value[6] * 100, 2)
            summ_value[9] = round(summ_value[3] / summ_value[7] * 100, 2)
            result_summary.append(summ_value)
            with open(r'C:\Bee\0Huaat\Starbucks\output\Plan_Summary_%s_%s.csv' % (veh_spd_kmh, run_time), 'w', newline='') as fww:
                writer = csv.writer(fww)
                for vv in result_summary:
                    writer.writerow(vv)


        return total_dist



def check_violation(route, vehicle_type):
    """To check if a route is feasible using given vehicle type, and return check result and route cost."""
    if len(route) == 2:  # [0, 0] route
        return True, 0, 0, 0
    else:
        accu_res = [0, 0, 0]  # 0-leaving time, 1-accumulated distance, 2-volume
        if vehicle_type == 2:
            veh_cap = small_veh
        elif vehicle_type == 3:
            veh_cap = medium_veh
        elif vehicle_type == 5:
            veh_cap = large_veh
        else:
            veh_cap = large_veh
            print('Input wrong vehicle type!', vehicle_type)
        # small_veh = [1, 12, 10, 400000, 0.012, 200]
        fixed_cost = veh_cap[5]
        trans_cost = 0
        # wait_cost = 0
        if time_mat[0, route[1]] < num_timez[route[1]][0]:
            accu_res[0] = num_timez[route[1]][0] - time_mat[0, route[1]]  # vehicle leaving depot time
            depart_time = accu_res[0]  # departing from depot time
        else:
            depart_time = 0
        for i in range(len(route) - 1):
            last_cust = route[i]
            curr_cust = route[i+1]
            # checking leaving time
            arr_time = accu_res[0] + time_mat[last_cust, curr_cust]
            if arr_time < num_timez[curr_cust][0]:
                accu_res[0] = num_timez[curr_cust][0] + oprt_t
                wait_time = num_timez[curr_cust][0] - arr_time
                # wait_cost += (wait_time / 60. * wait_cost0)
            elif arr_time <= num_timez[curr_cust][1]:
                accu_res[0] = arr_time + oprt_t
            else:
                # print('Infeasible route!(Service Time Error.)')
                return False, 1000000, 0, 0

            # checking vehicle max distance
            trans_cost += (dist_mat[last_cust, curr_cust] * veh_cap[4])

            accu_res[1] += dist_mat[last_cust, curr_cust]

            if accu_res[0] - oprt_t - depart_time > veh_cap[3]:
                # print('Infeasible route!(Max Time Error.)')
                return False, 1000000, 0, 0

            # checking vehicle max volume
            accu_res[2] += (num_demd[curr_cust][0] * bskt_vol + num_demd[curr_cust][1] * trsf_vol + (num_demd[curr_cust][2]
                            + num_demd[curr_cust][3]) * milk_vol + num_demd[curr_cust][4] * paper_bskt)

            if accu_res[2] > veh_cap[2]:
                # print('Infeasible route!(Max Weight/Volume Error.)', accu_res[2])
                return False, 1000000, 0, 0
    route_cost = fixed_cost + accu_res[1] * veh_cap[4]
    route_dist = accu_res[1]
    route_time = accu_res[0] - oprt_t - depart_time
    # print fixed_cost, trvl_cost, trvl_dist
    return True, route_cost, route_time, depart_time + 600


def plot_route(sol):
    px = loc[:, 0]
    py = loc[:, 1]
    plt.ion()  # 开启interactive mode 成功的关键函数
    plt.figure(1)
    plt.scatter(px, py, color='black', marker='*')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    color_list = ['b', 'g', 'pink', 'purple', 'y', 'grey', 'gold', 'violet']
    col_ind = 0
    for route in sol:
        if col_ind < len(color_list) - 1:
            route_color = color_list[col_ind]
            col_ind += 1
        else:
            route_color = color_list[col_ind]
            col_ind = 0
        for i in range(len(route)-1):
            curr_p = route[i]
            next_p = route[i+1]
            plt.plot([px[curr_p], px[next_p]], [py[curr_p], py[next_p]], color=route_color)
            plt.draw()
            plt.pause(0.1)

    # 关闭交互模式
    plt.ioff()
    plt.show()


def cust_loc(sol, cust):
    """Get the route location and customer location of a customer."""
    cust_ind = []  # [route_loc, cust_loc]
    for i, rt in enumerate(sol):
        if cust in rt:
            cust_ind.append(i)
            cust_ind.append(rt.index(cust))
            return cust_ind

    print('Costomer not in the solution: ', cust)


def route_type(route):
    """Given a route, return the vehicle type of the route. Samll vehicle first, medium second, large last."""
    typ = 2
    vol_accu = 0  # accumulated volume

    if len(route) <= 2:
        return typ
    else:
        for i in range(1, len(route) - 1):
            cust0 = route[i]
            vol_accu += (num_demd[cust0][0] * bskt_vol + num_demd[cust0][1] * trsf_vol + (num_demd[cust0][2] +
                         num_demd[cust0][3]) * milk_vol + num_demd[cust0][4] * paper_bskt)

    if vol_accu <= small_veh[2]:
        return 2
    elif vol_accu <= medium_veh[2]:
        return 3
    elif vol_accu <= large_veh[2]:
        return 5
    else:
        print('!!!Route is invalid: out of max volume!', route)


class SimulatedAnnealing(object):
    """Using simulated annealing algorithm to refine the initial solution.
    7 operators are used to generate vrp neighbour."""

    def __init__(self):
        self.b = 6


    def shift_1_cust(self, sol_in1, cust, c_loc, curr_temp, sol_type1, sa_lns):
        """Try to move 1 customer to anywhere it can be put, and see if the move can cut the total cost."""

        route_ing = copy.deepcopy(sol_in1[c_loc[0]])
        route_new = route_ing
        move_to_route = c_loc[0]
        orgn_type1 = sol_type1[c_loc[0]]
        origin_cost1 = check_violation(route_ing, orgn_type1)[1]
        route_ing.remove(cust)  # move c in the current route
        new_type1 = route_type(route_ing)
        adjust_cost1 = check_violation(route_ing, new_type1)[1]
        best_cut_cost0 = -1000
        best_cut_cost = best_cut_cost0  # best cost cut of moving this customer
        for j, rou in enumerate(sol_in1):
            orgn_type2 = sol_type1[j]
            origin_cost2 = check_violation(rou, orgn_type2)[1]
            if j == c_loc[0]:  # moving in the same route
                for k in range(1, len(route_ing)):
                    if k == c_loc[1]:
                        continue  # do not put it at the original position
                    rou_test = route_ing[:k] + [cust] + route_ing[k:]
                    if check_violation(rou_test, orgn_type2)[0]:
                        adjust_cost2 = check_violation(rou_test, orgn_type2)[1]
                        cost_cut_test = origin_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new = rou_test
                            move_to_route = j


            else:  # moving to a different route
                for k in range(1, len(rou)):
                    rou_test = rou[:k] + [cust] + rou[k:]

                    if check_violation(rou_test, 5)[0]:
                        new_type2 = route_type(rou_test)
                        adjust_cost2 = check_violation(rou_test, new_type2)[1]
                        cost_cut_test = origin_cost1 + origin_cost2 - adjust_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new = rou_test
                            move_to_route = j


        if best_cut_cost > 1e-5:
            # print('shift1 good', best_cut_cost)
            sol_in1[move_to_route] = route_new
            sol_type1[move_to_route] = route_type(route_new)
            if move_to_route != c_loc[0]:  # moving to a different route
                sol_in1[c_loc[0]] = route_ing
                sol_type1[c_loc[0]] = route_type(route_ing)
        elif sa_lns and best_cut_cost < -1e-5:
            prb = random.uniform(0, 1)
            if np.exp(best_cut_cost/curr_temp) > prb:
                # print('shift1', best_cut_cost)
                sol_in1[move_to_route] = route_new
                sol_type1[move_to_route] = route_type(route_new)
                if move_to_route != c_loc[0]:  # moving to a different route
                    sol_in1[c_loc[0]] = route_ing
                    sol_type1[c_loc[0]] = route_type(route_ing)



        # return sol_in1


    def shift_2_cust(self, sol_in2, cust, c_loc, curr_temp, sol_type2, sa_lns):
        """Try to move 2 consecutive customers to anywhere they can be put, see if they move can cut the total cost."""

        route_ing = copy.deepcopy(sol_in2[c_loc[0]])
        route_new = route_ing
        move_to_route = c_loc[0]
        orgn_type1 = sol_type2[c_loc[0]]
        cust_folw = route_ing[c_loc[1]+1]
        origin_cost1 = check_violation(route_ing, orgn_type1)[1]
        route_ing.remove(cust)  # remove c in the current route
        del route_ing[c_loc[1]]  # remove customer following c
        new_type1 = route_type(route_ing)
        adjust_cost1 = check_violation(route_ing, new_type1)[1]
        best_cut_cost0 = -1000
        best_cut_cost = best_cut_cost0  # best cost cut of moving this customer
        for j, rou in enumerate(sol_in2):
            orgn_type2 = sol_type2[j]
            origin_cost2 = check_violation(rou, orgn_type2)[1]
            if j == c_loc[0]:  # moving in the same route
                for k in range(1, len(route_ing)):
                    if k == c_loc[1]:
                        continue
                    rou_test = route_ing[:k] + [cust, cust_folw] + route_ing[k:]
                    if check_violation(rou_test, orgn_type2)[0]:
                        adjust_cost2 = check_violation(rou_test, orgn_type2)[1]
                        cost_cut_test = origin_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new = rou_test
                            move_to_route = j


            else:  # moving to a different route
                for k in range(1, len(rou)):
                    rou_test = rou[:k] + [cust, cust_folw] + rou[k:]
                    if check_violation(rou_test, 5)[0]:
                        new_type2 = route_type(rou_test)
                        adjust_cost2 = check_violation(rou_test, new_type2)[1]
                        cost_cut_test = origin_cost1 + origin_cost2 - adjust_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new = rou_test
                            move_to_route = j


        if best_cut_cost > 1e-5:
            # print('shift2 good', best_cut_cost)
            sol_in2[move_to_route] = route_new
            sol_type2[move_to_route] = route_type(route_new)
            if move_to_route != c_loc[0]:  # moving to a different route
                sol_in2[c_loc[0]] = route_ing
                sol_type2[c_loc[0]] = route_type(route_ing)

        elif sa_lns and best_cut_cost < -1e-5:
            prb = random.uniform(0, 1)
            if np.exp(best_cut_cost / curr_temp) > prb:
                # print('shift2', best_cut_cost)
                sol_in2[move_to_route] = route_new
                sol_type2[move_to_route] = route_type(route_new)
                if move_to_route != c_loc[0]:  # moving to a different route
                    sol_in2[c_loc[0]] = route_ing
                    sol_type2[c_loc[0]] = route_type(route_ing)

        # return sol_in2



    def shift_3_cust(self, sol_in6, cust, c_loc, curr_temp, sol_type6, sa_lns):
        """Try to move 3 consecutive customers to anywhere they can be put, see if they move can cut the total cost."""

        route_ing = copy.deepcopy(sol_in6[c_loc[0]])
        route_new = route_ing
        move_to_route = c_loc[0]
        orgn_type1 = sol_type6[c_loc[0]]
        cust_folw1 = route_ing[c_loc[1] + 1]
        cust_folw2 = route_ing[c_loc[1] + 2]
        origin_cost1 = check_violation(route_ing, orgn_type1)[1]
        route_ing.remove(cust)  # remove c in the current route
        del route_ing[c_loc[1]]  # remove customer following c
        del route_ing[c_loc[1]]  # remove customer following following c
        new_type1 = route_type(route_ing)
        adjust_cost1 = check_violation(route_ing, new_type1)[1]
        best_cut_cost0 = -1000
        best_cut_cost = best_cut_cost0  # best cost cut of moving this customer
        for j, rou in enumerate(sol_in6):
            orgn_type2 = sol_type6[j]
            origin_cost2 = check_violation(rou, orgn_type2)[1]
            if j == c_loc[0]:  # moving in the same route
                for k in range(1, len(route_ing)):
                    if k == c_loc[1]:
                        continue
                    rou_test = route_ing[:k] + [cust, cust_folw1, cust_folw2] + route_ing[k:]
                    if check_violation(rou_test, orgn_type2)[0]:
                        adjust_cost2 = check_violation(rou_test, orgn_type2)[1]
                        cost_cut_test = origin_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new = rou_test
                            move_to_route = j

            else:  # moving to a different route
                for k in range(1, len(rou)):
                    rou_test = rou[:k] + [cust, cust_folw1, cust_folw2] + rou[k:]
                    if check_violation(rou_test, 5)[0]:
                        new_type2 = route_type(rou_test)
                        adjust_cost2 = check_violation(rou_test, new_type2)[1]
                        cost_cut_test = origin_cost1 + origin_cost2 - adjust_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new = rou_test
                            move_to_route = j


        if best_cut_cost > 1e-5:
            # print('shift3 good', best_cut_cost)
            sol_in6[move_to_route] = route_new
            sol_type6[move_to_route] = route_type(route_new)
            if move_to_route != c_loc[0]:  # moving to a different route
                sol_in6[c_loc[0]] = route_ing
                sol_type6[c_loc[0]] = route_type(route_ing)

        elif sa_lns and best_cut_cost < -1e-5:

            prb = random.uniform(0, 1)
            if np.exp(best_cut_cost / curr_temp) > prb:
                # print('shift3', best_cut_cost)
                sol_in6[move_to_route] = route_new
                sol_type6[move_to_route] = route_type(route_new)
                if move_to_route != c_loc[0]:  # moving to a different route
                    sol_in6[c_loc[0]] = route_ing
                    sol_type6[c_loc[0]] = route_type(route_ing)


    def exchange_1_cust(self, sol_in3, cust, c_loc, curr_temp, sol_type3, sa_lns):
        """Exchange the position of two customers(same route or not) if feasible,
        and see if it can cut the total cost."""

        route_ing = copy.deepcopy(sol_in3[c_loc[0]])

        route_new_1 = route_ing
        route_new_2 = route_ing
        exch_to_route = c_loc[0]
        orgn_type1 = sol_type3[exch_to_route]
        origin_cost1 = check_violation(route_ing, orgn_type1)[1]
        # route_ing.remove(cust)  # move c in the current route
        # adjust_cost1 = check_violation(route_ing)[1]
        best_cut_cost0 = -1000
        best_cut_cost = best_cut_cost0  # best cost cut of moving this customer
        for j, rou in enumerate(sol_in3):
            orgn_type2 = sol_type3[j]
            origin_cost2 = check_violation(rou, orgn_type2)[1]
            if j == c_loc[0]:  # exchange in the same route
                for k in range(1, len(rou)-1):
                    if k == c_loc[1]:
                        continue
                    rou_test = copy.deepcopy(sol_in3[c_loc[0]])
                    rou_test[k], rou_test[c_loc[1]] = rou_test[c_loc[1]], rou_test[k]
                    if check_violation(rou_test, orgn_type2)[0]:
                        adjust_cost2 = check_violation(rou_test, orgn_type2)[1]
                        cost_cut_test = origin_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new_1 = rou_test
                            route_new_2 = rou_test
                            exch_to_route = j

            else:  # exchange to a different route
                for k in range(1, len(rou)-1):
                    rou_test_1 = copy.deepcopy(sol_in3[c_loc[0]])
                    rou_test_2 = copy.deepcopy(rou)
                    rou_test_1[c_loc[1]] = rou[k]
                    rou_test_2[k] = cust
                    if check_violation(rou_test_1, 5)[0] and check_violation(rou_test_2, 5)[0]:
                        new_type1 = route_type(rou_test_1)
                        new_type2 = route_type(rou_test_2)
                        adjust_cost1 = check_violation(rou_test_1, new_type1)[1]
                        adjust_cost2 = check_violation(rou_test_2, new_type2)[1]
                        cost_cut_test = origin_cost1 + origin_cost2 - adjust_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new_1 = rou_test_1
                            route_new_2 = rou_test_2
                            exch_to_route = j



        if best_cut_cost > 1e-5:
            # print('exchange1 good', best_cut_cost)
            sol_in3[c_loc[0]] = route_new_1
            sol_in3[exch_to_route] = route_new_2
            sol_type3[c_loc[0]] = route_type(route_new_1)
            sol_type3[exch_to_route] = route_type(route_new_2)

        elif sa_lns and best_cut_cost < -1e-5:
            prb = random.uniform(0, 1)
            if np.exp(best_cut_cost / curr_temp) > prb:
                # print('exchange1', best_cut_cost)
                sol_in3[c_loc[0]] = route_new_1
                sol_in3[exch_to_route] = route_new_2
                sol_type3[c_loc[0]] = route_type(route_new_1)
                sol_type3[exch_to_route] = route_type(route_new_2)

        # return sol_in3


    def exchange_2_cust(self, sol_in4, cust, c_loc, curr_temp, sol_type4, sa_lns):
        """Exchange 2 consecutive customers' position with another 2 customers' position, and see if it can cut cost."""

        route_ing = copy.deepcopy(sol_in4[c_loc[0]])
        route_new_1 = route_ing
        route_new_2 = route_ing
        cust_folw = route_ing[c_loc[1] + 1]
        exch_to_route = c_loc[0]
        origin_cost1 = check_violation(route_ing, sol_type4[c_loc[0]])[1]
        # route_ing.remove(cust)  # move c in the current route
        # adjust_cost1 = check_violation(route_ing)[1]
        best_cut_cost0 = -1000
        best_cut_cost = best_cut_cost0  # best cost cut of moving this customer
        for j, rou in enumerate(sol_in4):
            origin_cost2 = check_violation(rou, sol_type4[j])[1]
            if j != c_loc[0] and len(rou) >= 4:  # exchange to a different route
                for k in range(1, len(rou) - 2):
                    rou_test_1 = copy.deepcopy(sol_in4[c_loc[0]])
                    rou_test_2 = copy.deepcopy(rou)
                    rou_test_1[c_loc[1]], rou_test_1[c_loc[1] + 1] = rou[k], rou[k + 1]
                    rou_test_2[k], rou_test_2[k + 1] = cust, cust_folw
                    if check_violation(rou_test_1, 5)[0] and check_violation(rou_test_2, 5)[0]:
                        new_type1 = route_type(rou_test_1)
                        new_type2 = route_type(rou_test_2)
                        adjust_cost1 = check_violation(rou_test_1, new_type1)[1]
                        adjust_cost2 = check_violation(rou_test_2, new_type2)[1]
                        cost_cut_test = origin_cost1 + origin_cost2 - adjust_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new_1 = rou_test_1
                            route_new_2 = rou_test_2
                            exch_to_route = j



        if best_cut_cost > 1e-5:
            # print('exchange2 good', best_cut_cost)
            sol_in4[c_loc[0]] = route_new_1
            sol_in4[exch_to_route] = route_new_2
            sol_type4[c_loc[0]] = route_type(route_new_1)
            sol_type4[exch_to_route] = route_type(route_new_2)

        elif sa_lns and best_cut_cost < -1e-5:
            prb = random.uniform(0, 1)
            if np.exp(best_cut_cost / curr_temp) > prb:
                # print('exchange2', best_cut_cost)
                sol_in4[c_loc[0]] = route_new_1
                sol_in4[exch_to_route] = route_new_2
                sol_type4[c_loc[0]] = route_type(route_new_1)
                sol_type4[exch_to_route] = route_type(route_new_2)

        # return sol_in4


    def two_exchange_sol(self, sol_in5, curr_temp, sol_type5, sa_lns):
        """Two-Exchange operator: For two customers i and j on the same route where i is visited before j,
        remove arcs (i,i+),(j,j+); add arcs (i,j),(i+,j+); and reverse the orientation of the arcs between i+ and j.
        Given a solution, check all possible neighborhood.
        """
        solu = copy.deepcopy(sol_in5)
        best_cut_cost0 = -1000
        best_cut_cost = best_cut_cost0  # best cost cut of moving this customer
        adjust_rou_ind = 0
        route_new = sol_in5[adjust_rou_ind]

        for i, rou in enumerate(solu):
            if len(rou) >= 6:
                orgn_type = sol_type5[i]
                origin_cost = check_violation(rou, orgn_type)[1]
                for k in range(1, len(rou)-4):
                    for l in range(k+3, len(rou)-1):
                        route_test = copy.deepcopy(rou)
                        route_test[k], route_test[l] = route_test[l], route_test[k]
                        route_test[k+1: l] = route_test[l-1:k:-1]  # middle reverse
                        if check_violation(route_test, orgn_type)[0]:
                            adjust_cost = check_violation(route_test, orgn_type)[1]
                            if origin_cost - adjust_cost > best_cut_cost:
                                best_cut_cost = origin_cost - adjust_cost
                                adjust_rou_ind = i
                                route_new = route_test



        if best_cut_cost > 1e-5:
            # print('2exchange good', best_cut_cost)
            sol_in5[adjust_rou_ind] = route_new
            sol_type5[adjust_rou_ind] = route_type(route_new)

        elif sa_lns and best_cut_cost < -1e-5:
            prb = random.uniform(0, 1)
            if np.exp(best_cut_cost / curr_temp) > prb:
                # print('2exchange', best_cut_cost)
                sol_in5[adjust_rou_ind] = route_new
                sol_type5[adjust_rou_ind] = route_type(route_new)

        # return sol_in5


    def two_opt(self, sol_in7, cust, c_loc, curr_temp, sol_type7, sa_lns):
        """2-opt*: given customer i in route a and customer j in route b, exchange the following sequences of i and j.
        for example, initial route a: ...-i-i1-i2-..., initial route b: ...-j-j1-j2-...
        New route a: ...-i-j1-j2-..., new route b: ...-j-i1-i2-..."""

        route_ing = copy.deepcopy(sol_in7[c_loc[0]])

        route_new_1 = route_ing
        route_new_2 = route_ing
        exch_to_route = c_loc[0]
        orgn_type1 = sol_type7[c_loc[0]]
        origin_cost1 = check_violation(route_ing, orgn_type1)[1]
        # route_ing.remove(cust)  # move c in the current route
        # adjust_cost1 = check_violation(route_ing)[1]
        best_cut_cost0 = -1000
        best_cut_cost = best_cut_cost0  # best cost cut of moving this customer
        for j, rou in enumerate(sol_in7):
            orgn_type2 = sol_type7[j]
            origin_cost2 = check_violation(rou, orgn_type2)[1]
            if j != c_loc[0]:  # 2-opt* operator has to be implemented in 2 different routes
                for k in range(1, len(rou) - 1):
                    rou_test_1 = sol_in7[c_loc[0]][:c_loc[1]] + rou[k:]
                    rou_test_2 = rou[:k] + sol_in7[c_loc[0]][c_loc[1]:]
                    if check_violation(rou_test_1, 5)[0] and check_violation(rou_test_2, 5)[0]:
                        new_type1 = route_type(rou_test_1)
                        new_type2 = route_type(rou_test_2)
                        adjust_cost1 = check_violation(rou_test_1, new_type1)[1]
                        adjust_cost2 = check_violation(rou_test_2, new_type2)[1]
                        cost_cut_test = origin_cost1 + origin_cost2 - adjust_cost1 - adjust_cost2
                        if cost_cut_test > best_cut_cost:
                            best_cut_cost = cost_cut_test
                            route_new_1 = rou_test_1
                            route_new_2 = rou_test_2
                            exch_to_route = j


        if best_cut_cost > 1e-5:
            # print('2opt* good', best_cut_cost)
            sol_in7[c_loc[0]] = route_new_1
            sol_in7[exch_to_route] = route_new_2
            sol_type7[c_loc[0]] = route_type(route_new_1)
            sol_type7[exch_to_route] = route_type(route_new_2)

        elif sa_lns and best_cut_cost < -1e-5:
            prb = random.uniform(0, 1)
            if np.exp(best_cut_cost / curr_temp) > prb:
                # print('2opt*', best_cut_cost)
                sol_in7[c_loc[0]] = route_new_1
                sol_in7[exch_to_route] = route_new_2
                sol_type7[c_loc[0]] = route_type(route_new_1)
                sol_type7[exch_to_route] = route_type(route_new_2)


    def lns_sa(self, sol_in, veh_type_in, cost_in):
        """Neighborhood search based on 7 operators. In each iteration, select one operator randomly."""

        itr_cost = []
        solu = copy.deepcopy(sol_in)
        solu_type = copy.deepcopy(veh_type_in)
        best_solu = sol_in
        best_val = cost_in
        tabu_list = []
        random.seed(10)
        itr = 0
        temp = initial_temp
        t_run = time.time()
        while temp > stop_temp:
            itr += 1
            print(itr)
            if itr <= 0:
                sa_lns = True  # use sa or lns
            else:
                sa_lns = False
            c = random.randint(1, store_num - 1)  # randomly generated moving customer
            while c in tabu_list:
                c = random.randint(1, store_num - 1)  # randint(a, b), both a and b are selectable
            c_loc = cust_loc(solu, c)

            if len(solu[c_loc[0]]) < 4:  # customer number less than 2, can only implement shift1 and exchange1 operator
                wheel_value1 = random.uniform(0, 1)
                if wheel_value1 < 0.45:
                    self.shift_1_cust(solu, c, c_loc, temp, solu_type, sa_lns)
                elif wheel_value1 < 0.9:
                    self.exchange_1_cust(solu, c, c_loc, temp, solu_type, sa_lns)
                else:
                    self.two_opt(solu, c, c_loc, temp, solu_type, sa_lns)

            # customer number more than 2, can implement all operators
            elif len(solu[c_loc[0]]) >= 4 and c_loc[1] <= len(solu[c_loc[0]]) - 3:
                wheel_value2 = random.uniform(0, 1)
                if wheel_value2 < 0.2:
                    self.shift_1_cust(solu, c, c_loc, temp, solu_type, sa_lns)
                elif wheel_value2 < 0.4:
                    self.shift_2_cust(solu, c, c_loc, temp, solu_type, sa_lns)
                elif wheel_value2 < 0.6:
                    self.exchange_1_cust(solu, c, c_loc, temp, solu_type, sa_lns)
                elif wheel_value2 < 0.8:
                    self.exchange_2_cust(solu, c, c_loc, temp, solu_type, sa_lns)
                else:
                    self.two_opt(solu, c, c_loc, temp, solu_type, sa_lns)


            if itr % 100 == 0:  # implement two-exchange operator every 200 iteration
                self.two_exchange_sol(solu, temp, solu_type, sa_lns)


            temp -= delta
            tabu_list.append(c)
            if len(tabu_list) > 100:
                tabu_list.pop(0)


            cost_i = of.print_result(solu, solu_type, False)
            # print(solu_type)
            itr_cost.append(cost_i)
            if cost_i < best_val:
                best_solu = solu
                best_val = cost_i


            t_run = time.time()

        # Adjust0: delete [0, 0] routes
        adjust_sol0 = []
        for route0 in best_solu:
            if len(route0) <= 2:  # [0, 0] route
                continue
            else:
                adjust_sol0.append(route0)

        # Adjust1: use small vehicle if posiible
        adjust_type = []
        for route1 in adjust_sol0:
            adjust_type.append(route_type(route1))



        return adjust_sol0, adjust_type, best_val, itr_cost




if __name__ == '__main__':
    t0 = time.time()

    num_id, id_num, loc, num_demd, num_timez = read_data()
    dist_mat, time_mat = earth_dist(loc)
    dist_mat = dist_mat * math.sqrt(2)  # the transfer coefficient of travel distance and euclidean distance is sqrt(2)
    time_mat = time_mat * math.sqrt(2)
    store_num = len(num_id)  # number of stores including depot

    gi = GetInitial()
    of = OutputFormat()

    solu, solu_type, route_way_time = gi.greedy_initial()
    solu_cost = of.print_result(solu, solu_type, False)
    print('Initial number of vehicles: ', len(solu))
    print('Initial solution cost: ', solu_cost)
    # print(solu_veh_type)


    # Simulated Annealing

    sa = SimulatedAnnealing()
    initial_temp, stop_temp, delta = 40., 10., 30. / 10000
    new_solu, new_solu_type, new_cost, cost_t = sa.lns_sa(solu, solu_type, solu_cost)
    print('Optimized number of vehicles: ', len(new_solu))
    print('Optimized solution cost: ', new_cost)
    plt.plot(cost_t)
    plt.show()

    # total_dist1 = of.print_route_summary(solution=new_solu, vehicle_type=new_solu_type, if_write=True)
    # of.print_route_detail(solution=new_solu, vehicle_type=new_solu_type, if_write=True)
    # print('Total traveling diatance: ', total_dist1)

    # plot_route(new_solu)

    t1 = time.time()
    print('Total elapsed time: ', t1-t0)





