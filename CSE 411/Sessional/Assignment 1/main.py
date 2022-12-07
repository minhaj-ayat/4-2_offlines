import math
import statistics

import matplotlib.pyplot as plot
import numpy as np

Q_LIMIT = 100
BUSY = 1
IDLE = 0

delays_required = 0
mean_interval_time = 0
mean_service_time = 0

clock_time = 0
num_events = 2
next_event_type = 0
server_status = IDLE
q_length = 0
last_event_time = 0
customers_delayed = 0
total_delayed = 0
sum_area_q = 0
sum_area_u = 0
next_event_time = [0, 0]
arrival_time = [0] * (q_length+1)

list_ua = []
list_us = []
list_ea = []
list_es = []

STREAM = 5
MODLUS = 2147483647
MULT1 = 24112
MULT2 = 26143
zrng = [
    1,
    1973272912, 281629770, 20006270, 1280689831, 2096730329, 1933576050,
    913566091, 246780520, 1363774876, 604901985, 1511192140, 1259851944,
    824064364, 150493284, 242708531, 75253171, 1964472944, 1202299975,
    233217322, 1911216000, 726370533, 403498145, 993232223, 1103205531,
    762430696, 1922803170, 1385516923, 76271663, 413682397, 726466604,
    336157058, 1432650381, 1120463904, 595778810, 877722890, 1046574445,
    68911991, 2088367019, 748545416, 622401386, 2122378830, 640690903,
    1774806513, 2132545692, 2079249579, 78130110, 852776735, 1187867272,
    1351423507, 1645973084, 1997049139, 922510944, 2045512870, 898585771,
    243649545, 1004818771, 773686062, 403188473, 372279877, 1901633463,
    498067494, 2087759558, 493157915, 597104727, 1530940798, 1814496276,
    536444882, 1663153658, 855503735, 67784357, 1432404475, 619691088,
    119025595, 880802310, 176192644, 1116780070, 277854671, 1366580350,
    1142483975, 2026948561, 1053920743, 786262391, 1792203830, 1494667770,
    1923011392, 1433700034, 1244184613, 1147297105, 539712780, 1545929719,
    190641742, 1645390429, 264907697, 620389253, 1502074852, 927711160,
    364849192, 2049576050, 638580085, 547070247]


def lcgrand(stream, sig):
    zi = zrng[stream]
    lowprd = (zi & 65535) * MULT1
    hi31 = (zi >> 16) * MULT1 + (lowprd >> 16)
    zi = ((lowprd & 65535) - MODLUS) + ((hi31 & 32767) << 16) + (hi31 >> 15)
    if zi < 0:
        zi += MODLUS
    lowprd = (zi & 65535) * MULT2
    hi31 = (zi >> 16) * MULT2 + (lowprd >> 16)
    zi = ((lowprd & 65535) - MODLUS) + ((hi31 & 32767) << 16) + (hi31 >> 15)
    if zi < 0:
        zi += MODLUS
    zrng[stream] = zi
    val = (zi >> 7 | 1) / 16777216.0
    if sig == 1:
        list_ua.append(val)
    else:
        list_us.append(val)
    return val


def expon(mean,sig):
    val = (-1 * mean) * math.log(lcgrand(STREAM,sig))
    if sig == 1:
        list_ea.append(val)
    else:
        list_es.append(val)
    return val


def initialize():
    global mean_interval_time
    next_event_time[0] = clock_time + expon(mean_interval_time,1)
    next_event_time[1] = 100000


def readInput(k):
    k = float(k)
    file = open("inp.txt", "r")
    st = file.readline().split()
    global delays_required, mean_interval_time, mean_service_time, arrival_time
    mean_interval_time = float(st[0])
    if k > 1:
        mean_service_time = 0.9
    else:
        mean_service_time = float(st[0]) * k
    delays_required = int(st[2])
    arrival_time = [0] * (delays_required + 1)
    print(mean_service_time, mean_interval_time, delays_required)


def timing():
    global next_event_type
    min_time = 1e29
    next_event_type = 0

    for i in (1, num_events):
        if next_event_time[i-1] < min_time:
            min_time = next_event_time[i-1]
            next_event_type = i-1

    global clock_time
    clock_time = min_time


def update_area_stats():
    global last_event_time, sum_area_q, sum_area_u
    time_since_last_event = clock_time - last_event_time
    last_event_time = clock_time

    sum_area_q += q_length * time_since_last_event
    sum_area_u += server_status * time_since_last_event


def arrive():
    global q_length, arrival_time, total_delayed, customers_delayed, server_status, mean_service_time, mean_interval_time

    next_event_time[0] = clock_time + expon(mean_interval_time,1)

    if server_status == BUSY:
        q_length = q_length + 1
        arrival_time[q_length] = clock_time

    else:
        delay = 0
        total_delayed += delay
        customers_delayed = customers_delayed + 1
        server_status = BUSY
        next_event_time[1] = clock_time + expon(mean_service_time,2)


def depart():
    global q_length, server_status, total_delayed, customers_delayed
    if q_length == 0:
        server_status = IDLE
        next_event_time[1] = 100000

    else:
        q_length = q_length - 1
        delay = clock_time - arrival_time[1]
        total_delayed += delay
        customers_delayed = customers_delayed + 1
        next_event_time[1] = clock_time + expon(mean_service_time,2)

        for i in (1, q_length+1):
            arrival_time[i] = arrival_time[i+1]


def report():
    global total_delayed
    #print(total_delayed)
    print("Average delay in queue:", total_delayed/customers_delayed)
    print("Average number in queue:", sum_area_q/clock_time)
    print("Server utilization:", sum_area_u/clock_time)
    print("Time simulation ended:", clock_time)

    print("\nMin of uniform dis. of arrival : ", min(list_ua))
    print("Min of uniform dis. of service : ", min(list_us))
    print("Max of uniform dis. of arrival : ", max(list_ua))
    print("Max of uniform dis. of service : ", max(list_us))
    print("Median of uniform dis. of arrival : ", statistics.median(list_ua))
    print("Median of uniform dis. of service : ", statistics.median(list_us))

    print("\nMin of exp. dis. of arrival : ", min(list_ea))
    print("Min of exp. dis. of service : ", min(list_es))
    print("Max of exp. dis. of arrival : ", max(list_ea))
    print("Max of exp. dis. of service : ", max(list_es))
    print("Median of exp. dis. of arrival : ", statistics.median(list_ea))
    print("Median of exp. dis. of service : ", statistics.median(list_es))

    print("\nP(x)-F(x) for uniform dist. of arrival")
    count = [0,0,0,0,0,0,0,0,0,0]
    for v in list_ua:
        if v < 0.1:
            count[0] = count[0] + 1
        elif v < 0.2:
            count[1] = count[1] + 1
        elif v < 0.3:
            count[2] = count[2] + 1
        elif v < 0.4:
            count[3] = count[3] + 1
        elif v < 0.5:
            count[4] = count[4] + 1
        elif v < 0.6:
            count[5] = count[5] + 1
        elif v < 0.7:
            count[6] = count[6] + 1
        elif v < 0.8:
            count[7] = count[7] + 1
        elif v < 0.9:
            count[8] = count[8] + 1
        else:
            count[9] = count[9] + 1

    print(count)
    fx = 0
    for i in count:
        fx += i
        print(i / customers_delayed, "\t\t", fx / customers_delayed)

    print("\nP(x)-F(x) for uniform dist. of service")
    count = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for v in list_us:
        if v < 0.1:
            count[0] = count[0] + 1
        elif v < 0.2:
            count[1] = count[1] + 1
        elif v < 0.3:
            count[2] = count[2] + 1
        elif v < 0.4:
            count[3] = count[3] + 1
        elif v < 0.5:
            count[4] = count[4] + 1
        elif v < 0.6:
            count[5] = count[5] + 1
        elif v < 0.7:
            count[6] = count[6] + 1
        elif v < 0.8:
            count[7] = count[7] + 1
        elif v < 0.9:
            count[8] = count[8] + 1
        else:
            count[9] = count[9] + 1

    print(count)
    fx = 0
    for i in count:
        fx += i
        print(i/customers_delayed, "\t\t", fx/customers_delayed)

    print("\nP(x)-F(x) for expo. dist. of arrival")
    count = [0, 0, 0, 0]
    for v in list_ea:
        if v < mean_interval_time/2:
            count[0] = count[0] + 1
        elif v < mean_interval_time:
            count[1] = count[1] + 1
        elif v < 2*mean_interval_time:
            count[2] = count[2] + 1
        else:
            count[3] = count[3] + 1

    print(count)
    fx = 0
    for i in count:
        fx += i
        print(i / customers_delayed, "\t\t", fx / customers_delayed)

    print("\nP(x)-F(x) for expo. dist. of service")
    count = [0, 0, 0, 0]
    for v in list_es:
        if v < mean_service_time / 2:
            count[0] = count[0] + 1
        elif v < mean_service_time:
            count[1] = count[1] + 1
        elif v < 2 * mean_service_time:
            count[2] = count[2] + 1
        else:
            count[3] = count[3] + 1

    print(count)
    fx = 0
    for i in count:
        fx += i
        print(i / customers_delayed, "\t\t", fx / customers_delayed)


kv = input("Enter value of k : ")
readInput(kv)
initialize()

while customers_delayed < delays_required:
    timing()
    update_area_stats()

    if next_event_type == 0:
        arrive()
    else:
        depart()
report()
