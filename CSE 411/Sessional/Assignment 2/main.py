import random

import numpy as np
import lib as lib

END_TIME = 0
NUMBER_OF_FLOORS = 0
NUMBER_OF_ELEVATORS = 0
LIFT_CAPACITY = 0
BATCH_SIZE = 0
DOOR_HOLD_TIME = 0
INTER_FLOOR_ARRIVAL_TIME = 0
DOOR_OPEN_TIME = 0
DOOR_CLOSE_TIME = 0
EMBARK_TIME = 0
DISEMBARK_TIME = 0
MEAN_INTERVAL_TIME = 0

STREAM = random.randint(0, 100)


def readInput():
    global END_TIME, NUMBER_OF_ELEVATORS, NUMBER_OF_FLOORS, LIFT_CAPACITY, BATCH_SIZE, DOOR_HOLD_TIME, \
        INTER_FLOOR_ARRIVAL_TIME, DOOR_OPEN_TIME, DOOR_CLOSE_TIME, EMBARK_TIME, DISEMBARK_TIME, MEAN_INTERVAL_TIME
    c = 1
    file = open("inp.txt", "r")
    lines = file.readlines()
    for l in lines:
        st = l.split()
        if c == 1:
            END_TIME = int(st[0])
            c += 1
            print(END_TIME)
        elif c == 2:
            NUMBER_OF_FLOORS = int(st[0])
            NUMBER_OF_ELEVATORS = int(st[1])
            LIFT_CAPACITY = int(st[2])
            BATCH_SIZE = int(st[3])
            c += 1
            print(NUMBER_OF_FLOORS, NUMBER_OF_ELEVATORS, LIFT_CAPACITY, BATCH_SIZE)
        elif c == 3:
            DOOR_HOLD_TIME = int(st[0])
            INTER_FLOOR_ARRIVAL_TIME = int(st[1])
            DOOR_OPEN_TIME = int(st[2])
            DOOR_CLOSE_TIME = int(st[3])
            c += 1
            print(DOOR_HOLD_TIME, INTER_FLOOR_ARRIVAL_TIME, DOOR_OPEN_TIME, DOOR_CLOSE_TIME)
        elif c == 4:
            EMBARK_TIME = int(st[0])
            DISEMBARK_TIME = int(st[1])
            c += 1
            print(EMBARK_TIME, DISEMBARK_TIME)
        elif c == 5:
            MEAN_INTERVAL_TIME = float(st[0])
            c += 1
            print(MEAN_INTERVAL_TIME)


def simulate():
    global END_TIME, NUMBER_OF_ELEVATORS, NUMBER_OF_FLOORS, LIFT_CAPACITY, BATCH_SIZE, DOOR_HOLD_TIME, \
        INTER_FLOOR_ARRIVAL_TIME, DOOR_OPEN_TIME, DOOR_CLOSE_TIME, EMBARK_TIME, DISEMBARK_TIME, MEAN_INTERVAL_TIME

    # STEP 1
    DELTIME = 0
    ELEVTIME = 0
    MAXDEL = 0
    MAXELEV = 0
    QUELEN = 0
    QUETIME = 0
    MAXQUE = 0
    remain = 0
    quetotal = 0
    quecust = 0
    startque = 0
    queue = 0
    quetotal = 0
    remain = 0

    # my variables
    step8 = False
    step20 = False
    area_q = 0
    MAXWAIT = 0

    between = {}
    arrive = {}
    floor = {}
    elevator = {}
    wait = {}
    delivery = {}

    selvec = [[0 for x in range(0, NUMBER_OF_FLOORS)], [0 for x in range(0, NUMBER_OF_FLOORS)],
              [0 for x in range(0, NUMBER_OF_FLOORS)], [0 for x in range(0, NUMBER_OF_FLOORS)]]
    flrvec = [[0 for x in range(0, NUMBER_OF_FLOORS)], [0 for x in range(0, NUMBER_OF_FLOORS)],
              [0 for x in range(0, NUMBER_OF_FLOORS)], [0 for x in range(0, NUMBER_OF_FLOORS)]]
    occup = [0, 0, 0, 0]
    ereturn = [0, 0, 0, 0]
    first = [0, 0, 0, 0]
    stop = [0, 0, 0, 0]
    eldel = [0, 0, 0, 0]
    operate = [0, 0, 0, 0]
    load_size = [0, 0, 0, 0]
    departs = [0, 0, 0, 0]
    j = -1

    limit = 0

    # STEP 2
    i = 1
    between[i] = (lib.expon(MEAN_INTERVAL_TIME)) * 60
    floor[i] = (random.randint(2, NUMBER_OF_FLOORS) - 1)
    # print("FLOOR[i]",floor[i])
    delivery[i] = DOOR_HOLD_TIME
    wait[i] = 0

    # STEP 3
    TIME = between[i]
    arrive[i] = TIME
    batch = random.randint(1, 6)

    for k in range(0, NUMBER_OF_ELEVATORS):
        ereturn[k] = TIME
        stop[k] = 0
        operate[k] = 0

    # STEP 4
    while TIME <= END_TIME:
        # STEP 5
        if TIME >= ereturn[0]:
            j = 0
        elif TIME >= ereturn[1]:
            j = 1
        elif TIME >= ereturn[2]:
            j = 2
        elif TIME >= ereturn[3]:
            j = 3
        else:
            j = -1

        if j != -1 or step8:
            if not step8:
                # STEP 6
                first[j] = i
                #print("Changed first[j] to ", i)
                occup[j] = 0
                for k in range(0, NUMBER_OF_FLOORS):
                    selvec[j][k] = 0
                    flrvec[j][k] = 0

                # STEP 7
                # print("floor[i]", floor[i])
                selvec[j][floor[i]] = 1
                flrvec[j][floor[i]] += 1
                occup[j] += 1
            else:
                step8 = False

            # STEP 8
            i += 1
            between[i] = (lib.expon(MEAN_INTERVAL_TIME)) * 60
            floor[i] = (random.randint(2, NUMBER_OF_FLOORS) - 1)
            delivery[i] = DOOR_HOLD_TIME
            TIME += between[i]
            arrive[i] = TIME
            area_q += (queue * between[i])
            wait[i] = 0

            # STEP 9
            for k in range(0, NUMBER_OF_ELEVATORS):
                if TIME >= ereturn[k]:
                    ereturn[k] = TIME

            # STEP 10
            count = 1
            while (between[i]) <= DOOR_HOLD_TIME and occup[j] < LIFT_CAPACITY:
                for k in range(first[j], i):
                    delivery[k] += between[i]
                # STEP 7
                # print("floor[i]", floor[i])
                count += 1
                selvec[j][floor[i]] = 1
                flrvec[j][floor[i]] += 1
                occup[j] += 1

                # STEP 8
                i += 1
                between[i] = (lib.expon(MEAN_INTERVAL_TIME)) * 60
                floor[i] = (random.randint(2, NUMBER_OF_FLOORS) - 1)
                delivery[i] = DOOR_HOLD_TIME
                TIME += between[i]
                arrive[i] = TIME
                area_q += (queue * between[i])
                wait[i] = 0

                # STEP 9
                for k in range(0, NUMBER_OF_ELEVATORS):
                    if TIME >= ereturn[k]:
                        ereturn[k] = TIME

            limit = i - 1
            load_size[j] += count
            departs[j] += 1
            #print("first[j]", first[j], "j", j)
            #print("i", i)
            # STEP 11
            for k in range(first[j], limit + 1):
                # STEP 12
                N = floor[k] - 1
                val = N * INTER_FLOOR_ARRIVAL_TIME
                s = 0
                for m in range(1, N + 1):
                    s += flrvec[j][m]
                val = val + (DISEMBARK_TIME * s) + DISEMBARK_TIME

                s = 0
                for m in range(1, N + 1):
                    s += selvec[j][m]
                val = val + (DOOR_OPEN_TIME + DOOR_CLOSE_TIME) * s
                val = val + DOOR_OPEN_TIME
                elevator[k] = val
                ELEVTIME += elevator[k]

                # STEP 13
                delivery[k] += elevator[k]
                # STEP 14
                DELTIME += delivery[k]
                # STEP 15
                if delivery[k] > MAXDEL:
                    MAXDEL = delivery[k]
                    #print("maxdel", MAXDEL)
                # STEP 16
                if elevator[k] > MAXELEV:
                    MAXELEV = elevator[k]
                    #print("maxelev", MAXELEV)

            # STEP 17
            s = 0
            for m in range(1, NUMBER_OF_FLOORS):
                s += selvec[j][m]
            stop[j] += s
            mx = 0
            for x in range(NUMBER_OF_FLOORS - 1, 1, -1):
                if selvec[j][x] == 1:
                    mx = x
                    break

            val = 2 * INTER_FLOOR_ARRIVAL_TIME * (mx - 1)
            s = 0
            for m in range(1, NUMBER_OF_FLOORS):
                s += flrvec[j][m]
            val = val + DISEMBARK_TIME * s

            s = 0
            for m in range(1, NUMBER_OF_FLOORS):
                s += selvec[j][m]
            val = val + (DOOR_OPEN_TIME + DOOR_CLOSE_TIME) * s
            eldel[j] = val
            ereturn[j] = TIME + eldel[j]
            operate[j] += eldel[j]

        else:
            if not step20:
                # STEP 19
                quecust = i
                startque = TIME
                queue = 1
                arrive[i] = TIME
            else:
                step20 = False

            j = -1
            while j == -1:
                # STEP 20
                i += 1
                between[i] = (lib.expon(MEAN_INTERVAL_TIME)) * 60
                floor[i] = (random.randint(2, NUMBER_OF_FLOORS) - 1)
                TIME += between[i]
                arrive[i] = TIME
                queue += 1
                area_q += (queue * between[i])
                wait[i] = 0

                # STEP 21
                if TIME >= ereturn[0]:
                    j = 0
                elif TIME >= ereturn[1]:
                    j = 1
                elif TIME >= ereturn[2]:
                    j = 2
                elif TIME >= ereturn[3]:
                    j = 3
                else:
                    j = -1

            # STEP 22
            for k in range(0, NUMBER_OF_FLOORS):
                selvec[j][k] = 0
                flrvec[j][k] = 0
            remain = queue - 12

            # STEP 23
            if remain <= 0:
                R = i
                occup[j] = queue
            else:
                R = quecust + 11
                occup[j] = 12

            # STEP 24
            for k in range(quecust, R + 1):
                selvec[j][floor[k]] = 1
                flrvec[j][floor[k]] += 1

            # STEP 25
            if queue >= QUELEN:
                QUELEN = queue

            # STEP 26
            quetotal += occup[j]
            s = 0
            #print("quecust", quecust, "R", R)
            for m in range(quecust, R + 1):
                s += (TIME - arrive[m])
            QUETIME += s

            # STEP 27
            if (TIME - startque) >= MAXQUE:
                MAXQUE = TIME - startque

            # STEP 28
            first[j] = quecust
            #print("first[j]", first[j], "j", j)
            #print("i", i)
            #print("Changed first[j] to quecust", quecust)

            # STEP 29
            for k in range(first[j], R + 1):
                delivery[k] = (DOOR_HOLD_TIME + TIME - arrive[k])
                wait[k] += (TIME - arrive[k])
                if wait[k] > MAXWAIT:
                    MAXWAIT = wait[k]

            # STEP 30
            if remain <= 0:
                queue = 0
                step8 = True
            else:
                limit = R
                load_size[j] += limit - first[j]
                departs[j] += 1
                for k in range(first[j], limit + 1):
                    # STEP 12
                    N = floor[k] - 1
                    val = N * INTER_FLOOR_ARRIVAL_TIME
                    s = 0
                    for m in range(1, N):
                        s += flrvec[j][m]
                    val = val + (DISEMBARK_TIME * s) + DISEMBARK_TIME

                    s = 0
                    for m in range(1, N):
                        s += selvec[j][m]
                    val = val + (DOOR_OPEN_TIME + DOOR_CLOSE_TIME) * s
                    val = val + DOOR_OPEN_TIME
                    elevator[k] = val
                    ELEVTIME += elevator[k]

                    # STEP 13
                    delivery[k] += elevator[k]
                    # STEP 14
                    DELTIME += delivery[k]
                    # STEP 15
                    if delivery[k] > MAXDEL:
                        MAXDEL = delivery[k]
                    # STEP 16
                    if elevator[k] > MAXELEV:
                        MAXELEV = elevator[k]

                # STEP 17
                s = 0
                for m in range(1, NUMBER_OF_FLOORS):
                    s += selvec[j][m]
                stop[j] += s

                mx = 0
                for x in range(NUMBER_OF_FLOORS - 1, 1, -1):
                    if selvec[j][x] == 1:
                        mx = x
                        break

                val = 2 * INTER_FLOOR_ARRIVAL_TIME * mx
                s = 0
                for m in range(1, NUMBER_OF_FLOORS):
                    s += flrvec[j][m]
                val = val + DISEMBARK_TIME * s

                s = 0
                for m in range(1, NUMBER_OF_FLOORS):
                    s += selvec[j][m]
                val = val + (DOOR_OPEN_TIME + DOOR_CLOSE_TIME) * s
                eldel[j] = val
                ereturn[j] = TIME + eldel[j]
                operate[j] += eldel[j]

            # STEP 31
            queue = remain
            quecust = R
            startque = arrive[R]

            # STEP 32
            step20 = True

    # Output values
    # STEP 33
    N = i - queue
    print("\nTotal Customers served:", N)
    DELTIME = DELTIME / N
    print("Avg Delivery time", DELTIME)
    print("Max Delivery time", MAXDEL)
    print("Max Elev Time", MAXELEV)
    print("Avg Elev Time", ELEVTIME / N)
    print("Longest queue", QUELEN)
    avq = area_q / END_TIME
    #print("Average queue", area_q)
    print("Max wait", MAXWAIT)
    print("Avg wait", MAXWAIT / N)
    # print("Avg time in queue", QUETIME / quetotal)
    print("Longest time in queue", MAXQUE)

    for k in range(0, NUMBER_OF_ELEVATORS):
        print("stop[", k + 1, "]", stop[k])
        print("operate[", k + 1, "]", operate[k] / END_TIME)
        if departs[k] != 0:
            print("load size[", k+1, "]", load_size[k]/departs[k])
        else:
            print("load size[", k + 1, "]", 0)


readInput()
simulate()
