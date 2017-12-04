#!/usr/bin/python
# -*- coding: utf-8 -*

import sys
import math
import random
import copy
input_Maze=[[0,0,0,1,1],
	        [0,0,0,0,0],
	        [1,1,0,1,1],
	        [1,1,0,1,1],
	        [1,0,0,1,1]]

Maze=[]
Player =[]  #플레이어의 위치를 저장하는데 Player[0][1]은 각각 세로, 가로 좌표를 뜻함
Tragectory = [] #플레이어의 위치가 바뀔때마다 여기에 append 할거
#Seepast=[]  #움직인 과정 보고싶어서
MoveDirection = []  #플레이어의 이동방향이 결정될 때마다 여기에 append 할거
#Q_table = []    #플레이어가 각 방향을 선택할 확률을 여기에 저장할 것
Reward = 0  #리워드의 초기값은0
gamma=0.7
Start=[0,0] #시작위치
Before=[]   #플레이어가 직전에 지났던 위치를 여기에 저장
BasicQt=[[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
Rwrd_lst=[]
Mz_lst=[]
MzIndex_lst=[]




def print_info():   #미로판을 출력함

	s = ""

	for x in range(5):
		for y in range (5):
			if ([x,y] == Player):
				s= s+ "P "
				continue
		#	elif ([x,y] == Before):
		#		s= s+ "B "
		#		continue
			elif(Maze[x][y] == 2):
				s= s+ "* "
			elif (Maze[x][y] == 0):
				s = s+ "- "
			else:
				s = s+ "+ "
		s= s+ "\n"

	return s


def Move(Maze, Rwrd_lst, Mz_lst, Player):  #Rwrd_lst에 저장된 확률을 바탕으로 어떤 방향으로 움직일지 결정함
    if(Mz_lst.count(Maze)==0):
        anyway=[0.25,0.5,0.75,1]
        next=-1
        move=random.random()
        for x in range(4):
            if (move < anyway[x]):
                next = x
                break  # 다음 방향을 정함
        return next
    elif(Mz_lst.count(Maze)==1):
        total = 0
        M_Prob = []
        index=Mz_lst.index(Maze)
        for x in range(4):
            value = Rwrd_lst[index][Player[0]][Player[1]][x]
            total = total + value   #가치값을 더해서 총 가치값을 구함

        move_amount = 0.0
        for x in range(4):
            value = Rwrd_lst[index][Player[0]][Player[1]][x]
            move_amount = move_amount + float(value) / float(total)
            M_Prob.append(move_amount)  #가치값을 총 가치값으로 나눠서 확률을 구해서 M_Prob에 추가함

        move = random.random()  #임의의 수를 구해서
        next = -1

        for x in range(4):
            if (move < M_Prob[x]):
                next = x
                break   #다음 방향을 정함

        return next
    else:
        print "Move error!! count is nor 0 and 1"
def Chk_Move(move): #주어진 방향으로 움직이는게 타당한지(즉 벽이거나 장애물이 없는지) 확인한 후 타당하면 1을 반환하고 플레이어의 위치를 그 방향으로 옮기고. 타당하지 않으면 -1을 반환함
    global Before
    blocktest=0
    findway=2*3*5*7
    if(Player[0]==0):
        blocktest+=1
        findway=findway/2
    elif(Maze[Player[0]-1][Player[1]]==1):
        blocktest+=1
        findway =findway/2
    if(Player[0]==4):
        blocktest+=1
        findway =findway/3
    elif(Maze[Player[0]+1][Player[1]]==1):
        blocktest+=1
        findway =findway/3
    if (Player[1] == 0 ):
        blocktest += 1
        findway =findway/5
    elif(Maze[Player[0]][Player[1]-1] == 1):
        blocktest+=1
        findway =findway/5
    if (Player[1] == 4 ):
        blocktest += 1
        findway =findway/7
    elif(Maze[Player[0]][Player[1] + 1] == 1):
        blocktest+=1
        findway =findway/7
        #막힌길이면 blocktest를 1 증가시키고, 해당 값으로 나눔
    if(blocktest==3):   #blocktest값이 3이면 즉 막힌길이 3개면
        if(findway==2): #findway가 2로 안 나눠졌으면 즉 Player[0]-1쪽으로 갈 길이 있으면 현재 위치를 Before에 넣고 Player[0] 좌표를 1 뺌. 옮긴 자리 Maze값을 2로 변경
            if(move==0):
                Before=copy.deepcopy(Player)
                Player[0]+=-1
                Maze[Player[0]][Player[1]]=2
                return 1;
            else:
                return -1;

        elif (findway == 3):
            if(move==1):
                Before = copy.deepcopy(Player)
                Player[0]+= 1
                Maze[Player[0]][Player[1]] = 2
                return 1;
            else:
                return -1;
        elif (findway == 5):
            if(move==2):
                Before = copy.deepcopy(Player)
                Player[1]+=-1
                Maze[Player[0]][Player[1]] = 2
                return 1;
            else:
                return -1;
        elif (findway == 7):
            if(move==3):
                Before = copy.deepcopy(Player)
                Player[1]+=1
                Maze[Player[0]][Player[1]] = 2
                return 1;
            else:
                return -1;

    else: #갈 수 있는 길이 2개 이상이면
        if (move == 0): #Move up
            if (Player[0] == 0):    #젤 위쪽줄이면 못가는 길로 인식
                return -1;
            else:
                tmp = Player[0] - 1;
                if (Maze[tmp][Player[1]] == 1or(Before[0]==tmp and Before[1]==Player[1])):   #위쪽 칸이 장애물이거나 Before자리면 못 가는 길로 인식
                    return -1;
                Before = copy.deepcopy(Player)
                Player[0] = tmp;
                Maze[Player[0]][Player[1]]=2
                return 1;
        elif (move == 1): #Move down
            if (Player[0] == 4):
                return -1;
            else:
                tmp = Player[0] + 1;
                if (Maze[tmp][Player[1]] == 1or(Before[0]==tmp and Before[1]==Player[1])):
                    return -1;
                Before = copy.deepcopy(Player)
                Player[0] = tmp;
                Maze[Player[0]][Player[1]] = 2
                return 1;

        elif (move == 2): #Move <<
            if (Player[1] == 0):
                return -1;
            else:
                tmp = Player[1] - 1;
                if (Maze[Player[0]][tmp] == 1or(Before[0]==Player[0] and Before[1]==tmp)):
                    return -1;
                Before = copy.deepcopy(Player)
                Player[1] = tmp;
                Maze[Player[0]][Player[1]] = 2
                return 1;

        elif (move == 3): #Move >>
            if (Player[1] == 4):
                return -1;
            else:
                tmp = Player[1] + 1;
                if (Maze[Player[0]][tmp] == 1or(Before[0]==Player[0] and Before[1]==tmp)):
                    return -1;
                Before = copy.deepcopy(Player)
                Player[1] = tmp;
                Maze[Player[0]][Player[1]] = 2
                return 1;
        else:
            print ("Error!!!")
            sys.exit()


def Chk_End():
    global Reward
    check=0
    for x in range(5):
        for y in range(5):
            if(Maze[x][y]==0):
                check+=1    #Maze값이 0인게 있으면 check를 1 증가시킴
    if(check==0 and Player==Start):
        return 1;   #check값이 0이면 즉 Maze값이 0인게 하나도 없으면 1을 반환
    else:
        Reward = Reward*0.9
        return -1;  #아니면 리워드 0.7배로 감소시키고 -1을 반환


def Learning(Tragectory, MoveDirection, Rwrd_lst, MzIndex_lst, Reward, gamma):
    chk_count = 0
    while (1):
        [xp, yp] = Tragectory[-1]   #Player의 현재 위치를 xp와 yp에 저장
        direc = MoveDirection[-1]   #Player가 현재 위치로 오기위해 선택했던 방향을 direc에 저장
        index=MzIndex_lst[-1]

        [nx, ny] = [0, 0]
        if (direc == 0):    #Player가 현재 위치로 오기 위해 선택했던 방향이 0이면 플레이어 위치의 [0]값에서 1을 뺀걸 nx에 저장함
            [nx, ny] = [xp - 1, yp]
        elif (direc == 1):
            [nx, ny] = [xp + 1, yp]
        elif (direc == 2):
            [nx, ny] = [xp, yp - 1]
        elif (direc == 3):
            [nx, ny] = [xp, yp + 1]
        else:
            print "Error"

        dot_max = -1
        for x in range(4):
            if (Rwrd_lst[index][nx][ny][x] > dot_max):
                dot_max = Rwrd_lst[index][nx][ny][x]    #nx ny의 위치에서 4가지 방향으로의 Q값중 가장 큰 값을 dot_max에 저장

        Rwrd_lst[index][xp][yp][direc] += (gamma * dot_max - Rwrd_lst[index][xp][yp][direc])    #Player의 현재위치에서 현재위치로 오기위해 선택했던 방향에 대한 Q값을 gamma에 dot_max 값에서 뺀 값을 다시 Q값에 저장함

        if (chk_count == 0):    #첫 단계에서 Q값을 Reward만큼 증가시킴
            Rwrd_lst[index][xp][yp][direc] += Reward
            chk_count += 1

        if (len(MoveDirection) < 2):    #처음 위치까지 다 학습시켰으면 Learning을 끝냄
            break

        Tragectory.pop()    #이미 학습시킨 칸은 날려보냄
        MoveDirection.pop()
        MzIndex_lst.pop()

## Code Start

#Q_table=copy.deepcopy(BasicQt)
Maze=copy.deepcopy(input_Maze)
iteration = 0

while (1):
    Player = copy.deepcopy(Start)
    Before = copy.deepcopy(Start)  # Player, Before's initial value is Start
    Maze[Player[0]][Player[1]]=2
    Reward = 100.
    add = copy.deepcopy(Player) #Player의 현재(=시작) 위치를 add에 저장
    Tragectory.append(add)  #add의 값, 즉 Player의 현재(=시작) 위치를 Tragectory에 저장
    walk=0
    while (1):
        next = Move(Maze, Rwrd_lst, Mz_lst, Player)    #Move에서 선택된 방향을 next에 저장
        add = copy.deepcopy(Maze)
        if (Mz_lst.count(add) == 0):
            Mz_lst.append(add)
            Rwrd_lst.append(BasicQt)
            add2 = len(Mz_lst) - 1
            MzIndex_lst.append(add2)
        if (Mz_lst.count(add) == 1):
            add3 = Mz_lst.index(Maze)
            MzIndex_lst.append(add2)
        chk = Chk_Move(next)    #next에 저장된 방향을 Chk_Move에 넣음으로써 만약 가능한 방향이면 그 쪽으로 Player를 옮기고 Before도 옮기고 1을 반환함. 불가능한 방향이면 안 옮기고 -1을 반환해서 chk에 저장.
        if (chk == -1):
            continue    #불가능한 방향이면 처음으로 돌아가게 함
        walk+=1
        add = copy.deepcopy(Player) #가능한 방향이면 옮겨진 위치를 add에 저장
        Tragectory.append(add)  # 옮겨진 위치를 Tragectory에 추가
        MoveDirection.append(next)  #선택됐던 가능한 방향을 MoveDirection에 추가
        if (iteration % 500 == 1 and iteration>10):
            print print_info()
            print "%d walk is done" % (walk)
            raw_input("")#500번 돌리고 처음 위치에서 한 걸음 산 상태에서 상황 출력
        end = Chk_End()
        if (end == 1):
            #Reward += 50
            break
    Tragectory.pop()  # We does't need last objective goal
    Learning(Tragectory, MoveDirection, Rwrd_lst, MzIndex_lst, Reward, gamma)
    Tragectory = []
    MoveDirection = []
    Maze = copy.deepcopy(input_Maze)
    iteration += 1

    if (iteration % 100 == 0):
        print "Iteration %d done." % (iteration)
