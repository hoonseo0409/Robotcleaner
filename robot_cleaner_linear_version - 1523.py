#!/usr/bin/python
# -*- coding: utf-8 -*

import sys
import math
import random
import copy
Input_Maze=[[0,0,0,1,1],
            [0,0,0,0,0],
            [1,0,1,0,1],
            [1,0,0,0,1],
            [1,1,1,0,1]]

Maze=[]
Player =[]  #플레이어의 위치를 저장하는데 Player[0][1]은 각각 세로, 가로 좌표를 뜻함
Tragectory = [] #플레이어의 위치가 바뀔때마다 여기에 append 할거

MoveDirection = []  #플레이어의 이동방향이 결정될 때마다 여기에 append 할거
is_noise_arr=[]
#Q_table = []    #플레이어가 각 방향을 선택할 확률을 여기에 저장할 것
#Reward = 0  #리워드의 초기값은0
#gamma=0.9
Start=[0,0] #시작위치
Before=[]   #플레이어가 직전에 지났던 위치를 여기에 저장
basic_Rwrd=20.
BasicRwrd=[[[basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd]],
 [[basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd]],
 [[basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd]],
 [[basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd]],
 [[basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd], [basic_Rwrd, basic_Rwrd, basic_Rwrd, basic_Rwrd]]]
"""BasicWalk=[[[0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.]],
 [[0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.]],
 [[0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.]],
 [[0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.]],
 [[0.,0.], [0.,0.], [0.,0.], [0.,0.], [0.,0.]]]"""


#BasicRwrd=[[[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]]
Rwrd_lst=[]
Mz_lst=[]
MzIndex_lst=[]
#l_rate=0.8
local_sum=0.
local_average=0.
#rate=0.9
#walk_lst=[]
#WalkOfAction_lst=[]

decay_rate=0.9
momentum=1.
dropout_rate=1.
stage=0.2
restore_stage=0.2
second_stage=0.2
critical_lower_limit=-2.
start_noise_limit=0.05
noise_limit=start_noise_limit    #noise_limit must > noise_limit_add
noise_limit_add=0.04
is_noise=0
global_sum=0.
global_average=0.


def print_info():   #미로판을 출력함

   s = ""

   for x in range(5):
      for y in range (5):
         if ([x,y] == Player):
            s= s+ "P "
            continue
      #   elif ([x,y] == Before):
      #      s= s+ "B "
      #      continue
         elif(Maze[x][y] == 2):
            s= s+ "* "
         elif (Maze[x][y] == 0):
            s = s+ "- "
         else:
            s = s+ "+ "
      s= s+ "\n"

   return s


def Move(Maze, Rwrd_lst, Mz_lst, Player):  #Rwrd_lst에 저장된 확률을 바탕으로 어떤 방향으로 움직일지 결정함
    noise=random.random()
    global is_noise
    if(Mz_lst.count(Maze)==0):    #If The trajectory is not experienced or at probability of 5%, It walks randomly.
        is_noise = 0
        anyway=[0.25,0.5,0.75,1]
        next=-1
        move=random.random()
        for x in range(4):
            if (move < anyway[x]):
                next = x
                break  # 다음 방향을 정함
        return next
    if (noise < noise_limit):
        is_noise = 1
        anyway = [0.25, 0.5, 0.75, 1]
        next = -1
        move = random.random()
        for x in range(4):
            if (move < anyway[x]):
                next = x
                break  # 다음 방향을 정함
        return next
    elif(Mz_lst.count(Maze)==1):    #If you have previously experienced the trajectory
        is_noise=0
        total = 0
        M_Prob = []
        index=Mz_lst.index(Maze)
        for x in range(4):
            #value = 1/(1+math.exp(-0.05*Rwrd_lst[index][Player[0]][Player[1]][x]))
            value = Rwrd_lst[index][Player[0]][Player[1]][x]
            total = total + value   #가치값을 더해서 총 가치값을 구함

        move_amount = 0.0
        for x in range(4):
            #value = 1/(1+math.exp(-0.05*Rwrd_lst[index][Player[0]][Player[1]][x]))
            value=Rwrd_lst[index][Player[0]][Player[1]][x]
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
        #Reward = Reward
        return -1;



def Learning(Tragectory, MoveDirection, Rwrd_lst, MzIndex_lst, local_average, walk, is_noise_arr):

    gamma_rate=1.
    punish_count = 0

    noise_count=float(is_noise_arr.count(1))
    if(noise_count==0.):
        noise_count=1.
    while (1):

        [xp, yp] = Tragectory[-1]   #Player의 현재 위치를 xp와 yp에 저장
        direc = MoveDirection[-1]   #Player가 현재 위치에서 선택한 방향을 direc에 저장
        iss_noise=is_noise_arr[-1]
        index=MzIndex_lst[-2]   #Player의 현재 위치에서의 맵 번호

        if(stage*(gamma_rate*(float((global_average+local_average)/2-walk)/global_average)<critical_lower_limit)):
            #print "critical!!"
            break

        rand=random.random()
        punish_count += 1

        if(dropout_rate<rand or ((punish_count<12)and((global_average+local_average)/2<walk))  ):
            if (len(MoveDirection) < 2):  # 처음 위치까지 다 학습시켰으면 Learning을 끝냄
                break
            Tragectory.pop()  # 이미 학습시킨 칸은 날려보냄
            MoveDirection.pop()
            is_noise_arr.pop()
            MzIndex_lst.pop()
            continue


        sum=0.
        for i in range(4):
            sum+=Rwrd_lst[index][xp][yp][i]
        if(sum<40):
            for i in range(4):
                Rwrd_lst[index][xp][yp][i]=Rwrd_lst[index][xp][yp][i]*2.
        if(sum>160):
            for i in range(4):
                Rwrd_lst[index][xp][yp][i]=Rwrd_lst[index][xp][yp][i]/2.
        if(iss_noise==0 and (global_average+local_average)/2<walk):
            if (Rwrd_lst[index][xp][yp][direc] + (1./noise_count)*stage * (
                gamma_rate * (float((global_average + local_average) / 2 - walk) / global_average)) > 0):  # 업데이트 해서 음수가 되지 않는다면
                Rwrd_lst[index][xp][yp][direc] = Rwrd_lst[index][xp][yp][direc] + (1./noise_count)*stage * (
                gamma_rate * (float((global_average + local_average) / 2 - walk) / global_average))

        else:
            if(Rwrd_lst[index][xp][yp][direc] + stage*(gamma_rate*(float((global_average+local_average)/2-walk)/global_average))>0):    #업데이트 해서 음수가 되지 않는다면
                Rwrd_lst[index][xp][yp][direc] = Rwrd_lst[index][xp][yp][direc] + stage*(gamma_rate*(float((global_average+local_average)/2-walk)/global_average))

        #gamma_rate=gamma_rate-(1./walk)
        gamma_rate=gamma_rate*(float(walk-1)/walk)
#Q값에 (local_average-walk)/local_average 를 더해 줌으로써 평균 걸음수보다 더 많이 걷는 선택을 하게 되면 Q값을 깎아서 그 쪽으로 갈 확률을 감소시킴, 반대의 경우 증가시킴.
        """if (Rwrd_lst[index][xp][yp][direc] <= 0.):
            Rwrd_lst[index][xp][yp][direc] = 1."""

#만약 Q값이 음수가 되게 되면 오류가 발생하므로 음수가 될 경우 Q값을 1로 초기화시킴
        #number+=1
        if(walk<local_average):
            max_r=max(Rwrd_lst[index][xp][yp])
            index_max=Rwrd_lst[index][xp][yp].index(max_r)
            if(index_max!=direc):
                Rwrd_lst[index][xp][yp][direc]+=(Rwrd_lst[index][xp][yp][index_max]-Rwrd_lst[index][xp][yp][direc])*0.001
                #Q값이 작은 방향으로 한번 가 봤는데, walk가 줄어들었으면 최대 Q값을 0.1%정도 따라잡도록 하는 코드



        if (len(MoveDirection) < 2):    #처음 위치까지 다 학습시켰으면 Learning을 끝냄
            break

        Tragectory.pop()    #이미 학습시킨 칸은 날려보냄
        MoveDirection.pop()
        is_noise_arr.pop()
        MzIndex_lst.pop()
        #Remainwalk_lst.pop()

def Initialize_BasicRwrd(Input_Maze, BasicRwrd):
    for x in range(5):
        for y in range(5):
            if (x == 0):
                BasicRwrd[0][y][0] = 0.
            elif (x == 4):
                BasicRwrd[4][y][1] = 0.
            if (y == 0):
                BasicRwrd[x][0][2] = 0.
            elif (y == 4):
                BasicRwrd[x][4][3] = 0.
            if (Input_Maze[x][y] == 1):
                if (x != 4):
                    BasicRwrd[x + 1][y][0] = 0.
                if (x != 0):
                    BasicRwrd[x - 1][y][1] = 0.
                if (y != 4):
                    BasicRwrd[x][y + 1][2] = 0.
                if (y != 0):
                    BasicRwrd[x][y - 1][3] = 0.


## Code Start


Maze=copy.deepcopy(Input_Maze)
iteration = 0
local_iteration =0
Maze[Start[0]][Start[1]]=2
add = copy.deepcopy(Maze)
Mz_lst.append(add)
Rwrd_lst.append(BasicRwrd)
#WalkOfAction_lst.append(BasicWalk)
Initialize_BasicRwrd(Input_Maze, BasicRwrd)
MzIndex_lst=[0]
#stage=0
before_avg=0.

while (1):
    """if(iteration%1300==1 and iteration>1):
        stage+=1"""
    Player = copy.deepcopy(Start)
    Before = copy.deepcopy(Start)  # Player, Before's initial value is Start
    Maze[Start[0]][Start[1]] = 2
    #Reward = 0.
    add = copy.deepcopy(Player) #Player의 현재(=시작) 위치를 add에 저장
    Tragectory.append(add)  #add의 값, 즉 Player의 현재(=시작) 위치를 Tragectory에 저장
    walk=0
    #print Rwrd_lst
    if (iteration % 200 == 1 and iteration > 1):
        if(before_avg<local_average):
            local_sum=0.
            local_iteration=0.
            #stage = stage * decay_rate
            if(noise_limit+noise_limit_add>0.35):
                pass
            else:
                noise_limit=noise_limit+noise_limit_add
            #stage = stage / decay_rate
        elif(before_avg>local_average):
            #stage = stage * (1.+200./2000.)
            stage = stage /decay_rate
            if(noise_limit-noise_limit_add<0):
                noise_limit = start_noise_limit
            else:
                noise_limit=noise_limit-noise_limit_add



        before_avg=local_average
    if (iteration % 50 == 1 and iteration>1 ):
        print "global_average= %f, local_average=%f,iteration=%d"%(global_average,local_average,iteration)
        print Rwrd_lst




    while (1):
        next = Move(Maze, Rwrd_lst, Mz_lst, Player)    #Move에서 선택된 방향을 next에 저장
        chk = Chk_Move(next)    #next에 저장된 방향을 Chk_Move에 넣음으로써 만약 가능한 방향이면 그 쪽으로 Player를 옮기고 Before도 옮기고 1을 반환함. 불가능한 방향이면 안 옮기고 -1을 반환해서 chk에 저장.
        if (chk == -1):
            continue    #불가능한 방향이면 처음으로 돌아가게 함
        walk+=1
        add = copy.deepcopy(Maze)
        if (Mz_lst.count(add) == 0):
            Mz_lst.append(add)
            Rwrd_lst.append(BasicRwrd)
            #WalkOfAction_lst.append(BasicWalk)
            add2 = len(Mz_lst) - 1
            MzIndex_lst.append(add2)
        elif (Mz_lst.count(add) == 1):
            add2 = Mz_lst.index(add)
            MzIndex_lst.append(add2)
        else:
            print "That Maze count is not 1 nor 0"
        add = copy.deepcopy(Player) #가능한 방향이면 옮겨진 위치를 add에 저장
        Tragectory.append(add)  # 옮겨진 위치를 Tragectory에 추가
        MoveDirection.append(next)  #선택됐던 가능한 방향을 MoveDirection에 추가
        is_noise_arr.append(is_noise)
        if (iteration % 2001 == 1 and iteration > 1):
            stage = second_stage

            #second_stage=second_stage*decay_rate

        if (iteration % 1000000 == 1 and iteration>1):
            print print_info()
            #print "%d walked" % (walk)
            #second_stage=restore_stage
            #print local_average
            raw_input("")

            

        end = Chk_End()
        if (end == 1):
            iteration += 1
            local_iteration+=1
            global_sum+=walk
            local_sum+=walk
            local_average = local_sum / local_iteration
            global_average=global_sum/iteration
            #Reward+=(local_average/walk)
            break
    Tragectory.pop()  # We does't need last objective goal
    Learning(Tragectory, MoveDirection, Rwrd_lst, MzIndex_lst, local_average, walk, is_noise_arr)
    Tragectory = []
    MoveDirection = []
    is_noise_arr=[]
    MzIndex_lst=[0]
    Maze = copy.deepcopy(Input_Maze)
    """if (iteration % 500 == 0):
        print "Iteration %d done." % (iteration)"""