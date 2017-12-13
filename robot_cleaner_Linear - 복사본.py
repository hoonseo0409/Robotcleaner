#!/usr/bin/python
# -*- coding: utf-8 -*

import sys
import math
import random
import copy
import matplotlib.pyplot as plt
Input_Maze=[[0,0,0,1,1,1],
            [0,0,0,0,0,0],
            [1,0,1,1,0,1],
            [1,0,0,0,0,1],
            [1,1,0,1,0,0],
            [1,1,0,0,0,1]]
#Input_Maze: 0은 지나갈 수 있는 길, 1은 벽, 직사각형 형태로 만들 것. 또 고립된 섬을 만들지 말 것.
Xsize=len(Input_Maze)
Ysize=len(Input_Maze[0]) #17/12/11 업데이트 사항 : 임의의 크기의 미로를 입력받을 수 있게 됨

Maze=[] #player의 지나간 궤적을 표시해 놓은 미로
Player =[]  #플레이어의 위치를 저장하는데 Player[0][1]은 각각 세로, 가로 좌표를 뜻함
Tragectory = [] #플레이어의 위치가 바뀔때마다 여기에 append 할거
MoveDirection = []  #플레이어의 이동방향이 결정될 때마다 여기에 append 할거
is_noise_arr=[] #noise로 인한 선택인지 아닌지 여기에 저장함.
Start=[0,0] #시작위치
Before=[]   #플레이어가 직전에 지났던 위치를 여기에 저장
basic_Rwrd=20.  #reward의 시작 값 20정도면 적당한 듯
BasicRwrd=[]

for x in range(Xsize):
    BasicRwrd.append([])
    for y in range(Ysize):
        BasicRwrd[x].append([])
        for s in range(4):
            BasicRwrd[x][y].append(basic_Rwrd)


mini_batch_size=150 #must be integer
print_avg_distance=20   #must be integer
print_plot_distance=601  #must be integer
print_opt_distance=1901 #must be integer
print_curt_distance=2001 #must be integer
punish_count_limit=Xsize+Ysize
Best_Rwrd_lst=[]    #최고로 결과가 좋았던 check point
Rwrd_lst=[] #학습의 핵심인 Reward 저장 배열
Mz_lst=[]   #player가 지나온 궤적이 표시된 미로
MzIndex_lst=[]  #학습 단계에서 필요 : Mz_lst와 Rwrd_lst를 이어주는 역할
#l_rate=0.8 #learning rate : 안 쓰고 stage변수가 그 역할을 대신 함
local_sum=0.    #지역 평균 구할때 사용
local_average=0.    #지역 평균(mini batch 안에서의 평균)
local_avg_lst=[]    #지역 평균 저장용(plot할때 필요)

decay_rate=0.9  #mini batch가 지나갈 때마다 점차 학습 속도를 느리게 할 때 쓰임
momentum=1+1./mini_batch_size #mini batch 안에서 점차 학습 속도를 빠르게 할 때 쓰임, 지금은 이거 안 쓰고 stage=stage*(1+1./200)으로 대체함
dropout_rate=0.8    #이것에 해당하는 비율만큼 학습시킴
stage=0.5   #learning rate를 대신하는 개념
restore_stage=0.2   #3차적으로 stage를 복귀시킬 때 이 값으로 복귀시킴 : second_stage를 고정시키기로 한 이상 이건 쓸모 없어짐
second_stage=0.5    #2차적으로  stage를 복귀시킬 때 이 값으로 복귀시킴
critical_lower_limit=-40.   #이 정도로 치명적으로 결과가 안 좋은 데이터는 학습으로 제외시키는 역할, 즉 이상치를 제거하는 역할이다.
start_noise_limit=0.05  #noise의 시작값
noise_limit=start_noise_limit    #noise_limit must > noise_limit_add noise_limit의 확률로 노이즈가 발생한다
noise_limit_add=0.05    #noise_limit을 업데이트 할 때마다 이 만큼 변화시킨다
is_noise=0  #노이즈에 의한 선택이면 1, 노이즈에 의한 선택이 아니면 0
global_sum=0.   #전역 평균 구할 때 쓰인다
global_average=0.   #전역 평균
min_walk=100    #최고기록 저장해 놓음
best_Trj=[] #최단 경로 저장해 놓음


def print_info():   #미로판을 출력함

   s = ""

   for x in range(Xsize):
      for y in range (Ysize):
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
            s = s+ "@ "
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
    elif (noise < noise_limit):
        is_noise = 1    #노이즈로 인한 선택임을 나타냄
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
            #value = 1/(1+math.exp(-0.05*Rwrd_lst[index][Player[0]][Player[1]][x])) #sigmoid 방식
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
    if(Player[0]==Xsize-1):
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
    if (Player[1] == Ysize-1 ):
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
            if (Player[0] == Xsize-1):
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
            if (Player[1] == Ysize-1):
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
    for x in range(Xsize):
        for y in range(Ysize):
            if(Maze[x][y]==0):
                check+=1    #Maze값이 0인게 있으면 check를 1 증가시킴
    if(check==0 and Player==Start):
        return 1;   #check값이 0이면 즉 Maze값이 0인게 하나도 없으면 1을 반환
    else:
        return -1;



def Learning(Tragectory, MoveDirection, Rwrd_lst, MzIndex_lst, local_average, walk, is_noise_arr):
    global stage
    gamma_rate=1.
    punish_count = 0    #결과가 나빴을 때 뒤에서부터 이 count 갯수 만큼의 해당하는 만큼의 선택들은 처벌하지 아니한다.

    noise_count=float(is_noise_arr.count(1))    #noise 갯수를 구한다. 나중에 Rwrd로 인한 선택들의 처벌을 감경시킬 때 이걸로 나눠준다.
    stage=stage*momentum #momentum 역할을 한다. mini batch 시작일 때 1이었다가 mini batch 끝날때 2가 되도록 선형적으로 증가.
    if(noise_count==0.):    #noise가 하나도 없는데 결과 나쁘면 0으로 나누기 에러 뜨므로 1을 넣어줌
        noise_count=1.
    while (1):

        [xp, yp] = Tragectory[-1]   #Player의 현재 위치를 xp와 yp에 저장
        direc = MoveDirection[-1]   #Player가 현재 위치에서 선택한 방향을 direc에 저장
        iss_noise=is_noise_arr[-1]  #noise로 인한 선택1인지 아닌지0를 iss_noise에 저장
        index=MzIndex_lst[-2]   #Player의 현재 위치에서의 맵 번호

        if(stage*(gamma_rate*(float((global_average+local_average)/2-walk)/global_average)<critical_lower_limit)):  #결과가 너무 나쁜 데이터는 학습 안 시킴 : 이상치 제거 목적
            break

        rand=random.random()
        punish_count += 1

        if(dropout_rate<rand or ((punish_count<punish_count_limit)and((global_average+local_average)/2<walk))  ):   #drop out에 걸러지거나 결과가 나빴을 때 12걸음은 처벌 안 함.
            if (len(MoveDirection) < 2):  # 처음 위치까지 다 학습시켰으면 Learning을 끝냄
                break
            Tragectory.pop()  # 이미 학습시킨 칸은 날려보냄
            MoveDirection.pop()
            is_noise_arr.pop()
            MzIndex_lst.pop()
            continue


        sum=0.
        for i in range(4):  #reward들의 합이 너무 작으면 변화에 너무 민감하고, 너무 크면 변화에 너무 둔감하므로 2배로 곱하거나 반으로 나눠줌.
            sum+=Rwrd_lst[index][xp][yp][i]
        if(sum<basic_Rwrd*2):
            for i in range(4):
                Rwrd_lst[index][xp][yp][i]=Rwrd_lst[index][xp][yp][i]*2.
        if(sum>basic_Rwrd*8):
            for i in range(4):
                Rwrd_lst[index][xp][yp][i]=Rwrd_lst[index][xp][yp][i]/2.
        if(iss_noise==0 and (global_average+local_average)/2<walk): #결과가 나빴고 noise로 인한 선택이 아니면
            if (Rwrd_lst[index][xp][yp][direc] + (1./noise_count)*stage * (
                gamma_rate * (float((global_average + local_average) / 2 - walk) / global_average)) > 0):  # 업데이트 해서 음수가 되지 않는다면
                Rwrd_lst[index][xp][yp][direc] = Rwrd_lst[index][xp][yp][direc] + (1./noise_count)*stage * (
                gamma_rate * (float((global_average + local_average) / 2 - walk) / global_average)) #reward 업데이트 시켜 학습시킴

        else:   #결과가 좋았거나 noise로 인한 선택이면
            if(Rwrd_lst[index][xp][yp][direc] + stage*(gamma_rate*(float((global_average+local_average)/2-walk)/global_average))>0):    #업데이트 해서 음수가 되지 않는다면
                Rwrd_lst[index][xp][yp][direc] = Rwrd_lst[index][xp][yp][direc] + stage*(gamma_rate*(float((global_average+local_average)/2-walk)/global_average))
                # reward 업데이트 시켜 학습시킴

        #gamma_rate=gamma_rate-(1./walk)
        #gamma_rate=gamma_rate*(float(walk-1)/walk) 마지막 선택에 가까울 수록 보다 많이 업데이트 시켜 줌. 일단 폐지함


        if(walk<local_average): #결과가 괜찮았는데
            max_r=max(Rwrd_lst[index][xp][yp])
            index_max=Rwrd_lst[index][xp][yp].index(max_r)
            if(index_max!=direc):   #선택했던 방향이 Reward가 최대인 방향이 아니면
                Rwrd_lst[index][xp][yp][direc]+=(Rwrd_lst[index][xp][yp][index_max]-Rwrd_lst[index][xp][yp][direc])*0.001
                #최대 Reward값을 0.1%정도 따라잡도록 하는 코드



        if (len(MoveDirection) < 2):    #처음 위치까지 다 학습시켰으면 Learning을 끝냄
            break

        Tragectory.pop()    #이미 학습시킨 칸은 날려보냄
        MoveDirection.pop()
        is_noise_arr.pop()
        MzIndex_lst.pop()


def PreTraining_BasicRwrd(Input_Maze, BasicRwrd):    #input maze에 적힌 정보를 바탕으로 못 가는 방향(reward=0)을 BasicRewrd에 반영시킴. 이렇게 하면 pre processing 효과를 얻어 초기 학습 속도가 대폭 상승함
    for x in range(Xsize):
        for y in range(Ysize):
            if (x == 0):
                BasicRwrd[0][y][0] = 0.
            elif (x == Xsize-1):
                BasicRwrd[Xsize-1][y][1] = 0.
            if (y == 0):
                BasicRwrd[x][0][2] = 0.
            elif (y == Ysize-1):
                BasicRwrd[x][Ysize-1][3] = 0.
            if (Input_Maze[x][y] == 1):
                if (x != Xsize-1):
                    BasicRwrd[x + 1][y][0] = 0.
                if (x != 0):
                    BasicRwrd[x - 1][y][1] = 0.
                if (y != Ysize-1):
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
Best_Rwrd_lst.append(BasicRwrd)
PreTraining_BasicRwrd(Input_Maze, BasicRwrd)
MzIndex_lst=[0]
before_avg=0.   #지난번 mini batch의 평균 걸음수를 저장해 둠

while (1):
    Player = copy.deepcopy(Start)
    Before = copy.deepcopy(Start)  # Player, Before's initial value is Start
    Maze[Start[0]][Start[1]] = 2
    #Reward = 0.
    add = copy.deepcopy(Player) #Player의 현재(=시작) 위치를 add에 저장
    Tragectory.append(add)  #add의 값, 즉 Player의 현재(=시작) 위치를 Tragectory에 저장
    walk=0
    #print Rwrd_lst
    if (iteration % mini_batch_size == 1 and iteration > 1):    #mini batch 개념으로 mini_batch_size 단위로 진행사항을 평가한 뒤 noise와 같은 parameter들을 변화시켜 줌
        if(before_avg<local_average):   #상황이 안 좋아지고 있으면 즉 walk가 증가 추세에 있으면
            local_sum=0.
            local_iteration=0.
            #stage = stage * decay_rate
            stage = second_stage    #momentum의 초기화
            if(noise_limit+noise_limit_add>0.2):
                Rwrd_lst=copy.deepcopy(Best_Rwrd_lst)   #안전장치: noise가 너무 커지다보면 안 좋은 방향으로 무한히 나아갈 수 있으므로 noise가 0.4를 넘을 정도로 계속 안 좋아졌다면 Best_Rwrd_lst에 저장해 두었던 좋았던 check point로 복귀함.
                noise_limit = noise_limit - noise_limit_add
            else:
                noise_limit=noise_limit+noise_limit_add
            #stage = stage * momentum
        elif(before_avg>local_average): #상황이 좋아지고 있으면
            #stage = stage * (1.+200./2000.)
            #stage = stage /decay_rate
            stage = second_stage
            Best_Rwrd_lst=copy.deepcopy(Rwrd_lst)
            if(noise_limit-noise_limit_add<0):
                noise_limit = start_noise_limit #noise가 너무 작아져서 업데이트시 음수가 된다면 초기 noise로 복귀시킴
            else:
                noise_limit=noise_limit-noise_limit_add



        before_avg=local_average
    if (iteration % print_avg_distance == 1 and iteration>1 ):
        print "global_average= %f, local_average=%f,iteration=%d"%(global_average,local_average,iteration)
        #print Rwrd_lst
        local_avg_lst.append(global_average)

    if (iteration % print_plot_distance == 1 and iteration > 1):
        plt.plot(local_avg_lst)
        plt.show()


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
        """if (iteration % 2001 == 1 and iteration > 1):
            stage = second_stage"""

            #second_stage=second_stage*decay_rate

        if (iteration % print_curt_distance == 1 and iteration>1):
            "This is the current path."
            print print_info()
            #print "%d walked" % (walk)
            #second_stage=restore_stage
            #print local_average
            raw_input("")

            

        end = Chk_End()
        if (end == 1):
            if(min_walk>=walk):
                min_walk=walk
                best_Trj=copy.deepcopy(Tragectory)

            iteration += 1
            local_iteration+=1
            global_sum+=walk
            local_sum+=walk
            local_average = local_sum / local_iteration
            global_average=global_sum/iteration
            #Reward+=(local_average/walk)
            break
    if (iteration % print_opt_distance == 1 and iteration > 1):
        print ("This is the shortest path to the present.")
        tmp=[]
        tmp=copy.deepcopy(best_Trj)
        while(1):
            if(len(tmp)==0):
                break
            Player=copy.deepcopy(tmp.pop(0))
            print print_info()
            raw_input("")
    Tragectory.pop()  # We does't need last objective goal
    Learning(Tragectory, MoveDirection, Rwrd_lst, MzIndex_lst, local_average, walk, is_noise_arr)
    Tragectory = []
    MoveDirection = []
    is_noise_arr=[]
    MzIndex_lst=[0]
    Maze = copy.deepcopy(Input_Maze)
