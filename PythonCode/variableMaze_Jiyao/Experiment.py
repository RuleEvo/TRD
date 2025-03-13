import argparse
import os
import numpy as np
import math
import random
import torchvision.transforms as transforms
from torchvision.utils import save_image
from agent import Agent
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import pandas as pd  # 确保在文件顶部导入 pandas
import winsound
import seaborn as sns
import copy

import matplotlib.pyplot as plt

import torch.nn as nn
import torch.nn.functional as F
import torch
from pathlib import Path

from mini_env import MiniEnv
from Q_brain import QLearningAgent
from DQN_brain import DQNAgent
from warnings import simplefilter

import pandas as pd
import json, sys

simplefilter(action="ignore",category=FutureWarning)
simplefilter(action="ignore",category=UserWarning)

os.makedirs("agents", exist_ok=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()

# Rule
parser.add_argument("--random_count", type=int, default=0, help="0, number of Random Action agents")
parser.add_argument("--cheater_count", type=int, default=0, help="1, number of Always Cheat agents")
parser.add_argument("--cooperator_count", type=int, default=0, help="2, number of Always Cooperate agents")
parser.add_argument("--copycat_count", type=int, default=0, help="3, number of Copycat agents")
parser.add_argument("--grudger_count", type=int, default=0, help="4, number of Grudger agents")
parser.add_argument("--detective_count", type=int, default=1, help="5, number of Detective agents")
parser.add_argument("--ai_count", type=int, default=1, help="6, number of AI agents")
parser.add_argument("--human_count", type=int, default=0, help="7, number of human agents")

parser.add_argument("--trade_rules", type=int, nargs=6, default=[0, 0, 3, -1, 2, 2], help="8 to 13, Trade rules as a list")
parser.add_argument("--round_number", type=int, default=1, help="14, The round number of a competition")
parser.add_argument("--reproduction_number", type=int, default=0, help="15, The reproduction number of each round")
parser.add_argument("--mistake_possibility", type=float, default=0.01, help="16, the possibility to take opposite action")

parser.add_argument("--fixed_rule", type=str2bool, default = True, help="Use some fixed rule for agent training and testing")


# Strategy
parser.add_argument("--humanPlayer", type=str2bool, default=False, help="True means there is human player")
parser.add_argument("--ai_type", type=str, default='Q', help="type of AI agent (e.g., 'Q', 'DQN')")

# Evaluation
parser.add_argument("--cooperationRate", type=float, default= 1, help="cooperation rate")
parser.add_argument("--individualIncome", type=float, default= 2, help="individual income")
parser.add_argument("--giniCoefficient", type=float, default= 0.50, help="Gini Coefficient")

# Designer and Evaluator
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.01, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--RuleDimension", type=int, default= 17, help="trade rule dimension")
parser.add_argument("--DE_train_episode", type=int, default=1, help="number of episode during Q table training")
parser.add_argument("--DE_test_episode", type=int, default=1, help="number of episode during test")
parser.add_argument("--layersNum", type=int, default=1, help="layer number of the generator output")
parser.add_argument("--evaluationSize", type=int, default=1, help="size of the evaluation metrics")

# Agent training
parser.add_argument("--agent_train_epoch", type=int, default=1, help="number of epoch of training agent")
parser.add_argument("--gamma", type=float, default=0.99, help="discount factor for rewards")
parser.add_argument("--epsilon", type=float, default=1.0, help="initial epsilon for epsilon-greedy")
parser.add_argument("--epsilon_decay", type=float, default=0.999, help="epsilon decay rate")
parser.add_argument("--epsilon_min", type=float, default=0.1, help="minimum epsilon")
parser.add_argument("--memory_size", type=int, default=10000, help="memory size for experience replay")
parser.add_argument("--target_update", type=int, default=10, help="how often to update the target network")
parser.add_argument("--state_size", type=int, default=20, help="size of the state vector")

# Other
# parser.add_argument("--publish", type=bool, default=True, help="True means publish data on web")

actionlist = ["cheat", "cooperation"]

publish = True
opt = parser.parse_args()

cuda = True if torch.cuda.is_available() else False

# 在模块顶层定义全局变量，用于判断是否已经初始化 Excel 文件
excel_initialized = False
first_save = False


def publish_excel_update(update_data,
                         excel_path="C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/dataupdates.xlsx"):
    global excel_initialized

    # 第一次调用时，删除已有的 Excel 文件（或清空）
    if not excel_initialized:
        if os.path.exists(excel_path):
            os.remove(excel_path)
        excel_initialized = True

    # 如果文件存在，则读取已有数据并追加，否则直接创建新 DataFrame
    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path)
        df_new = pd.DataFrame([update_data])
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = pd.DataFrame([update_data])

    df_combined.to_excel(excel_path, index=False)
    print("UPDATE_EXCEL:" + json.dumps({"excel_updated": True}))
    sys.stdout.flush()


def plot_q_table(q_table, output_path="q_table_heatmap.png"):
    """
    绘制 Q-table 的热图，并保存为图像文件。
    数据根据 observation 前缀（格式如 "0_..."）分为 8 组，
    分别对应 8 种性格（前缀 0 到 7），在一张图中自上而下绘制 8 个热力图。

    参数：
      q_table: pandas DataFrame，包含至少三列：'cheat', 'cooperation' 和 'observation'。
      output_path: 图像保存的路径，默认保存为 "q_table_heatmap.png"。
    """
    # 如果没有 'observation' 列，则使用索引作为 observation
    if "observation" not in q_table.columns:
        q_table = q_table.copy()
        q_table["observation"] = q_table.index.astype(str)

    # 按照 observation 排序
    df_sorted = q_table.sort_values(by="observation").copy()

    # 提取前缀，假设 observation 格式为 "0_XXXXXXXX", "1_XXXXXX" 等
    def get_prefix(obs):
        try:
            return obs.split('_')[0]
        except Exception:
            return None

    df_sorted['prefix'] = df_sorted["observation"].apply(get_prefix)

    # 定义前缀对应的性格映射
    personality_map = {
        "0": "Random",
        "1": "Cheater",
        "2": "Cooperator",
        "3": "Copycat",
        "4": "Grudger",
        "5": "Detective",
        "6": "AI",
        "7": "Human"
    }

    # 建立 8 个子图（nrows=8）
    fig, axes = plt.subplots(nrows=8, figsize=(12, 4 * 8))

    for i in range(8):
        prefix_str = str(i)
        group = df_sorted[df_sorted['prefix'] == prefix_str]
        ax = axes[i]
        if group.empty:
            # 若无数据，显示性格名称及提示 "No Data"，关闭坐标轴
            ax.set_title(f"{personality_map.get(prefix_str, prefix_str)}: No Data")
            ax.axis('off')
        else:
            # 对该组数据归一化 cheat 和 cooperation 列
            combined_min = min(group["cheat"].min(), group["cooperation"].min())
            combined_max = max(group["cheat"].max(), group["cooperation"].max())
            group = group.copy()
            group["cheat_norm"] = (group["cheat"] - combined_min) / (combined_max - combined_min)
            group["cooperation_norm"] = (group["cooperation"] - combined_min) / (combined_max - combined_min)
            # 绘制热图
            data = group[["cheat_norm", "cooperation_norm"]].T
            sns.heatmap(data, cmap="coolwarm", annot=False,
                        xticklabels=group["observation"], yticklabels=["cheat", "cooperation"],
                        ax=ax)
            ax.set_title(f"{personality_map.get(prefix_str, prefix_str)} (n={len(group)})")
            ax.tick_params(axis='x', rotation=90, labelsize=8)
    plt.tight_layout()

    # 如果目标文件已存在则删除
    if os.path.exists(output_path):
        os.remove(output_path)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_test_results_to_excel(test_result, epoch,
                               excel_path="C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/test_results.xlsx"):
    """
    将测试结果保存到 Excel 文件中。
    格式：
        epoch, gini_coefficient, cooperation_rate, individual_income
    其中 epoch 为数字，其它3个字段为 9 维列表（以字符串形式保存）。

    如果是第一次调用（即文件存在时），则删除文件后重写；后续调用则追加记录。

    参数：
        test_result: 字典，包含 'gini_coefficient', 'cooperation_rate', 'individual_income'
                     每个键对应一个长度为9的列表。
        epoch: 数字，表示当前测试的 epoch 数量
        excel_path: Excel 文件保存路径
    """
    global first_save

    # 将 9 维列表转换为字符串保存
    row = {
        "epoch": epoch,
        "gini_coefficient": str(test_result['gini_coefficient']),
        "cooperation_rate": str(test_result['cooperation_rate']),
        "individual_income": str(test_result['individual_income'])
    }
    df_new = pd.DataFrame([row])

    # 第一次调用时，如果文件已存在则删除文件
    if not first_save:
        if os.path.exists(excel_path):
            try:
                os.remove(excel_path)
                print(f"已删除旧文件：{excel_path}")
            except Exception as e:
                print("删除旧Excel文件时出错：", e)
        # 保存新文件
        df_new.to_excel(excel_path, index=False)
        first_save = True
    else:
        # 后续调用时追加记录
        try:
            if os.path.exists(excel_path):
                df_existing = pd.read_excel(excel_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_excel(excel_path, index=False)
            else:
                df_new.to_excel(excel_path, index=False)
        except Exception as e:
            print("保存Excel数据时出错：", e)
    print(f"测试结果已保存到 {excel_path}")
# def save_test_results_to_excel(test_result, epoch,
#                                excel_path="C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/test_results.xlsx"):
#     """
#     将测试结果保存到 Excel 文件中。
#     格式：
#         epoch, gini_coefficient, cooperation_rate, individual_income
#     其中 epoch 为数字，其它3个字段为 9 维列表（以字符串形式保存）。
#
#     如果文件存在，则追加记录；否则新建文件。
#
#     参数：
#         test_result: 字典，包含 'gini_coefficient', 'cooperation_rate', 'individual_income'
#                      每个键对应一个长度为9的列表。
#         epoch: 数字，表示当前测试的 epoch 数量
#         excel_path: Excel 文件保存路径
#     """
#     # 将 9 维列表转换为字符串保存
#
#     row = {
#         "epoch": epoch,
#         "gini_coefficient": str(test_result['gini_coefficient']),
#         "cooperation_rate": str(test_result['cooperation_rate']),
#         "individual_income": str(test_result['individual_income'])
#     }
#     df_new = pd.DataFrame([row])
#
#     if os.path.exists(excel_path):
#         try:
#             df_existing = pd.read_excel(excel_path)
#             df_combined = pd.concat([df_existing, df_new], ignore_index=True)
#             df_combined.to_excel(excel_path, index=False)
#         except Exception as e:
#             print("保存Excel数据时出错：", e)
#     else:
#         df_new.to_excel(excel_path, index=False)
#     print(f"测试结果已保存到 {excel_path}")




class RuleDesigner(nn.Module):
    def __init__(self):
        super(RuleDesigner, self).__init__()

        def block(in_feat, out_feat, normalize=False):  # the batchsize is 1, so we don't need batch normalize.
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(opt.evaluationSize, 8, normalize=False),   # input is the persentage of cooperation prediction
            *block(8, 16),    # the second layer 8 to 16 dimension
            nn.Linear(16, int(np.prod((opt.layersNum, opt.RuleDimension)))),
            nn.Tanh(),
            nn.Softmax()
        )

    def forward(self, z):
        output = self.model(z)
        output = output.view(output.size(0), int(np.prod((opt.layersNum, opt.RuleDimension))))
        return output

class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod((opt.layersNum, opt.RuleDimension))), 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity

class Environment(nn.Module):
    def __init__(self):
        super(Environment, self).__init__()

        trade_rules = [0, 0, 3, -1, 2, 2]
        round_number = 1
        reproduction_number = 5
        mistake_possibility = 0.05
        beta = [0, 0]

        self.shared_q_table = pd.DataFrame(columns=actionlist, dtype=np.float64)

        self.env = MiniEnv(trade_rules, round_number, reproduction_number, mistake_possibility, beta)  # [-3, -3, 0, 2, 5, 5]  [0,0,3,-1,2,2 ]
        self.agents = []

    def reset_shared_q_table(self):
        self.shared_q_table = pd.DataFrame(columns=self.shared_q_table.columns, dtype=np.float64)

        for agent in self.agents:
            if isinstance(agent, QLearningAgent):
                agent.q_table = self.shared_q_table

    def create_agents(self, agent_counts):
        # 如果 agent_counts 是 numpy 数组，则转换为字典
        if isinstance(agent_counts, np.ndarray):
            agent_counts = {
                'Random': int(agent_counts[0]),
                'Cheater': int(agent_counts[1]),
                'Cooperator': int(agent_counts[2]),
                'Copycat': int(agent_counts[3]),
                'Grudger': int(agent_counts[4]),
                'Detective': int(agent_counts[5]),
                'AI': int(agent_counts[6]),
                'Human': int(agent_counts[7])
            }

        agents = []
        id_counter = 0

        for stereotype, count in agent_counts.items():
            for _ in range(count):
                if stereotype == 'AI':
                    # 根据 opt.ai_type 来区分创建 Q-learning 或 DQN 代理
                    if opt.ai_type == 'Q':
                        agent = QLearningAgent(
                            id=id_counter,
                            actions=actionlist,
                            shared_q_table=self.shared_q_table,  # 传入共享的 Q 表
                            learning_rate=opt.lr,  # 使用全局学习率
                            reward_decay=opt.gamma,  # 使用全局折扣因子
                            e_greedy=opt.epsilon  # 使用全局探索率
                        )
                    elif opt.ai_type == 'DQN':
                        agent = DQNAgent(
                            id=id_counter,
                            actions=actionlist,
                            learning_rate=opt.lr,
                            gamma=opt.gamma,
                            epsilon=opt.epsilon,
                            epsilon_min=opt.epsilon_min,
                            epsilon_decay=opt.epsilon_decay,
                            batch_size=opt.batch_size,
                            memory_size=opt.memory_size,
                            target_update=opt.target_update,
                            state_size=opt.state_size
                        )
                    else:
                        # 如果未指定具体的AI类型，则退回创建通用Agent
                        agent = Agent(id=id_counter, stereotype=stereotype)
                else:
                    # 对于其他代理类型，直接创建 Agent
                    agent = Agent(id=id_counter, stereotype=stereotype)
                agents.append(agent)
                id_counter += 1
        return agents

    def trainAgent(self, initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                        extrinsic_reward, DesignerEpochID, save_model):

        # initialize
        if not self.agents:
            self.reset_shared_q_table()
            self.agents = self.create_agents(initial_agent_counts)

        record_data = True

        # 外层循环：进行 num_episodes 次游戏（Epoch）
        for epoch_id in range(opt.agent_train_epoch):
            if printGame:
                print(f"========== Epoch {epoch_id + 1} 开始 ==========")

            self.env.setup(self.agents, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward)
            self.env.oneCompetition(epoch_id, agent_type_names, agent_types_order, record_data, save_model, printGame)

            if printGame:
                print(f"========== Epoch {epoch_id + 1} 结束 ==========")

        # self.env.plot_results()  # 训练结束后绘制图表
        # self.env.save_results_to_csv()  # 保存结果为 CSV 文件

        save_dir = os.path.join('data', agent_type_names['AI'])
        os.makedirs(save_dir, exist_ok=True)
        fileName = os.path.join(save_dir, f"{DesignerEpochID}.csv")
        self.env.save_results_to_csv(fileName)

        # 所有 Epoch 结束后的统计
        print("所有游戏（Epoch）结束，统计最终代理的种类及数量：")
        agent_type_counts = {}
        for agent in self.agents:
            agent_type = agent.stereotype
            agent_type_counts[agent_type] = agent_type_counts.get(agent_type, 0) + 1

        for agent_type in agent_types_order:
            count = agent_type_counts.get(agent_type, 0)
            agent_name = agent_type_names.get(agent_type, f"Type {agent_type}")
            print(f"{agent_name}: 数量={count}")

    def testAgent(self, initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                  test_epoch_number=1):
        """
        测试AI代理在给定交易规则下的表现，并统计每一类及总体的：
          - gini_coefficient,
          - cooperation_rate,
          - individual_income
        返回一个包含上述3个指标的字典，每个指标为长度为9的列表，
        前8项对应各类型（顺序见 agent_types_order），最后一项为总体结果。
        本函数通过多次调用 oneCompetition 后，将每次的结果累加再平均。
        """
        # 初始化累加器（长度为9）
        accum_gini = np.zeros(9)
        accum_coop = np.zeros(9)
        accum_income = np.zeros(9)

        # 循环运行多个 epoch，并累计各指标
        for epoch_id in range(test_epoch_number):
            # 这里假设 oneCompetition 返回一个字典，统计当前 epoch 的指标
            results = self.env.oneCompetition(1, agent_type_names, agent_types_order, False, False, printGame)
            # results['gini_coefficient'], results['cooperation_rate'], results['individual_income'] 均为长度为9的列表
            accum_gini += np.array(results['gini_coefficient'])
            accum_coop += np.array(results['cooperation_rate'])
            accum_income += np.array(results['individual_income'])

        # 求各指标的平均值
        final_gini = (accum_gini / test_epoch_number).tolist()
        final_coop = (accum_coop / test_epoch_number).tolist()
        final_income = (accum_income / test_epoch_number).tolist()

        return {
            'gini_coefficient': final_gini,
            'cooperation_rate': final_coop,
            'individual_income': final_income
        }

    # def testAgent(self, initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
    #               test_epoch_number = 1):
    #     """
    #     测试AI代理在给定交易规则下的表现。
    #
    #     参数:
    #         tradeRules (dict): 定义交易规则的字典。
    #         test_epoch_number (int): 测试的回合数，默认为100。
    #
    #     返回:
    #         dict: 包含合作率、最终代理比例、平均金币数、最高金币数和最低金币数的字典。
    #     """
    #     # 初始化统计数据记录
    #     cooperation_rates = []
    #     individual_income = []
    #     gini_coefficient = []
    #     record_data = False
    #     save_model = False
    #
    #     # 设置AI代理的策略参数，例如降低探索率以更多利用已学策略
    #     for agent in self.agents:
    #         if isinstance(agent, QLearningAgent) or isinstance(agent, DQNAgent):
    #             agent.set_greedy(0.01)  # 设置较低的探索率
    #
    #
    #     for epoch_id in range(test_epoch_number):
    #         extrinsic_reward = [0,0]
    #         self.env.setup(self.agents, trade_rules, round_number, reproduction_number, mistake_possibility,
    #                        extrinsic_reward)
    #         self.env.oneCompetition(1, agent_type_names, agent_types_order, record_data, save_model, printGame)
    #
    #         individualIncomeTemp = sum(self.env.IndividualIncome[-5:])/5
    #         individual_income.append(individualIncomeTemp)
    #
    #         # 统计合作次数和交易次数
    #         epoch_cooperations = sum(agent.cooperation_count for agent in self.agents)
    #         epoch_trades = sum(agent.tradeNum for agent in self.agents)
    #         cooperation_rates.append(epoch_cooperations / epoch_trades if epoch_trades > 0 else 0)
    #
    #
    #         # if individualIncomeTemp == max(individual_income):
    #         #     save_dir = os.path.join('data', 'DQN')
    #         #     os.makedirs(save_dir, exist_ok=True)
    #         #     fileName = os.path.join(save_dir, f"{EpochID}.csv")
    #         #     self.env.save_results_to_csv(fileName)
    #
    #         # # 统计合作次数和交易次数
    #         # epoch_cooperations = sum(agent.cooperation_count for agent in self.agents)
    #         # epoch_trades = sum(agent.tradeNum for agent in self.agents)
    #         # cooperation_rates.append(epoch_cooperations / epoch_trades if epoch_trades > 0 else 0)
    #         #
    #         # # 记录每个代理的金币数
    #         # money_distribution = {agent.id: agent.money for agent in self.agents}
    #         # money_distributions.append(money_distribution)
    #         #
    #         # # 累计总合作次数和总交易次数
    #         # total_cooperations += epoch_cooperations
    #         # total_trades += epoch_trades
    #     individual_income_result = sum(individual_income)/len(individual_income)
    #     cooperation_rates_result = sum(cooperation_rates)/len(cooperation_rates)
    #
    #     # 计算 Gini 系数
    #     incomes = np.array([agent.money for agent in self.agents])
    #     incomes_sorted = np.sort(incomes)
    #     n = len(incomes_sorted)
    #     cumulative_incomes = np.cumsum(incomes_sorted)
    #     gini_numerator = np.sum((2 * np.arange(1, n + 1) - n - 1) * incomes_sorted)
    #     gini_denominator = n * cumulative_incomes[-1]
    #     gini_coefficient_result = gini_numerator / gini_denominator if gini_denominator != 0 else 0
    #
    #     # 计算整体合作率
    #     # cooperation_rate = total_cooperations / total_trades if total_trades > 0 else 0
    #
    #     # # 计算最终代理比例
    #     # for agent in self.agents:
    #     #     agent_type = agent.stereotype
    #     #     final_proportion[agent_type] = final_proportion.get(agent_type, 0) + 1
    #     #
    #     # for agent_type in final_proportion:
    #     #     final_proportion[agent_type] = (final_proportion[agent_type] / len(self.agents)) * 100
    #     #
    #     # # 计算其他统计指标
    #     # final_money = {agent.id: agent.money for agent in self.agents}
    #     # average_money = np.mean(list(final_money.values()))
    #     # max_money = np.max(list(final_money.values()))
    #     # min_money = np.min(list(final_money.values()))
    #     #
    #     # # 打印测试结果
    #     # print(f"测试回合数: {testEpisodeNum}")
    #     # print(f"整体合作率: {cooperation_rate * 100:.2f}%")
    #     # print("最终代理比例:")
    #     # for agent_type, proportion in final_proportion.items():
    #     #     print(f"{agent_type}: {proportion:.2f}%")
    #     # print(f"平均金币数: {average_money:.2f}")
    #     # print(f"最高金币数: {max_money}")
    #     # print(f"最低金币数: {min_money}")
    #
    #     # # 记录测试结果到日志
    #     # logging.info(f"测试回合数: {testEpisodeNum}")
    #     # logging.info(f"整体合作率: {cooperation_rate * 100:.2f}%")
    #     # logging.info(f"最终代理比例: {final_proportion}")
    #     # logging.info(f"平均金币数: {average_money:.2f}, 最高金币数: {max_money}, 最低金币数: {min_money}")
    #
    #     # 可选：绘制测试结果图表
    #     # self.plot_test_results(cooperation_rates, money_distributions)
    #
    #
    #     # 返回测试结果
    #     # return {
    #     #     'cooperation_rate': cooperation_rate,
    #     #     'final_proportion': final_proportion,
    #     #     'average_money': average_money,
    #     #     'max_money': max_money,
    #     #     'min_money': min_money,
    #     #     'cooperation_rates': cooperation_rates,
    #     #     'money_distributions': money_distributions
    #     # }
    #     return gini_coefficient_result

    def forward(self, initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                extrinsic_reward, DesignerEpochID,
                save_model):

        # initialize
        self.agents = []
        self.env.setup(self.agents, trade_rules, round_number, reproduction_number, mistake_possibility, extrinsic_reward)

        # train
        self.trainAgent(initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                        extrinsic_reward, DesignerEpochID, save_model)

        # test
        Evaluation = self.testAgent(initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                                          test_epoch_number= 10)
        for agent in self.agents:
            if isinstance(agent, QLearningAgent):
                plot_q_table(agent.q_table, output_path="C:/Users/hilab/OneDrive/Desktop/Rule_Generation/WebDash/data/q_table_heatmap.png")
                break
        return Evaluation


def save_beta_results_to_csv(filename='BETA.csv', epoch_list=[], BetaList=[]):
    """
    将 epoch_list 和 dqn_total_money_div2_list 保存为 CSV 文件。

    参数:
        filename (str): 保存的文件名，默认为 'dqn_money_over_epochs.csv'
    """
    data = {
        'Epoch': epoch_list,
        'Beta': BetaList
    }
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"结果已保存到 {filename}")


def get_fixed_rules_vector(opt):
    return np.array([
        opt.random_count,        # rules[0]: Number of Random Action agents
        opt.cheater_count,       # rules[1]: Number of Always Cheat agents
        opt.cooperator_count,    # rules[2]: Number of Always Cooperate agents
        opt.copycat_count,       # rules[3]: Number of Copycat agents
        opt.grudger_count,       # rules[4]: Number of Grudger agents
        opt.detective_count,     # rules[5]: Number of Detective agents
        opt.ai_count,            # rules[6]: Number of AI agents
        opt.human_count,         # rules[7]: Number of Human agents

        opt.trade_rules[0],      # rules[8]: Payoff for both agents cheating
        opt.trade_rules[1],      # rules[9]: Payoff for the other agent when one cheats and the other cooperates
        opt.trade_rules[2],      # rules[10]: Payoff for agent A when A cheats and B cooperates
        opt.trade_rules[3],      # rules[11]: Payoff for agent B when B cheats and A cooperates
        opt.trade_rules[4],      # rules[12]: Payoff for both agents cooperating
        opt.trade_rules[5],      # rules[13]: Payoff for both agents cooperating (second dimension)

        opt.round_number,        # rules[14]: Number of rounds in each competition
        opt.reproduction_number, # rules[15]: Reproduction number of top players each round
        opt.mistake_possibility  # rules[16]: Mistake probability
    ])

def rule_translation(rule_vector):
    # 提取前 8 个值，代表不同类型代理的数量
    total_agents = 25
    initial_counts_raw = rule_vector[:8]

    # 将这 8 个值归一化，使它们的和为 1
    normalized_counts = initial_counts_raw / np.sum(initial_counts_raw)

    # 将归一化后的比例乘以总数量，并四舍五入为整数
    initial_counts = np.round(normalized_counts * total_agents).astype(int)

    # 确保总和为 total_agents，如果不为 total_agents，进行调整
    current_total = np.sum(initial_counts)
    if current_total != total_agents:
        diff = total_agents - current_total
        # 根据差值进行调整，随机选择一个索引来增加或减少
        while diff != 0:
            index = np.random.choice(8)
            if diff > 0:
                initial_counts[index] += 1
                diff -= 1
            elif diff < 0 and initial_counts[index] > 0:
                initial_counts[index] -= 1
                diff += 1

    # 提取支付矩阵部分
    payoff_raw = rule_vector[8:14]
    min_payoff = -5
    max_payoff = 5
    # 线性映射到期望范围
    payoff_matrix = payoff_raw * (max_payoff - min_payoff) + min_payoff

    # 提取轮数部分并映射为整数
    min_round = 5
    max_round = 20
    round_number = int(rule_vector[14] * (max_round - min_round) + min_round)

    # 提取复制比例部分并映射为整数
    min_rate = 1
    max_rate = 10
    reproduction_rate = int(rule_vector[15] * (max_rate - min_rate) + min_rate)

    # 提取犯错概率并映射为期望范围
    max_probability = 0.5
    mistake_probability = rule_vector[16] * max_probability

    return initial_counts, payoff_matrix, round_number, reproduction_rate, mistake_probability










#########################################################


# Loss function
MSELoss = torch.nn.MSELoss()

# Initialize generator and discriminator
ruleDesigner = RuleDesigner()
evaluator = Evaluator()
environment = Environment()


if cuda:
    ruleDesigner.cuda()
    evaluator.cuda()
    environment.cuda()
    MSELoss.cuda()
    device = torch.device('cuda')

print(cuda)


# Optimizers
optimizer_Designer = torch.optim.Adam(ruleDesigner.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))   # betas=(opt.b1, opt.b2) ???
optimizer_Evaluator = torch.optim.Adam(evaluator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor



save_model = False


reward_cooperation_list = []
reward_cheat_list = []

epoch_list = []

# 定义代理类型编号与名称的映射
agent_type_names = {
    'Random': 'Random',
    'Cheater': 'Cheater',
    'Cooperator': 'Cooperator',
    'Copycat': 'Copycat',
    'Grudger': 'Grudger',
    'Detective': 'Detective',
    'AI': opt.ai_type,  # 假设 opt.ai_type 为 'AI' 或实际的类型名称
    'Human': 'Human'
}

# 定义代理类型顺序
agent_types_order = ['Random', 'Cheater', 'Cooperator', 'Copycat', 'Grudger', 'Detective', 'AI', 'Human']



# -----------------
#  Train Designer
# -----------------
printSER = False
printGame = True

for DE_epoch_id in range(opt.DE_train_episode):

    if DE_epoch_id == opt.DE_train_episode - 1:
        save_model = True

    optimizer_Designer.zero_grad()
    # Sample noise as generator input
    noise_std = 0.005
    evaluation_requirement = torch.normal(
        mean=opt.giniCoefficient,
        std=noise_std,
        size=(opt.batch_size, opt.evaluationSize),
        device=device
    )

    # Generate a batch of agents
    rule_vector = ruleDesigner(evaluation_requirement)      # inistates and reward
    loss_g = MSELoss(evaluator(rule_vector), evaluation_requirement)
    loss_g.backward()
    optimizer_Designer.step()

    optimizer_Evaluator.zero_grad()

    # start of one batch
    for i in range(opt.batch_size):
        #  Rule translation
        rules = rule_vector[i].detach().cpu().numpy()


        if opt.fixed_rule:
            initial_agent_counts = {
                'Random': opt.random_count,
                'Cheater': opt.cheater_count,
                'Cooperator': opt.cooperator_count,
                'Copycat': opt.copycat_count,
                'Grudger': opt.grudger_count,
                'Detective': opt.detective_count,
                opt.ai_type: opt.ai_count,
                'Human': opt.human_count
            }
            initial_agent_counts = np.array(list(initial_agent_counts.values()))
            trade_rules = opt.trade_rules
            round_number = opt.round_number
            reproduction_number = opt.reproduction_number
            mistake_possibility = opt.mistake_possibility
        else:
            initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility = rule_translation(
                rules)
        if publish:
            # 构造更新数据字典
            update_data = {
                "epoch": DE_epoch_id,
                "initial_agent_counts": initial_agent_counts.tolist() if isinstance(initial_agent_counts, np.ndarray) else initial_agent_counts,
                "trade_rules": trade_rules.tolist() if isinstance(trade_rules, np.ndarray) else trade_rules,
                "round_number": round_number,
                "reproduction_number": reproduction_number,
                "mistake_possibility": mistake_possibility
            }
            # 发布更新数据
            publish_excel_update(update_data)

        # extrinsic_reward = (rules - 0.5) * 10
        extrinsic_reward = [0,0]

        # One Game
        evaluation_e_temp = environment(initial_agent_counts, trade_rules, round_number, reproduction_number, mistake_possibility,
                    extrinsic_reward, DE_epoch_id,
                    save_model)

        save_test_results_to_excel(evaluation_e_temp, DE_epoch_id)

        evaluation_e_temp= np.expand_dims(evaluation_e_temp, axis=0)

        # 确保 gini_coefficient_e 是数组（长度至少为1）
        if i == 0:
            gini_coefficient_e = np.array([evaluation_e_temp[0]['gini_coefficient'][-1]])
        else:
            gini_coefficient_e = np.append(gini_coefficient_e, evaluation_e_temp[0]['gini_coefficient'][-1])
        # end of one batch

        environment_evaluation_result = Variable(Tensor(gini_coefficient_e), requires_grad=False)
        environment_evaluation_result.to(device)



    loss_d = MSELoss(evaluator(rule_vector.detach()), environment_evaluation_result)
    loss_d.backward()
    optimizer_Evaluator.step()

    average_extrinsic_reward = (rule_vector.mean(dim=0) - 0.5) * 10

    reward_cooperation_list.append(average_extrinsic_reward[0].cpu().item())
    reward_cheat_list.append(average_extrinsic_reward[1].cpu().item())
    epoch_list.append(DE_epoch_id + 1)


    if save_model:
        PATH = "C:/Users/hilab/OneDrive/Desktop/Rule_Generation/PythonCode/variableMaze_Jiyao/designer/designer.pth"
        torch.save(ruleDesigner.state_dict(), PATH)
    if printSER:
        print('----- SER training, Epoch ID: ', DE_epoch_id, ' -----')
        print('  loss_g: ', loss_g,'  loss_d: ', loss_d)
        print('  Gini Coefficient: ', environment_evaluation_result)
        print('  Expectation: ', evaluation_requirement.squeeze())

save_beta_results_to_csv('BETA_A.csv', epoch_list, reward_cooperation_list)
save_beta_results_to_csv('BETA_B.csv', epoch_list, reward_cheat_list)
print("=========done========")

# 在代码末尾添加
duration = 1000  # 持续时间，毫秒
freq = 440  # 频率，赫兹
winsound.Beep(freq, duration)


