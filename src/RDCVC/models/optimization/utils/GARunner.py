import geatpy as ea
from geatpy import Algorithm, Population, Problem

from CoreGA.utils.Args import Arguments


class GARunner:
    def __init__(
        self,
        args: Arguments,
        problem: Problem,
        population: Population,
        algorithm: Algorithm,
    ) -> None:
        self.args = args
        self.problem = problem
        self.population = population
        self.algorithm = algorithm

    def run(self):
        print("Algorithm initialized. Start evolution...")
        save_path = self.args.save_dir

        # ------------------------- 种群进化 ------------------------- #
        result = ea.optimize(
            self.algorithm,
            # seed=1,  # 随机数种子
            prophet=None,  # 先验知识
            verbose=True,  # 是否打印输出日志信息
            drawing=1,  # 绘图方式
            outputMsg=True,
            drawLog=True,
            saveFlag=True,
            dirName=save_path,
        )
        print("Evolution finished.")
        return result
