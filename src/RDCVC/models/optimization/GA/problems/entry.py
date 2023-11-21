from geatpy import Problem, crtfld, Population

from CoreGA.utils.Args import Arguments, problems_map, algorithms_map


def prepare_problem(args: Arguments):
    return problems_map[args.problem](args)


def prepare_population(args: Arguments, problem: Problem):
    Encoding = "RI"  # 编码方式
    Field = crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)

    return Population(Encoding, Field, args.nind)


def prepare_algorithm(args: Arguments, problem: Problem, population: Population):
    algorithm = algorithms_map[args.algorithm](
        problem,
        population,
        MAXGEN=args.maxgen,  # 最大进化代数
        MAXTIME=args.maxtime,  # 最大运行时间 (s)
        logTras=1,  # 设置每隔多少代记录日志，若设置成 0 则表示不记录日志
        verbose=True,  # 设置是否打印输出日志信息
        drawing=1,  # 设置绘图方式
        aimFuncTrace=True,  # 设置是否记录目标函数值的变化
    )
    # algorithm.outFunc = None  # 设置每次进化记录的输出函数

    return algorithm
