def generate_samples(start, end, m, n, refs: list = None, ref_num=0,
                     data_type='int'):
    """ 生成采样数据

    采样点数必须多于基准点数。

    可以指定基准点，也可以指定基准点数，使用采样点周围的正态分布生成采样数据。
        若指定基准点，则在基准点附近生成采样点，默认的基准采样范围为 [start, end]/2。
            步骤：
            1. r = (end - start)/2
            2. 在距离每个基准点 ±r 的范围内，按照正态分布随机取 n/ref_num 个整数，共计取 n 个采样点为一组。
            3. 打乱这组采样点的顺序。
            4. 采样 m 组
        若指定基准点个数，则在范围 [start, end] 内选取 ref_num 个基准点。
            步骤：
            1. 在 [start, end] 内等距平均取 r_num 个基准点，基准点的采样距离 r = (b-a)/(s+1)。
            2. 在距离每个基准点 ±r 的范围内，按照正态分布随机取 n/ref_num 个整数，共计取 n 个采样点为一组。
            3. 打乱这组采样点的顺序。
            4. 采样 m 组

    Args：
        start: int, 范围的起始点
        end: int, 范围的终止点
        m: int, 采样组数
        n: int, 每组采样点的个数
        refs: list, 基准点
        ref_num: int, 基准点的个数
        data_type: str, 采样数据的类型，'int' 或 'float'

    Returns:
        samples: list, 采样数据
    """
    r = 0  # 基准点的采样距离（正态分布 3σ取值范围）
    # 检查传入参数
    if refs is None and ref_num == 0:
        raise ValueError('必须提供基准点或基准点数')
    if refs is not None and ref_num != 0:
        raise ValueError('基准点和基准点数不能同时提供')
    # 若提供了基准点
    if refs is not None:
        # 检查基准点数是否多于采样点数
        if ref_num > n:
            raise ValueError('基准点数必须少于采样点数')
        # 检查基准点是否在范围内
        for ref in refs:
            if ref < start or ref > end:
                raise ValueError('基准点必须在范围内')
        ref_num = len(refs)
        r = (end - start) / 2
    # 若提供了基准点数
    if ref_num != 0:
        # 检查基准点数是否多于采样点数
        if ref_num > n:
            raise ValueError('基准点数必须少于采样点数')
        # 计算基准点之间距离
        r = (end - start) / (ref_num + 1)
        # 等距平均取基准点
        refs = np.linspace(start + r, end - r, num=ref_num)

    # 上述检查提供了 refs 和 ref_num 两个参数
    # 检查通过后，为下面的代码提供 refs(基准点)、ref_num(基准点数) 和 r(基准点的采样距离)
    """ 生成采样数据 """
    samples = []
    for i in range(m):
        # 生成一组采样点
        group = []
        for ref in refs:
            # 以基准点为中心，随机取数
            samples_per_ref = np.random.normal(ref, r / 3,
                                               size=int(n / ref_num))
            # 若采样点超出范围，则超范围的采样点取边界值
            samples_per_ref = np.clip(samples_per_ref, start, end)
            # 将采样点加入该组，检查数据类型
            if data_type == 'int':
                group.extend(samples_per_ref.astype(int))
            elif data_type == 'float':
                group.extend(samples_per_ref)
        # 若该组采样点不足 n 个，则在整体范围内随机取数补足。若该组采样点多于 n 个，则随机去掉多余的采样点。
        if len(group) < n:
            if data_type == 'int':
                group.extend(
                    np.random.randint(start, end, size=n - len(group)))
            elif data_type == 'float':
                group.extend(
                    np.random.uniform(start, end, size=n - len(group)))
        elif len(group) > n:
            group = group[:n]
        # 将改组采样点打乱顺序
        np.random.shuffle(group)
        # 将该组采样点加入总列表
        samples.append(group)

    return samples
