"""
从 0 开始，每次加 9，取模 90，重复 n 次。
检查是否能够全覆盖 0~90 中的所有数字。
"""
num = 11
is_success = False
while not is_success:
    print("-----------------------")
    print(f"num = {num}")
    print("-----------------------")
    aim = []

    # run 90 次
    for i in range(100):
        aim.append(i * num % 90)

    # 检查是否能够全覆盖 0~90 中的所有数字
    is_success = True
    for i in range(90):
        if i not in aim:
            is_success = False
            print(f"i = {i} not in aim")
            break

    if is_success:
        print(f"Success!champion is {num}")
        break
    num += 2
    # 超时退出
    if num > 90:
        break
