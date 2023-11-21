import matplotlib.pyplot as plt
import pandas as pd
import scienceplots


class Ploter:

    @staticmethod
    def plot(y, x=None, title=None, ltitle=None, xlabel=None, ylabel=None, save_path=None, dpi=300,
             show=False,
             size=None, style='science'):
        """science 风格绘制折线图

        Args:
            y (dict): 数据字典
            x (list): X 轴数据
            title (str): 图标题
            ltitle (str): 图例标题
            xlabel (str): X 轴标题
            ylabel (str): Y 轴标题
            save_path (str): 保存路径
            dpi (int): 图片分辨率
            show (bool): 是否显示
            size (tuple): 图片大小
            style (str): 风格
        """
        # 绘制
        with plt.style.context([style, 'grid', 'no-latex']):
            fig, ax = plt.subplots()
            if size:
                fig.set_size_inches(size)
            if x:
                for k, v in y.items():
                    ax.plot(x, v, label=k)
            else:
                for k, v in y.items():
                    ax.plot(v, label=k)
            if title:
                plt.title(title)  # 图标题
            if ltitle:
                ax.legend(title=ltitle)  # 图例标题
            if xlabel:
                ax.set(xlabel=xlabel)  # X 轴标题
            if ylabel:
                ax.set(ylabel=ylabel)  # Y 轴标题
            # ax.autoscale(tight=True)
            # ax.set_ylim(top=3) # 设置 Y 刻度最大值
            if save_path:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()

    @staticmethod
    def plot_predict(data, save_path="result.png"):
        """ 绘制压差预测对比图

        X 轴为 7 个房间；
        Y 轴为压差值

        Args:
            data (dict): 数据字典
            save_path: 保存路径
        """
        with plt.style.context(['science', 'grid', 'no-latex']):
            x_tick = ['RM1', 'RM2', 'RM3', 'RM4', 'RM5', 'RM6', 'RM7']
            fig, ax = plt.subplots()
            ax.plot(x_tick, data.get('Prediction'), label="Prediction")
            ax.plot(x_tick, data.get('BIM'), linestyle='--', label="BIM")
            plt.title("Result")  # 图标题
            ax.legend()  # 图例标题
            ax.set(xlabel="Room")  # X 轴标题
            ax.set(ylabel="Room Pressure (Pa)")  # Y 轴标题
            # ax.autoscale(tight=True)
            # ax.set_ylim(top=3) # 设置 Y 刻度最大值
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

    @staticmethod
    def plot_pres_per_rooms(data, save_path="result.png"):
        with plt.style.context(['science', 'grid', 'no-latex']):
            x_tick = ['RM1', 'RM2', 'RM3', 'RM4', 'RM5', 'RM6', 'RM7']
            fig, ax = plt.subplots()
            ax.plot(x_tick, data.get('ref_press'), linestyle='--', label="ref_press")
            ax.plot(x_tick, data.get('opt_press'), label="opt_press")
            plt.title("Optimal Result")  # 图标题
            ax.legend()  # 图例标题
            ax.set(xlabel="Room")  # X 轴标题
            ax.set(ylabel="Room Pressure (Pa)")  # Y 轴标题
            # ax.autoscale(tight=True)
            # ax.set_ylim(top=3) # 设置 Y 刻度最大值
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

    @staticmethod
    def plot_trace(data, save_path="trace.png"):
        """science 风格绘制折线图 -> 迭代过程

        Args:
            data (dict): 数据字典
            save_path (str): 保存路径
        """
        with plt.style.context(['science', 'grid', 'no-latex']):
            fig, ax = plt.subplots()
            ax.plot(data.get("f_best"), color='red', label="Best of Generation")  # 最优值
            ax.plot(data.get("f_avg"), color='blue', label="Average of Generation")  # 平均值
            plt.title("Iterative Trace")  # 图标题
            ax.legend()  # 图例标题
            ax.set(xlabel="Generation")  # X 轴标题
            ax.set(ylabel="RMSE(Pressure) (Pa)")  # Y 轴标题
            # ax.autoscale(tight=True)
            # ax.set_ylim(top=3) # 设置 Y 刻度最大值
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()

    @staticmethod
    def plot_loss(data, save_path="loss.png"):
        """science 风格绘制折线图 -> 训练损失

        Args:
            data (dict): 数据字典
            save_path (str): 保存路径
        """
        with plt.style.context(['science', 'grid', 'no-latex']):
            fig, ax = plt.subplots()
            ax.plot(data.get("val/loss"), label="val/loss")  # 最优值
            ax.plot(data.get("train/loss"), linestyle='-.', label="train/loss")  # 平均值
            plt.title("Train Loss")  # 图标题
            ax.legend()  # 图例标题
            ax.set(xlabel="Epoch")  # X 轴标题
            ax.set(ylabel="Loss(L1) (Pa)")  # Y 轴标题
            # ax.autoscale(tight=True)
            # ax.set_ylim(top=3) # 设置 Y 刻度最大值
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()


if __name__ == '__main__':
    # trace = { 'f_best': [24.718328267471456, 17.272902091833707, 17.272902091833707,
    # 15.636878889505505, 15.636878889505505, 15.632683509412503, 15.461599524946982,
    # 15.461599524946982, 15.379440227695637, 14.834903209811541, 14.834903209811541,
    # 12.717855938791955, 12.717855938791955, 7.770304874427957, 5.981298380728292], 'f_avg': [
    # 24.718328267471456, 23.26271801984379, 24.77141706303868, 23.23647678504599,
    # 21.283859684203485, 17.110401157495552, 16.679099842244206, 15.924775013504192,
    # 15.559912492361425, 15.385647694683225, 15.253525276972166, 14.910383838417342,
    # 14.585333003709106, 13.678843255547537, 11.812604084894327]}
    # result = { 'ref_press': [3, 4, 2, 1, 3, 2, 1], 'opt_press': [1, 2, 3, 4, 5, 6, 7] }
    # Ploter.plot_trace(trace)
    # Ploter.plot_result(result)
    df_1 = pd.read_csv(
        r"C:\Users\KAI\Desktop\run-NN_mlp-L12_BS128_LR0.001_EP1000_2023-05-27T11-33-13-tag-val_room_pres_mae.csv")
    df_2 = pd.read_csv(
        r"C:\Users\KAI\Desktop\run-NN_mlp-L12_resume_BS128_LR0.001_EP2000_2023-05-28T15-32-41-tag-val_room_pres_mae.csv")
    df = pd.concat([df_1, df_2], ignore_index=True)  # 合并两个 DataFrame
    df.sort_values(by="Step", inplace=True, ascending=True)
    loss = {
        'val/pres_mae': df["Value"].tolist()
    }
    Ploter.plot_loss(loss)
