class ResultsVisualizer:
    def visualize(self, results):
        # 使用matplotlib或其他库对结果进行可视化
        plt.figure()
        # 示例：展示Accuracy的柱状图
        results['Accuracy'].plot(kind='bar')
        plt.show()
